"""A compact temporal neural forecaster (GRU over daily demand vectors).

Justification for including it: the panel has 41 routes x 20 months of hourly counts with strong
weekly structure, which is the regime where gradient-boosted trees on lag features usually win.
The GRU tests whether a sequence model that sees the raw last 14 days, rather than hand-built lags,
changes that conclusion. It is small enough (about 60k parameters) to train on CPU in about a minute,
so no GPU is required; ``kaggle/train_gru_sweep.py`` holds a larger sweep for GPU runs.

Protocol matches the tree model: fit on train, choose the epoch count on validation, refit on
train+validation, predict test. Inputs end at day D - h, so no future demand is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from busopt.forecast.features import holiday_flags, wide

SEQ_DAYS = 14


def _daily_tensor(panel: pd.DataFrame, train_end: str):
    y = wide(panel)
    routes = list(y.columns)
    days = pd.DatetimeIndex(sorted(set(y.index.normalize())))
    arr = y.to_numpy().reshape(len(days), 24, len(routes)).transpose(2, 0, 1)  # [route, day, hour]
    tr = days <= pd.Timestamp(train_end)
    scale = np.nanmean(arr[:, tr, :], axis=(1, 2))  # per-route mean hourly demand, training window only
    dt = np.where(days.dayofweek < 5, 0, np.where(days.dayofweek == 5, 1, 2))
    prof = np.stack(
        [np.nanmean(arr[:, tr & (dt == k), :], axis=1) for k in range(3)], axis=1
    )  # [route, 3, 24]
    return arr, routes, days, scale, dt, prof


class GRUForecaster:
    name = "gru"

    def __init__(
        self, horizon_days: int, seed: int = 17, hidden: int = 64, max_epochs: int = 40, lr: float = 2e-3
    ):
        self.h, self.seed, self.hidden, self.max_epochs, self.lr = horizon_days, seed, hidden, max_epochs, lr
        self.best_epoch: int | None = None

    def _samples(self, arr, days, scale, dt, prof, day_mask):
        hol = holiday_flags(days).astype(np.float32)
        n_routes = arr.shape[0]
        filled = np.where(np.isnan(arr), prof[:, dt, :], arr) / scale[:, None, None]
        miss = np.isnan(arr).any(axis=2).astype(np.float32)
        xs, cs, ys, rs, idx = [], [], [], [], []
        for d in np.nonzero(day_mask)[0]:
            end = d - self.h  # last observed day
            if end - SEQ_DAYS + 1 < 0:
                continue
            sl = slice(end - SEQ_DAYS + 1, end + 1)
            dow_seq = np.eye(7, dtype=np.float32)[days[sl].dayofweek]
            seq_extra = np.concatenate([dow_seq, hol[sl, None]], axis=1)
            cal = np.concatenate(
                [
                    np.eye(7, dtype=np.float32)[days[d].dayofweek],
                    [hol[d]],
                    [
                        np.sin(2 * np.pi * days[d].dayofyear / 365.25),
                        np.cos(2 * np.pi * days[d].dayofyear / 365.25),
                    ],
                ]
            )
            for r in range(n_routes):
                target = arr[r, d] / scale[r]
                xs.append(np.concatenate([filled[r, sl], seq_extra, miss[r, sl, None]], axis=1))
                cs.append(cal)
                ys.append(target)
                rs.append(r)
                idx.append((r, d))
        return (
            np.asarray(xs, np.float32),
            np.asarray(cs, np.float32),
            np.asarray(ys, np.float32),
            np.asarray(rs, np.int64),
            np.asarray(idx),
        )

    def _net(self, n_routes: int):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(s, hidden):
                super().__init__()
                s.emb = nn.Embedding(n_routes, 8)
                s.gru = nn.GRU(24 + 9, hidden, batch_first=True)
                s.head = nn.Sequential(nn.Linear(hidden + 10 + 8 + 24, 128), nn.ReLU(), nn.Linear(128, 24))

            def forward(s, x, c, r):
                _, hN = s.gru(x)
                recent = x[:, -7:, :24].mean(dim=1)  # skip connection: last-week mean profile
                return torch.relu(s.head(torch.cat([hN[-1], c, s.emb(r), recent], dim=1)) + recent)

        return Net(self.hidden)

    def _train(self, data, epochs: int, valid=None):
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        x, c, y, r, _ = data
        ok = ~np.isnan(y).any(axis=1)
        x, c, y, r = (torch.from_numpy(a[ok]) for a in (x, c, y, r))
        net = self._net(self.n_routes)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.SmoothL1Loss(beta=0.1)
        g = torch.Generator().manual_seed(self.seed)
        curve = []
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(x), generator=g)
            for i in range(0, len(x), 256):
                b = perm[i : i + 256]
                opt.zero_grad()
                loss = loss_fn(net(x[b], c[b], r[b]), y[b])
                loss.backward()
                opt.step()
            if valid is not None:
                curve.append(self._mae(net, valid))
        return net, curve

    def _mae(self, net, data) -> float:
        p, y, sc = self._predict(net, data)
        m = ~np.isnan(y)
        return float(np.abs(p - y)[m].mean())

    def _predict(self, net, data):
        import torch

        x, c, y, r, _ = data
        net.eval()
        with torch.no_grad():
            p = net(torch.from_numpy(x), torch.from_numpy(c), torch.from_numpy(r)).numpy()
        sc = self.scale[r][:, None]
        return p * sc, y * sc, sc

    def fit_predict(self, panel: pd.DataFrame, train_end: str, valid_end: str) -> pd.DataFrame:
        """Return long test predictions (ts, route_id, gru)."""
        arr, routes, days, scale, dt, prof = _daily_tensor(panel, train_end)
        self.scale, self.n_routes = scale, len(routes)
        tr = days <= pd.Timestamp(train_end)
        va = (days > pd.Timestamp(train_end)) & (days <= pd.Timestamp(valid_end))
        te = days > pd.Timestamp(valid_end)
        d_tr, d_va, d_te = (self._samples(arr, days, scale, dt, prof, m) for m in (tr, va, te))
        _, curve = self._train(d_tr, self.max_epochs, valid=d_va)
        self.best_epoch = int(np.argmin(curve)) + 1
        self.valid_curve = curve
        both = tuple(np.concatenate([a, b]) for a, b in zip(d_tr, d_va, strict=True))
        net, _ = self._train(both, self.best_epoch)
        p, _, _ = self._predict(net, d_te)
        idx = d_te[4]
        rows = pd.DataFrame(
            {
                "route_id": np.repeat(np.asarray(routes)[idx[:, 0]], 24),
                "ts": (
                    days[idx[:, 1]].to_numpy()[:, None] + (np.arange(24) * np.timedelta64(1, "h"))[None, :]
                ).ravel(),
                self.name: p.ravel(),
            }
        )
        return rows
