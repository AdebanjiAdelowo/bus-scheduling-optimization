"""Minimal experiment tracking: one directory per run with a manifest and metrics.

A manifest records what is needed to reproduce a number: Git commit, configuration, seeds, package
and solver versions, hardware and data provenance hashes.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from busopt.config import REPO_ROOT, resolve

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "networkx", "ortools", "numba", "torch", "anthropic"]


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def environment() -> dict[str, Any]:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
    }


class Run:
    def __init__(self, name: str, cfg: dict, out_root: str | Path = "results/runs"):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.dir = resolve(out_root) / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.manifest: dict[str, Any] = {
            "name": name,
            "started_utc": stamp,
            "git_commit": git_commit(),
            "config": cfg,
            "environment": environment(),
            "metrics": {},
        }
        prov = resolve(cfg["data"]["raw_dir"]) / "PROVENANCE.json"
        if prov.exists():
            self.manifest["data_provenance"] = json.loads(prov.read_text())
        self._flush()

    def log(self, **metrics: Any) -> None:
        self.manifest["metrics"].update(metrics)
        self._flush()

    def finish(self) -> Path:
        self.manifest["elapsed_s"] = round(time.time() - self.t0, 2)
        self._flush()
        return self.dir

    def _flush(self) -> None:
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))
