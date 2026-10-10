# Mathematical methodology

Scope: **service-frequency setting with fleet allocation** on a fixed route network, for hourly
periods of one service day. Route design, timetabling to the minute, vehicle blocking, crew
scheduling and transfer synchronisation are out of scope (section 9).

## 1. Network model

The network is a directed multigraph $G=(V,E)$ built from GTFS (`busopt.network.graph`).

- $V$: stops of the dominant stop pattern of each route and direction (1,815 stops).
- Ride edges $(u,v,r,d)$: consecutive stops of route $r$, direction $d$, with scheduled running time
  $\tau_{uv}$ (median over trips of the pattern) and great-circle length (2,488 edges).
- Transfer edges: stops within 150 m that serve different route sets, walk time at 1.2 m/s
  (3,150 edges).

Validation (`busopt.network.validate`) checks positive running times, stop-to-stop speeds within
2 to 80 km/h, monotone cumulative times, two directions per route, connectivity of the stop graph and
of the route-transfer graph, and that the terminals of opposite directions lie within 1 km so that a
round trip closes. On the Manhattan feed every check passes except one edge at 71.7 km/h, measured
against a great-circle distance on an express segment.

The transfer graph is used for validation and visualisation. The optimisation model does not route
passengers through transfers.

**Assumptions.** (A1) Each route-direction is represented by its most frequent weekday stop pattern;
short-turn and branch variants are folded into it (the dominant pattern carries 90% of trips on
average, 48% at worst on M1). (A2) Distances are great-circle between stops, used only for
validation.

## 2. Notation

| Symbol | Meaning | Source |
|---|---|---|
| $r\in R$ | routes (31 in the instance) | GTFS |
| $t\in T$ | hourly periods 05:00 to 23:00 ($\lvert T\rvert=19$) | config |
| $d_{rt}$ | boardings on $r$ during $t$, both directions | observed, or forecast |
| $\sigma_{rt}^{\delta}$ | share of boardings in direction $\delta$ | APC |
| $\rho_{rt}^{\delta}$ | peak load per boarding in direction $\delta$ | APC |
| $c_{rt}=\max_\delta \rho_{rt}^{\delta}\sigma_{rt}^{\delta}$ | peak-load coefficient | derived |
| $\tau_{rt}$ | round-trip cycle time, minutes | GTFS plus layover |
| $C_r$ | loading standard, passengers per bus | assumed |
| $F$ | fleet size | from the existing schedule |
| $B$ | vehicle-hour budget | from the existing schedule |
| $\underline{k},\overline{k}$ | minimum and maximum buses per hour (2 and 30) | policy |
| $a_{rt}\in\{0,1\}$ | route operated in period $t$ | GTFS |

**Peak load.** With normalised stop boarding shares $b_s$ and alighting shares $e_s$ along a
direction, the load per boarding after stop $s$ is $L_s=\sum_{u\le s}(b_u-e_u)$ and
$\rho=\max_s L_s$. The hourly passenger flow through the maximum load point is $c_{rt}d_{rt}$.

**Cycle time.** $\tau_{rt}=T^0_{rt}+T^1_{rt}+\max\{0.1\,(T^0_{rt}+T^1_{rt}),\,6\}$, with $T^\delta_{rt}$
the median scheduled one-way running time of trips starting in hour $t$.

## 3. Deterministic model

**Decision.** $x_{rt}\in\{\underline{k},\dots,\overline{k}\}$: buses per hour per direction on route
$r$ in period $t$ (the same in both directions).

**Waiting time.** For passengers arriving at random and headway $h=60/x$ with coefficient of
variation $\mathrm{cv}$, the mean wait is $\mathbb{E}[W]=\tfrac{h}{2}(1+\mathrm{cv}^2)$. The planning
model uses $\mathrm{cv}=0$: $w(x)=30/x$ minutes.

**Vehicles.** Holding $x$ buses per hour on a round trip of $\tau$ minutes needs
$n(x,\tau)=\lceil x\tau/60\rceil$ vehicles.

**Passengers above the loading standard.** $o_{rt}=\max\{0,\;c_{rt}d_{rt}-C_r x_{rt}\}$.

$$
\begin{aligned}
\min_{x}\quad & \sum_{r,t} a_{rt}\Big[d_{rt}\,\frac{30}{x_{rt}} + p\,\max\{0,\,c_{rt}d_{rt}-C_r x_{rt}\}\Big] \\
\text{s.t.}\quad & \sum_{r} a_{rt}\Big\lceil \frac{x_{rt}\tau_{rt}}{60}\Big\rceil \le F \qquad \forall t \quad\text{(fleet)}\\
& \sum_{r,t} a_{rt}\Big\lceil \frac{x_{rt}\tau_{rt}}{60}\Big\rceil \le B \quad\text{(vehicle-hours)}\\
& x_{rt}\in\{\underline{k},\dots,\overline{k}\}\quad\forall r,t .
\end{aligned}
$$

$p=30$ minutes is the penalty per passenger above the standard. The capacity constraint is soft so
that the model stays feasible when demand exceeds what the fleet can carry and reports the excess
instead of failing.

### 3.1 Exact linearisation

The objective is non-linear in $x$ and the vehicle requirement contains a ceiling. Because $x$ is
discrete, both become data. Introduce $y_{rtk}\in\{0,1\}$, equal to 1 when $x_{rt}=k$:

$$
\begin{aligned}
\min\quad & \sum_{r,t,k} \gamma_{rtk}\,y_{rtk}\\
\text{s.t.}\quad & \sum_k y_{rtk}=1 \quad \forall (r,t): a_{rt}=1\\
& \sum_r\sum_k \nu_{rtk}\,y_{rtk}\le F \quad\forall t,\qquad \sum_{r,t,k}\nu_{rtk}\,y_{rtk}\le B,
\end{aligned}
$$

with $\gamma_{rtk}=d_{rt}\,30/k+p\max\{0,c_{rt}d_{rt}-C_rk\}$ and $\nu_{rtk}=\lceil k\tau_{rt}/60\rceil$.

This is a multiple-choice knapsack problem with $\lvert T\rvert+1$ coupling constraints. It is an
exact reformulation, not a piecewise-linear approximation. The weekday instance has 17,023 binaries
and 607 constraints. The problem class is NP-hard, but these instances solve to proven optimality in
about one second (section 8).

### 3.2 Variants

- **Weighted sum.** Add $\lambda\sum \nu_{rtk}y_{rtk}$ to the objective, with $\lambda$ in
  passenger-minutes per vehicle-hour, and relax $B$. Sweeping $\lambda$ traces the supported points of
  the cost-service frontier.
- **$\varepsilon$-constraint.** Sweep $B$ (and $F$). This also reaches unsupported Pareto points.
- **LP relaxation.** $y\in[0,1]$ gives a lower bound and shadow prices of fleet and budget. The
  integer model has no exact duals; the relaxation's duals are reported with that caveat.

## 4. Baseline policies

All baselines receive the same $F$ and $B$ as the optimiser.

| Policy | Rule |
|---|---|
| Existing schedule | GTFS trips per hour, mean of the two directions, rounded, clipped to $[\underline{k},\overline{k}]$ |
| Fixed frequency | One frequency per route all day (its day-average existing level), scaled to fit |
| Load-proportional | $x_{rt}\propto c_{rt}d_{rt}/C_r$ (equal load factor) |
| Square-root rule | $x_{rt}\propto\sqrt{d_{rt}/\tau_{rt}}$ |
| Greedy | From $\underline{k}$, repeatedly add the bus/hour with the largest cost reduction per vehicle-hour |
| Greedy + local search | Pairwise exchange of one bus/hour between cells |

Rule-based policies are scaled by the largest multiplier that keeps the rounded plan feasible, then
topped up in the rule's own priority order until no cell can be raised.

**Why the square-root rule is a strong baseline.** Relax integrality and the ceiling, keep only the
budget constraint, and ignore capacity: $\min\sum d_{rt}30/x_{rt}$ s.t. $\sum x_{rt}\tau_{rt}/60\le B$.
The Lagrangian condition $-30d_{rt}/x_{rt}^2+\mu\tau_{rt}/60=0$ gives $x_{rt}\propto\sqrt{d_{rt}/\tau_{rt}}$.
The rule is the continuous optimum of a relaxation of the model. The MILP can beat it only through
integrality, the per-period fleet constraint and capacity.

**The existing schedule is a projection.** The real timetable has different frequencies by
direction, mid-hour changes, short turns and interlined blocks. The baseline is its image in the
model's decision space. $F$ and $B$ are defined as what that image uses, so the comparison is
resource-neutral inside the model. It is not a claim about the resources MTA actually deploys.

## 5. Uncertainty-aware models

### 5.1 Scenarios

Let $\hat d$ be the day-ahead forecast. Scenarios are $d^{s}=\hat d\odot \kappa^{s}$, where
$\kappa^{s}\in\mathbb{R}^{R\times T}$ is the matrix of ratios actual/forecast of one validation day of
the same day type, from the model trained on the training window only (out-of-sample errors).
Drawing whole days keeps the correlation of errors across routes and hours. Ratios use an additive
floor of 20 boardings and are clipped to $[0.2,5]$.

### 5.2 Two-stage stochastic program (SAA)

First stage: $x$. Recourse: passengers above the standard in scenario $s$. With equiprobable
scenarios,

$$
\min_x\; \sum_{r,t} a_{rt}\Big[\bar d_{rt}\frac{30}{x_{rt}} + \frac{p}{S}\sum_{s}\max\{0,\,c_{rt}d^{s}_{rt}-C_rx_{rt}\}\Big]
\quad\text{s.t. fleet and budget.}
$$

Two structural facts:

1. **The waiting term is linear in demand**, so its expectation depends only on $\bar d$. Demand
   uncertainty affects the optimal plan only through the convex overload term.
2. **The recourse is separable by $(r,t)$**, so its sample average can be folded into the cost
   coefficients: $\gamma_{rtk}=\bar d_{rt}30/k+\tfrac{p}{S}\sum_s\max\{0,c_{rt}d^s_{rt}-C_rk\}$. The SAA
   has the same variables and constraints as the deterministic model, for any number of scenarios.

A corollary of (1): where the loading standard is far from binding, the stochastic and deterministic
solutions coincide up to the difference between $\bar d$ and $\hat d$. Section 5 of the results
confirms this at observed demand.

### 5.3 CVaR extension

To penalise the tail of total overload $O^s(x)=\sum_{r,t}o^s_{rt}$, add
$\beta\,p\,\mathrm{CVaR}_\alpha(O)$ with the Rockafellar-Uryasev linearisation:

$$
\mathrm{CVaR}_\alpha(O)=\min_{\eta}\;\eta+\frac{1}{(1-\alpha)S}\sum_s z_s,\qquad
z_s\ge \sum_{r,t}o^s_{rt}-\eta,\quad o^s_{rt}\ge c_{rt}d^s_{rt}-C_r\sum_k k\,y_{rtk},\quad z,o,\eta\ge 0 .
$$

This couples scenarios and needs explicit recourse variables (only for cells where overload is
possible at $\underline{k}$).

### 5.4 Quantile-robust plan

Solve the deterministic model with $d_{rt}$ replaced by the cell-wise $q$-quantile of the scenarios
($q=0.9$). It is conservative cell by cell and ignores that not all cells peak together. It also
distorts the waiting term, which the stochastic model does not.

### 5.5 Running-time slack

Solve the deterministic model with $\tau_{rt}$ replaced by $(1+s)\tau_{rt}$, $s=0.1$. The plan
reserves vehicles so that the planned frequency survives a 10% increase in running time.

### 5.6 Delivered frequency under disruption

If running times are multiplied by $m_{rt}$ and $\ell_{rt}$ vehicles are lost, the vehicles allocated
by a plan deliver

$$x^{\text{eff}}_{rt}=\min\Big\{x_{rt},\;\frac{60\,(n_{rt}-\ell_{rt})^+}{\tau_{rt}m_{rt}}\Big\},\qquad n_{rt}=\Big\lceil \frac{x_{rt}\tau^{\text{plan}}_{rt}}{60}\Big\rceil .$$

Ceiling slack and planned slack absorb part of a disruption before service degrades.

## 6. Forecasting

**Task.** For target day $D$ and horizon $h\in\{1,7,14\}$ days, forecast all 24 hourly boardings of
every route using information dated $D-h$ or earlier.

**Leak prevention.** All demand features are lags of at least $24h$ hours, or rolling statistics of a
series already shifted by $24h$ hours. A test perturbs demand after the forecast origin and asserts
that no feature of the target day changes, for every horizon. Static profiles and scaling constants
use the training window only. APC profiles use the training window only.

**Split.** Train to 2026-02-28, validation 2026-03-01 to 2026-05-31, test 2026-06-01 to 2026-08-31
(92 days). Learned models are tuned on validation, refit on train plus validation with the tuned
round or epoch count, then scored once on test.

**Models.**

| Model | Definition |
|---|---|
| Seasonal naive | Same hour on the most recent admissible same weekday ($7\lceil h/7\rceil$ days back) |
| Historical average | Training-window mean by route, day type, hour (static) |
| Rolling average | Mean of the four most recent admissible same-weekday values |
| Ridge | Per route: lags, rolling means, holiday flags, 72 day-type x hour dummies |
| Gradient-boosted trees | One global model, Poisson loss, route as categorical |
| GRU | Last 14 daily 24-vectors, route embedding, target-day calendar, residual on last-week mean |

Holidays (US federal calendar) are the only special events the data supports. Weather variants use
observed target-hour weather and are labelled as an oracle.

**Metrics.** MAE, RMSE, WAPE $=\sum\lvert e\rvert/\sum y$ and bias, on service hours, by horizon, day
type, weekday peak and holidays.

## 7. Simulation

The simulator (`busopt.sim.simulator`) is the evaluation referee. It shares no code with the
optimiser and does not assume $w=30/x$.

- **Passengers.** For route $r$, direction $\delta$, hour $t$, stop $s$: $N\sim\text{Poisson}(d_{rt}\sigma^\delta_{rt}b_s)$
  arrivals, uniform within the hour. Each passenger is simulated individually.
- **Time frame.** Demand of hour $t$ at stop $s$ is placed in the window during which buses dispatched
  in hour $t$ pass $s$ (the hour shifted by the scheduled running time to $s$). Without this, long
  routes would show artificial waits at the start and end of the day.
- **Dispatch.** Even headways $60/x_{rt}$ with a random phase, plus Gaussian dispatch jitter (30 s).
- **Running time.** Scheduled segment time times a per-trip lognormal factor (unit mean, cv 0.10)
  times per-segment Gaussian noise. Buses cannot overtake.
- **Dwell.** 2.5 s per boarding or alighting passenger. Together with running-time noise this
  produces bunching.
- **Alighting.** A passenger on board alights at stop $s$ with probability
  $e_s/\sum_{u<s}(b_u-e_u)$, which reproduces the APC alighting profile in expectation.
- **Capacity.** First-come-first-served boarding up to crush capacity (85 standard, 115
  articulated, assumed). Passengers left behind wait for the next bus.
- **Common random numbers.** One passenger draw per day is reused for every policy.

**Metrics.** Mean, median, 90th and 95th percentile wait (10-second bins); share served within 10
minutes; unmet demand (left behind at the end of service, split into capacity and end-of-service);
denied-boarding events; overcrowded passenger-minutes (load above the loading standard times time);
load utilisation (passenger-time over capacity-time); maximum load factor; trips operated.

**Cross-check.** With all noise switched off, simulated mean wait on the weekday instance is 4.482
minutes against 4.481 from the formula. A test asserts agreement within 3% on the toy network.

## 8. Solvers and verification

| Solver | Version | Licence | Role |
|---|---|---|---|
| HiGHS | 1.12.0 (bundled with OR-Tools 9.15) | MIT | Default MILP |
| SCIP | 10.0.0 | Apache 2.0 | Cross-check |
| CBC | 2.10.12 | EPL 2.0 | Cross-check |
| CP-SAT | 9.15 | Apache 2.0 | Cross-check |
| GLOP | 9.15 | Apache 2.0 | LP relaxation, shadow prices |

No commercial solver is used. Settings: relative MIP gap $10^{-4}$, time limit 60 s, 2 threads.

**Independent verification.** `busopt.opt.problem.evaluate` recomputes feasibility (fleet per hour,
budget, bounds, integrality) and the objective directly from the frequency matrix, with no solver
code. `verify` compares it with the solver's objective (relative tolerance $10^{-4}$). Tests compare
all four solvers with exhaustive enumeration on a 2x2 instance (625 plans), for the deterministic
and the scenario model.

## 9. Limitations of the mathematical model

1. **Fare-paying demand.** Unpaid boardings are not observed. Loads are understated by an unknown,
   route-dependent factor.
2. **Inelastic demand.** $d_{rt}$ does not respond to frequency. Real ridership rises with service,
   so benefits of adding service are understated and those of cutting it overstated.
3. **Waiting time only.** In-vehicle time, transfers, walking and reliability are not in the
   objective. The model moves buses toward short, busy routes because that is where a bus removes the
   most waiting per vehicle-hour.
4. **Hourly stationarity.** Demand and headways are constant within the hour. Within-hour peaks, which
   drive real crowding, are averaged out.
5. **Symmetric frequencies.** Both directions get the same frequency. The real timetable does not.
6. **Fleet model.** $\sum_r\lceil x\tau/60\rceil\le F$ per hour ignores deadheading, pull-out and
   pull-in, interlining across routes, depots, charging or fuelling and driver rules. It also allows
   the fleet to be reassigned freely between hours.
7. **Fleet and budget are inferred** from the projected GTFS schedule, not from MTA fleet records.
8. **Capacities, the loading standard, the overload penalty and the cost rate are assumptions.**
9. **Single dominant pattern** per route-direction.
10. **Scenario model.** Multiplicative day-resampled errors from three months of validation data: 65
    weekdays, 13 Saturdays, 14 Sundays. Tail events absent from that window are not represented.
11. **Schedule vintage.** The GTFS feed postdates the demand period.
12. **Simulator.** Running-time noise, dwell and dispatch parameters are not calibrated to observed
    vehicle data, since no archive of observed running times was available.
