# Dataset investigation

This document records which data sources were considered, what was verified about each, and which
quantities in this project are observed, derived or simulated.

## 1. What the project needs

| Need | Requirement | Why |
|---|---|---|
| Network | Stops, routes, trips, stop times | Graph model, running times, the existing schedule |
| Demand level | A complete time series of passengers per route and hour | Forecasting ground truth, optimisation input |
| Demand shape | Boardings and alightings by stop | Peak load per boarding, direction split, simulation |
| Covariates | Weather, calendar | Forecast features |
| Travel times | Observed running times | Travel-time uncertainty |

A GTFS feed alone supplies only the first row. It contains no passenger counts.

## 2. Sources considered

| Source | Observed demand? | Granularity | Verdict |
|---|---|---|---|
| **MTA Bus Hourly Ridership** (NY Open Data `gxb3-akrn`, `kv7t-n8in`) | Yes: fare payments | Route x hour, 2020 to present | **Used: demand level** |
| **MTA Bus Stop Level Ridership** (NY Open Data `fvdm-uavx`) | Yes: automatic passenger counters | Route x direction x stop x hour, Sep 2024 to present | **Used: demand shape only** |
| **MTA NYCT GTFS static, Manhattan** | No | Stop times | **Used: network and existing schedule** |
| **Open-Meteo archive** (ERA5-based reanalysis) | n/a | Hourly weather | **Used: forecast covariate** |
| CTA bus ridership (Chicago data portal) | Yes | Route x **day** only | Rejected: no hourly resolution |
| Pittsburgh Regional Transit open data | Partly | Monthly aggregates; raw APC on request only | Rejected: not reproducible |
| "BostonBusEquity" (Hugging Face) | Derived | Third-party derivative of MBTA data | Rejected: unclear provenance |
| MTA GTFS-Realtime | Vehicle positions | Live feed only; no public archive was located | Not used: no reproducible history |
| OpenStreetMap road network | No | Road geometry | Not needed: GTFS stop sequences suffice for frequency setting |
| Traffic speed datasets | No | — | Not used |

## 3. Chosen data and what was verified

Primary network: **MTA New York City Transit buses in Manhattan** (routes with prefix `M`).

### 3.1 GTFS static feed

- URL: `https://rrgtfsfeeds.s3.amazonaws.com/gtfs_m.zip` (9.5 MB), `feed_version` 20260813, valid
  2026-09-06 to 2027-01-02, retrieved 2026-10-10.
- 44 `M` routes with trips. 41 have fare-ridership data; 31 exceed 2,000 mean daily boardings and form
  the planning instance.
- Day types are resolved through `calendar.txt` and `calendar_dates.txt` on three reference dates with
  no calendar exception: weekday 2026-09-15, Saturday 2026-09-12, Sunday 2026-09-13.
- **Version mismatch (limitation):** the feed describes the autumn 2026 timetable, while demand runs
  from January 2025 to August 2026. MTA publishes only the current feed at this URL. The "existing
  schedule" is therefore the timetable that followed the demand period, not the one in force during
  it. Routes present in ridership data but absent from the feed (`M100`, `M90`) are excluded.

### 3.2 Fare-payment ridership (demand level)

- Dataset `gxb3-akrn`, "MTA Bus Hourly Ridership: Beginning 2025". Columns: `transit_timestamp`
  (local time, floored to the hour), `bus_route`, `payment_method`, `fare_class_category`,
  `ridership`, `transfers`.
- Downloaded with a server-side aggregation (sum over payment method and fare class) for
  2025-01-01 to 2026-08-31: 671,232 route-hour rows, a complete grid with no missing cells.
- **What it measures:** OMNY and MetroCard entries. Passengers who board without paying are not
  counted. It is treated throughout as *fare-paying demand*, a lower bound on boardings. No
  correction for unpaid boardings is applied, because no route-level estimate was verified here.
  Demand-scaling experiments (section E5 of the results) show how conclusions change if true demand
  is higher.
- `ridership` is used as the total. The dataset description does not state whether `transfers` is a
  subset of `ridership`; the subway dataset with the same schema defines it as a subset, and that
  reading is assumed here. `transfers` is about 22.5% of `ridership`.
- **Reporting outage:** all Manhattan routes report zero ridership on 2026-01-10 to 2026-01-15. These
  six days are set to missing, never imputed, and excluded from training and evaluation.

### 3.3 APC stop-level counts (demand shape)

- Dataset `fvdm-uavx`. Columns: `date`, `hour`, `route_id`, `direction` (N/S/E/W), `stop_id`,
  `stop_sequence`, `boardings`, `alightings`, `trips`.
- Aggregated server-side over the **training window only** (2025-01-01 to 2026-02-28) by route,
  direction, stop, day of week and hour of day: 607,337 rows. Test-period APC data never enters a
  profile.
- **Why it is not used for demand levels.** The ratio of APC boardings to fare payments over the
  training window ranges from 0.24 to 4.05 across the 41 routes (median 0.99). A complete count
  that includes unpaid boardings would sit consistently above 1. The spread shows that coverage of
  APC-equipped trips is partial and uneven, and the dataset documents no expansion factor. Only
  ratios within a route are taken from it: stop boarding and alighting shares, the direction split
  and the resulting peak-load coefficient.
- APC compass directions are matched to GTFS `direction_id` by stop-set overlap. On average 95.6% of
  a route's APC boardings fall on stops of the dominant GTFS pattern (minimum 82%); the remainder,
  on variant patterns, is dropped.
- Hour-level profiles with fewer than 300 APC boardings fall back to the route's day-type profile
  (7.2% of route-direction-hours in service hours).

### 3.4 Weather

- Open-Meteo historical archive at 40.78 N, 73.97 W: hourly temperature, precipitation, snowfall and
  wind speed, 14,592 hours.
- Weather for the target hour is observed weather, which a real forecaster would not have. Models
  that use it are labelled "weather oracle" and reported as an optimistic bound.

## 4. Licences and terms

| Source | Terms found | Consequence for this repository |
|---|---|---|
| NY Open Data (both ridership tables) | Portal metadata lists no licence. The Open NY Terms of Use grant a non-exclusive, revocable licence to download and use content, with no warranty. | Raw and processed data are **not committed**. Download scripts and SHA-256 hashes are. |
| MTA GTFS | MTA developer terms: feeds are free to use, provided "as is"; data must not be presented in a misleading way; MTA branding needs a separate licence. | Feed is downloaded at run time, not redistributed. No MTA branding is used. |
| Open-Meteo | CC BY 4.0 (attribution). | Attribution here and in the README. |

These terms were read from the portals in October 2026. They were not reviewed by a lawyer; anyone
redistributing derived data should check them again.

## 5. Observed, derived, simulated

| Quantity | Status |
|---|---|
| Route-hour boardings (fare-paying) | **Observed** |
| Stop boarding and alighting shares, direction split | **Observed** (APC sample, training window) |
| Stop-level hourly demand | **Derived**: observed route totals disaggregated by observed shares |
| Peak-load coefficient | **Derived** from APC shares |
| Scheduled running times, trips per hour | **Observed** (GTFS schedule, not measured operations) |
| Cycle times | **Derived**: scheduled running time plus assumed layover |
| Vehicle capacities, loading standard | **Assumed** (configuration) |
| Cost per vehicle-hour | **Assumed** (configuration) |
| Passenger arrival instants | **Simulated** (Poisson within the hour) |
| Running-time variation, dwell, bunching | **Simulated** |
| Breakdowns, congestion, disruptions, demand spikes | **Simulated** (synthetic stress scenarios) |

No experiment presents simulated or synthetic demand as observed. The synthetic toy network in
`busopt.synthetic` is used only by tests, CI and the offline demo, and labels its output.

## 6. Reproducing the downloads

```bash
uv run busopt download   # about 20 MB in total; writes data/raw/PROVENANCE.json
uv run busopt build      # processed tables and the network validation report
```

`PROVENANCE.json` records the query, retrieval time, row count and SHA-256 of each file. The ridership
portals are updated in place, so a later download can differ; the hashes identify the version used
for the reported results.
