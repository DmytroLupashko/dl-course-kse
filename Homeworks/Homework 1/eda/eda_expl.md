# EDA: what each step showed

Task: predict hourly energy use of ~1,450 buildings at 16 sites; metric RMSLE. Each section: what we found → what it
changes. Section numbers match [EDA.ipynb](EDA.ipynb).

## Data overview (§2)

- 2016, hourly: 20.2M readings from 2,380 building × meter series.
- 875 of 1,449 buildings have one meter, almost always electricity (60% of rows).
- → Keep one row per building × meter × hour; a column per meter type would be mostly empty.

## Data quality (§3)

- Missing: `floor_count` 76% and `year_built` 53% of buildings; cloud coverage 49% and precipitation 36% of weather
  rows. Site 5 never reports pressure. Up to 330 hours missing at a site. No duplicates.
- → Fill weather gaps within each site; use other sites only when a site has no data for a column.

## Building metadata (§4)

- Education (549) and office (279) dominate. Floor area is bell-shaped on a log scale.
- Missing `year_built` / `floor_count` depends on the site: whole sites report none.
- → Use `log1p(square_feet)`.

## Target (§5)

- Extremely skewed: one steam meter (building 1099) holds 78% of all energy. Well-behaved on a log scale.
- Zeros: 27% of hot water, 16% of chilled water, 13% of steam, 4% of electricity readings.
- → Target is `log1p(meter_reading)`.

## Temporal patterns (§6)

- Chilled water peaks in summer, steam and hot water in winter. Use is lower at night and at weekends.
- → Hour and day-of-week features; the model must combine meter type with weather.

## Time zones (§6.1)

- Weather timestamps are UTC, meter timestamps local (site 14: UTC too). Checked on our data: after shifting, the
  temperature peaks at 14:00–16:00 at every site.
- → Shift weather by the site's UTC offset before merging (not at site 14). LightGBM, 4-fold average: 1.290 → 1.286
  (clean rows 1.147 → 1.140); no fold gets worse.

## Features vs target (§7)

- Floor area is the strongest building feature (Spearman ρ = 0.68 for electricity).
- Temperature: chilled water up (ρ = 0.43), steam and hot water down (ρ ≈ −0.45).
- → Drop precipitation and cloud coverage: barely related to the target, often missing.

## Metric (§8)

- RMSLE = RMSE of `log1p` values → train on `log1p` with MSE loss.
- Relative error: a 10% miss costs ≈ 0.095 at any scale. Harsh near zero: predicting 5 for a true 0 costs 1.79.
- → A few wrong zeros can dominate the score. Also track RMSLE per meter and without anomaly rows.

## Anomalies (§9)

- Zero electricity = broken meter or data gap. Heating at zero for a week in the cold, or cooling in the heat, too.
- → Drop them from training only. LightGBM, 4-fold average on clean rows: 1.140 → 1.061 without zero electricity,
  → 1.048 with both rules, better in every fold. Shorter cut-offs work better: even a few hours of zeros teach the
  model that a building can use no power.
- On all rows the average barely moves (1.286 → 1.281): Jul–Dec improves, Jan–Jun gets worse, because there the
  broken meters stay broken into the validation months. Oct–Dec alone (1.383 → 1.168) overstated the gain.
- Anomaly rows stay in validation and cost up to 0.48 RMSLE per fold
  ([season_validation.ipynb](../validation/season_validation.ipynb)).

## What didn't work

- Two more anomaly rules: stuck meters (same non-zero value for 48+ hours) and spikes (> 10× the daily median). On
  top of both rules they change the 4-fold average by 0.003 at most; the spike rule mostly flagged normal on/off
  cycling. Both left out.

## What was removed from the notebook

- Summary tables (per-column statistics, per-meter percentiles), the `year_built` histogram, a box plot by building
  use (boxes overlapped too much), and per-site tables already shown by the time-zone plot.
