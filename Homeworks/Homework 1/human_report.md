# ASHRAE Great Energy Predictor III

## Preprocessing

Careful performing of this process was inspired by three prior write-ups:

- a data-cleaning methodology from a top ASHRAE solution ([1](#references))
- an anomaly-correction approach from a BirdCLEF competition ([2](#references))
- site and building identification via internet search ([3](#references))

As a reslut, for Handout validation it seemed like boost, but for 4-fold seasonal validation it seems like slight change([mlp.ipynb, Findings](models/mlp.ipynb#follow-up)).

### Local time zone correction

As noted in the [competition forum](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/115698), weather
timestamps are UTC while meter timestamps are local. We confirmed it on our data: after shifting weather by each site's UTC offset, temperature peaks at 14:00–16:00 at every site. The exception is site 14, whose meters are in UTC too ([EDA §6.1](eda/EDA.ipynb#temporal)). The fix improved the LightGBM baseline from 1.290 to 1.286 ([EDA §9](eda/EDA.ipynb#broken)).

### Remove anomalies

Some rows had broken energy use, like 0 energy use for some period, spikes, constanst electricity uses for some period of time, etc.

The decision was to remove zero electricity rows, zero heating rows for cold periods and zero chilled water during warm periods. Other anomalies do not matter so much. ([EDA §9](eda/EDA.ipynb#broken))

Together they flag 599,086 rows, 3.0% of the training data ([data_aggregation.ipynb §4](preprocessing/data_aggregation.ipynb)).

We drop them from training data only, while validate on all, because kaggle test contains pathological data as well.
([data_aggregation.ipynb §4](preprocessing/data_aggregation.ipynb)).

With a LightGBM baseline, dropping zero electricity alone improved RMSLE on clean rows from 1.140 to 1.061, and both rules together to 1.048 ([EDA §9](eda/EDA.ipynb#broken)).

### Removed features

Some features just were not appropriate for training model:

- floor count which is missing for 83% of rows ([data_aggregation.ipynb §3](preprocessing/data_aggregation.ipynb))
- precipitation and cloud coverage, which barely relate to the target and are often missing ([EDA §7](eda/EDA.ipynb))

### Interpolated features

- year_built, filled with the training median, plus a missing flag: it is missing for 60% of rows ([data_aggregation.ipynb §3](preprocessing/data_aggregation.ipynb), [mlp.ipynb, Features](models/mlp.ipynb)).
- air / dew temperature and sea level pressure were interpolated liearly over time
- wind direction represented as sin/cos, than interpolated linearly ([mlp.ipynb, Features](models/mlp.ipynb))

### Feature Engineering

- Cyclic encoding of hour and day of week (sin/cos), from local time ([mlp.ipynb, Features](models/mlp.ipynb)).
- One-hot `meter` and `primary_use` ([ashrae/features.py](ashrae/features.py)).
- Building identity, as either a learned 16-number vector per building (embedding) or the mean `log1p` reading of each building x meter series ([mlp.ipynb, Models](models/mlp.ipynb)).

### Target transformations

- RMSLE is the RMSE of `log1p` values, so we predict `log1p(meter_reading)` with an MSE loss, which optimizes the metric
directly ([EDA §8](eda/EDA.ipynb#metric)). Moreover, log1p of meter reading has bell shape, which generally good sign.
- Site 0 electricity is converted from kBTU to kWh for training (908,409 rows), and back for the submission ([data_aggregation.ipynb §1](preprocessing/data_aggregation.ipynb)).

### Adversarial validation

Adversarial validation found no feature to drop. A LightGBM classifier tells 2016 from 2017–2018 with AUC 0.75, but
only 0.52 without weather, and each weather feature alone scores 0.51–0.54.

Together the weather features effectively act like a timestamp fingerprint, so this leak is not actually exploitable.

## EDA and Metric analysis

The metric measures relative error: a 10% miss costs about 0.095 at any scale. It is harsh near zero, though: predicting 5 for a true 0 costs 1.79 ([EDA §8](eda/EDA.ipynb#metric)).

## Validation Strategy

### Handout approach (failed)

First split trained on Jan–Sep and validated on Oct–Dec ([mlp.ipynb, Data](models/mlp.ipynb)). the problem was that adversarial validation showed that these the two parts are told apart with AUC 0.92, all from weather features ([adversarial.ipynb §2](validation/adversarial.ipynb)).

### Seasonal 4-Fold

From then on we used four folds: each holds out one 3-month block of 2016 and trains on the others ([adversarial.ipynb, How to use this](validation/adversarial.ipynb); folds are defined in [ashrae/config.py](ashrae/config.py#L19)).

Each fold is scored twice: on all rows, as Kaggle scores, and on clean rows without anomalies, which measures the error on real consumption ([season_validation.ipynb](validation/season_validation.ipynb)). This explained the large spread between seasons ([season_validation.ipynb, Findings](validation/season_validation.ipynb); [mean_vs_embedding.ipynb](validation/mean_vs_embedding.ipynb) for the embedding model).

## Fails

- He initialization made no difference
- Model regularization (except Batch Norm) did not help
- MLP models and EDA where evaluated using Handout validation, while adv val showed that Jan-Mar period makes huge difference, so we reevaluated and reanalyzed the whole project.

## References

1. 1st Place Solution in "ASHRAE Great Energy Predictor III" by Isamu & Matt — <https://www.kaggle.com/competitions/ashrae-energy-prediction/writeups/isamu-matt-1st-place-solution-team-isamu-matt>
2. "1st place solution: Correct Data is All You Need" by Volodymyr, "BirdCLEF 2023" competition — <https://www.kaggle.com/competitions/birdclef-2023/writeups/volodymyr-1st-place-solution-correct-data-is-all-y>
3. "Sites, buildings identified by internet search" for "ASHRAE Great Energy Predictor III" by Poe Dator — <https://www.kaggle.com/c/ashrae-energy-prediction/discussion/112841>
