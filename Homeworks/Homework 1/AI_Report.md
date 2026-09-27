# AI Report — ASHRAE Great Energy Predictor III

We predict hourly energy use of ~1,450 buildings at 16 sites for 4 meter types, scored with RMSLE
([EDA.ipynb](eda/EDA.ipynb)). Our final model is a small MLP with one output per meter type and a learned vector per
building. It scores 1.200 against 1.280 for LightGBM ([mlp.ipynb](models/mlp.ipynb)). Every score in this report is
an average over four seasonal folds of 2016 ([Validation](#validation)), and links to the notebook it comes from.

## Preprocessing

### Remove anomalies

Some readings don't reflect energy use, and training on them teaches the model that a building can use no power
([EDA §9](eda/EDA.ipynb#broken)). We flag two kinds:

- **Zero electricity:** every electricity reading of exactly 0. A building always draws some power, so a zero is a
  broken meter or a data gap ([EDA §9](eda/EDA.ipynb#broken)).
- **Off when needed:** steam or hot water at 0 for at least a week while the week's mean temperature is below 10 °C,
  and chilled water the same above 20 °C ([EDA §9](eda/EDA.ipynb#broken)).

Together they flag 599,086 rows, 3.0% of the training data ([data_aggregation.ipynb §4](preprocessing/data_aggregation.ipynb)).
We drop them from training only, and keep them when scoring validation, since Kaggle scores every test row
([data_aggregation.ipynb §4](preprocessing/data_aggregation.ipynb)).

With a LightGBM baseline, dropping them makes the model better at real consumption in every fold: on clean rows
RMSLE goes from 1.140 to 1.061 without zero electricity, and to 1.048 with both rules ([EDA §9](eda/EDA.ipynb#broken)).
Shorter cut-offs worked better: requiring a zero run of 30 days gave 1.095, a week 1.075, a day 1.067, and dropping
every zero 1.061, so even a few hours of zeros hurt ([EDA §9](eda/EDA.ipynb#broken)).

On all rows, as Kaggle scores, the average barely moves: 1.286 → 1.281. Jul–Sep and Oct–Dec improve (Oct–Dec
1.383 → 1.168), but Jan–Mar and Apr–Jun get worse (1.363 → 1.485, 1.187 → 1.301). Those folds have the most anomalies
in validation, and they still improve on clean rows. So the loss comes from the anomaly rows: a model trained on a
broken meter's zeros predicts low values, and that pays off when the outage continues into the validation months
([EDA §9](eda/EDA.ipynb#broken)). We keep dropping anomalies. Whether 2016's outages continue into the test years
can only be checked with a leaderboard score.

### Impute missing weather values

Cloud coverage (49%) and precipitation (36%) are missing most often, some sites have no data at all for a column
(site 5 never reports pressure), and up to 330 hours are missing at a site ([eda_expl.md, Data quality](eda/eda_expl.md)).
We reindex each site onto a full hourly grid and fill gaps by linear interpolation within the site. A site with no
data for a column gets the mean of the other sites at the same hour, and wind direction is interpolated as sin/cos
([data_aggregation.ipynb §2](preprocessing/data_aggregation.ipynb)).

### Local time zone correction

As noted in the [competition forum](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/115698), weather
timestamps are UTC while meter timestamps are local. We confirmed it on our data: after shifting weather by each
site's UTC offset, temperature peaks at 14:00–16:00 at every site. The exception is site 14, whose meters are in UTC
too ([EDA §6.1](eda/EDA.ipynb#temporal)). The fix improved the LightGBM baseline from 1.290 to 1.286 (clean rows
1.147 → 1.140). The gain is small, but no fold gets worse ([EDA §9](eda/EDA.ipynb#broken)).

### Target transformations

RMSLE is the RMSE of `log1p` values, so we predict `log1p(meter_reading)` with an MSE loss, which optimizes the metric
directly ([EDA §8](eda/EDA.ipynb#metric)). The metric measures relative error: a 10% miss costs about 0.095 at any
scale. It is harsh near zero, though: predicting 5 for a true 0 costs 1.79 ([EDA §8](eda/EDA.ipynb#metric)). Site 0
electricity is converted from kBTU to kWh for training (908,409 rows), and back for the submission
([data_aggregation.ipynb §1](preprocessing/data_aggregation.ipynb), [mlp.ipynb](models/mlp.ipynb#test-predictions)).

## Feature Engineering and Feature Selection

We kept the feature set small and built it the same way for training, validation and test rows
([ashrae/features.py](ashrae/features.py)):

- `log1p(square_feet)`: floor area is the strongest building feature (Spearman ρ = 0.68 with the target for
  electricity) and is bell-shaped on a log scale ([EDA §4, §7](eda/EDA.ipynb#relationships)).
- `year_built`, filled with the training median, plus a missing flag: it is missing for 60% of rows
  ([data_aggregation.ipynb §3](preprocessing/data_aggregation.ipynb), [mlp.ipynb, Features](models/mlp.ipynb)).
- Weather: air and dew temperature, sea-level pressure, wind speed, and wind direction as sin/cos
  ([mlp.ipynb, Features](models/mlp.ipynb)). Temperature pushes chilled water up (ρ = 0.43) and steam and hot water
  down (ρ ≈ −0.45), so the model has to combine meter type with weather ([EDA §7](eda/EDA.ipynb#relationships)).
- Cyclic encoding of hour and day of week (sin/cos), from local time ([mlp.ipynb, Features](models/mlp.ipynb)). Use is
  lower at night and at weekends ([eda_expl.md, Temporal patterns](eda/eda_expl.md)).
- One-hot `meter` and `primary_use` ([ashrae/features.py](ashrae/features.py)).
- Building identity, as either a learned 16-number vector per building (embedding) or the mean `log1p` reading of each
  building × meter series ([mlp.ipynb, Models](models/mlp.ipynb)).

Dropped: precipitation and cloud coverage, which barely relate to the target and are often missing
([EDA §7](eda/EDA.ipynb#relationships)), and `floor_count`, which is missing for 83% of rows
([data_aggregation.ipynb §3](preprocessing/data_aggregation.ipynb)).

Adversarial validation found no feature to drop. A LightGBM classifier tells 2016 from 2017–2018 with AUC 0.75, but
only 0.52 without weather, and each weather feature alone scores 0.51–0.54. Together the weather values fingerprint
one hour of one year, which is not a shift a model can learn from ([adversarial.ipynb §1](validation/adversarial.ipynb)).

## Validation

Our first split trained on Jan–Sep and validated on Oct–Dec. Adversarial validation showed this tests one season: the
two parts are told apart with AUC 0.92, all from weather ([adversarial.ipynb §2](validation/adversarial.ipynb)). So
every comparison in the project uses four folds instead, from the anomaly rules to the optimizers. Each fold holds out
one 3-month block of 2016 and trains on the other nine, with 2M training and 500K validation rows sampled per fold, and
settings are compared by the mean over the folds ([adversarial.ipynb, How to use this](validation/adversarial.ipynb)).
Training on months after the validation block is acceptable here: the test set is a later year with known weather
([adversarial.ipynb, How to use this](validation/adversarial.ipynb)).

Each fold is scored twice: on all rows, as Kaggle scores, and on clean rows without anomalies, which measures the
error on real consumption ([season_validation.ipynb](validation/season_validation.ipynb)). This explained the large
spread between seasons. On all rows, validation ranges from 1.12 (Oct–Dec) to 1.46 (Jan–Mar), in the same order as
the anomaly share (1.1% → 6.1%). On clean rows it's 0.93–1.04 ([season_validation.ipynb, Findings](validation/season_validation.ipynb)).
For our final model, of the gap between training (0.83) and validation (1.20), 0.26 is anomaly rows in validation
and only 0.11 is season plus memorization ([mean_vs_embedding.ipynb, Findings](validation/mean_vs_embedding.ipynb)).

## Models

All networks are trained by one PyTorch Lightning module and logged to MLflow ([mlp.ipynb, Frameworks](models/mlp.ipynb#frameworks)).
We built the model up one change at a time, 10 epochs each ([mlp.ipynb, Results](models/mlp.ipynb)):

| Model | Change | RMSLE, all rows | RMSLE, clean rows |
| --- | --- | --- | --- |
| mlp | 2 hidden layers of 64, ReLU, Adam | 1.559 | 1.380 |
| mlp_bn | + BatchNorm | 1.546 | 1.368 |
| mlp_4out | + one output per meter type | 1.534 | 1.350 |
| mlp_4out_mean | + series mean input | 1.244 | 0.994 |
| mlp_4out_emb | + building embedding instead | 1.200 | 0.938 |
| lgbm | LightGBM, untuned, same features | 1.280 | 1.045 |

- **Knowing the building matters most:** 1.534 → 1.244 with the series mean, 1.200 with the embedding. Both beat
  LightGBM in every fold. Buildings with the same size, use and weather differ in consumption by orders of magnitude
  ([mlp.ipynb, Findings](models/mlp.ipynb#follow-up)).
- **Architecture tweaks help a little:** BatchNorm improves every fold, and one output per meter type improves 3 of 4.
  Both gains are around 0.01, close to seed noise ([mlp.ipynb, Findings](models/mlp.ipynb#follow-up)).
- **Embedding beats series mean** in every season, 0.943 vs 0.990 on clean rows. The two nearly tie on Oct–Dec
  (1.111 vs 1.122) ([mean_vs_embedding.ipynb](validation/mean_vs_embedding.ipynb)). The gain is in heating and cooling
  (chilled water 0.078, steam 0.090, electricity 0.009): the embedding can learn how a building reacts to weather, the
  mean only gives its average level ([mean_vs_embedding.ipynb, Findings](validation/mean_vs_embedding.ipynb)).
- **Optimizer: Adam.** It averaged 1.213 over both building-aware models, against 1.223 for momentum and Nesterov
  and 1.237 for plain SGD, and it's best in every fold. Its lead comes from the embedding model (1.190 vs 1.211–1.229)
  ([optimizers_test.ipynb](models/optimizers_test.ipynb)).
- **Epochs: 20.** Validation keeps improving up to the last epoch tested: 1.219 at 5 epochs, 1.200 at 10, 1.186 at 20,
  averaged over folds ([epochs_test.ipynb](models/epochs_test.ipynb)).
- **Hot water is hardest:** 1.57 against 0.95 for electricity on all rows ([mlp.ipynb, Results](models/mlp.ipynb)),
  and 1.57 against 0.42 on clean rows ([mean_vs_embedding.ipynb](validation/mean_vs_embedding.ipynb)).

## Ensembling

We didn't ensemble. The final submission is a single model: mlp_4out_emb, retrained on all of 2016 (4M rows,
anomalies dropped) for 20 epochs. It predicts all 41,697,600 test rows, and site 0 electricity is converted back to
kBTU ([mlp.ipynb, Test predictions](models/mlp.ipynb#test-predictions)). No leaderboard score is recorded in the
notebooks yet.

## What Didn't Work

- **Regularization.** Weight decay (0.01 or 0.1) changed nothing. Dropout (1.208), a smaller embedding (1.209) and
  all three together (1.231) were worse than none (1.200) ([regularization_test.ipynb](models/regularization_test.ipynb)).
  Most weights feed a BatchNorm layer, which rescales their output anyway, and most of the gap isn't overfitting
  ([regularization_test.ipynb, Findings](models/regularization_test.ipynb)).
- **Early stopping.** There was nothing to stop: validation still improved at the last epoch
  ([regularization_test.ipynb](models/regularization_test.ipynb), [epochs_test.ipynb](models/epochs_test.ipynb)).
- **Choosing on one fold.** On Oct–Dec alone, dropout looked like a win (1.111 → 1.102) but lost on the other three
  seasons ([regularization_test.ipynb, Findings](models/regularization_test.ipynb)). Oct–Dec also made the series mean
  look tied with the embedding and made validation look flat after epoch 9. It also overstated the anomaly rules:
  1.383 → 1.168 there, 1.286 → 1.281 on the fold average ([mean_vs_embedding.ipynb](validation/mean_vs_embedding.ipynb),
  [epochs_test.ipynb](models/epochs_test.ipynb), [EDA §9](eda/EDA.ipynb#broken)).
- **SGD-based optimizers.** Plain SGD changed 3–4× as much from epoch to epoch as the others. Momentum and Nesterov
  were stable but still behind Adam on the embedding model (1.211–1.212 vs 1.190)
  ([optimizers_test.ipynb](models/optimizers_test.ipynb)).
- **He initialization:** same score as PyTorch's default (1.199 vs 1.200), since BatchNorm re-normalizes every hidden
  layer ([mlp.ipynb, What didn't work](models/mlp.ipynb#follow-up)).
- **Two more anomaly rules:** stuck meters (the same non-zero value for 48+ hours) and spikes (more than 10× the daily
  median). On top of the two rules, they change the score by 0.003 at most while dropping another 0.8–1.5% of
  training rows ([EDA §9](eda/EDA.ipynb#broken)). The spike rule mostly flagged normal on/off cycling
  ([eda_expl.md, What didn't work](eda/eda_expl.md)).

## Not Done Yet

The course pipeline also asks for these, and they aren't in the notebooks yet
([DL_Development_Pipeline.md, Cross-check](../../Development%20Guide/DL_Development_Pipeline.md#cross-check-with-assignment-1)):

- a leaderboard score, to check that our validation tracks the test set;
- feature importance for the LightGBM baseline;
- a custom layer and a custom optimizer;
- gradient-flow plots;
- an ensemble of LightGBM and the MLP.
