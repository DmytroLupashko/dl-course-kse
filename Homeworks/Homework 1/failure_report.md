# Failure Report — Homework 1 (ASHRAE Energy Prediction)

Engineering mistakes made during the project and how each was fixed.

## Data preprocessing

### Weather gap-filling invented data

`cloud_coverage` (50% missing) and `precip_depth_1_hr` (36% missing) were forward-filled, so much of each column was
made up. Both barely correlate with the target. **Fix:** dropped both columns.

### Some weather gaps were never filled

Sites with a column missing for the whole year had nothing to fill from, and whole hours were missing from the weather
file. **Fix:** fill regional variables (pressure, wind) from other sites for the same hour, and add the missing hours
before interpolating.

### Weather was 5–8 hours out of step with the meters

Weather is recorded in UTC and meters in local time. **Fix:** shift weather to local time (1.395 → 1.385). The first plan
shifted every site, but site 14's meters are also in UTC, so it is left unshifted.

### The model learned from broken meters

Zero readings from broken meters stayed in training, so the model learned that some buildings use no energy. **Fix:**
these rows are left out of training but still scored in validation. Dropping every zero electricity reading worked
better than the first "zero for a week" rule. Together with the heating and cooling rule, this gives 1.395 → 1.164. The
spike rule flagged normal on/off cycling, so it was removed.

## Modelling and evaluation

### The MLP could not tell buildings apart

The MLP lost to LightGBM (1.524 vs 1.392), which could pick out buildings by their floor area. **Fix:** give the MLP a
building identity, either a per-series mean or an embedding. Both now beat LightGBM.

### Results went stale after the data changed

Model results and adversarial validation still came from the old data. **Fix:** re-ran both, and every MLP improved by
0.11–0.26.

### Decisions were made on one validation season

Validating only on Oct–Dec led to three wrong conclusions. Adversarial validation showed those months differ strongly
from the training months (AUC 0.92).

| Conclusion on Oct–Dec | Result on 4 seasonal folds |
|---|---|
| Dropout helps | Worse in the other three seasons |
| Mean and embedding tie | Embedding wins in every season |
| Training stops improving at about 9 epochs | Still improving at 20 |

**Fix:** four seasonal folds, compared by their mean. The final model is retrained on all of 2016.

### The train–validation gap was misdiagnosed twice

The gap was first treated as overfitting, but regularization didn't help and both learning curves were still falling.
It was then blamed on seasons. In fact, most of it comes from anomaly rows in validation. **Fix:** every notebook now
gives the same explanation. **Lesson:** read the learning curves before choosing a fix.

### Optimizers were ranked on noise

Ranking by the last epoch's score depended on epoch-to-epoch noise. **Fix:** rank by the mean of the last 5 epochs.

### The submission came from the wrong model

The submission used a weaker model trained on 9 months. **Fix:** it now uses the best model, trained on the full year.

## Still open

- The AI report still quotes the Models table and the anomaly results from the Oct–Dec split. They need updating with
  the 4-fold numbers.
