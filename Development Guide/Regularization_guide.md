# Stage 4.5 Guide: Regularization

This expands **§4.5 Regularization** of the [DL Development Pipeline](DL_Development_Pipeline.md). It builds on the course material ([regularization_and_friends.ipynb](../Module_1/Lecture_4/regularization_and_friends.ipynb)) and adds a practical order of work, typical settings, and how to regularize against a **distribution shift** such as seasons. Goal: close the gap between training and validation error without making either worse.

> **Rule of thumb:** regularization trades a little training fit for better generalization. It only helps when the model is overfitting. If training error is already high, you need more capacity or better features, not regularization.

---

## 1. Diagnose Before You Regularize

Plot training and validation loss per epoch, and read the pattern:

| Pattern | Diagnosis | Action |
|---|---|---|
| both high, close together | underfitting | more capacity, better features, train longer |
| training keeps falling, validation flat or rising | overfitting | regularize (this guide) |
| validation best in epoch 1–2, then rises | fast memorization, often of noisy labels or IDs | clean the data first, then early stopping and weight decay |
| validation noisy between epochs | validation too small or one-sided, or learning rate too high | fix validation (more folds), lower learning rate |

Also check the gap *per group* (per meter, per site, per season). A model can overfit one group while underfitting another.

## 2. The Toolbox, in Order of Cost

Try them roughly in this order: the first ones are nearly free and rarely hurt.

1. **Clean the labels.** Training on errors (broken meters, wrong units) is the most common source of "overfitting". Removing them is regularization at the data level, and it usually beats any model-side technique.
2. **Early stopping.** Track validation loss every epoch, keep the weights of the best epoch, and stop after a few epochs without improvement (patience 2–5). Report the best epoch count and reuse it when retraining on all data.
3. **Weight decay (L2).** Use `torch.optim.AdamW`: it decouples the decay from the adaptive update, which plain Adam with an L2 loss term does not. Try `1e-4`, `1e-3`, `1e-2`. Don't decay biases or normalization parameters.
4. **Dropout.** For tabular MLPs, 0.1–0.3 after each hidden activation; never on the output layer. Remember `model.eval()` at prediction time. With BatchNorm, put dropout after BatchNorm and activation, not before: dropout before BatchNorm shifts the statistics BatchNorm learns during training.
5. **Smaller capacity.** Fewer or narrower hidden layers, smaller embedding dimension. Cheap to try, and often as effective as adding penalties.
6. **Regularize embeddings.** Embeddings of IDs (buildings, users) memorize fastest, since each ID has its own free parameters. Use a small dimension (4–16), weight decay on the embedding table, or dropout on the embedding output.
7. **Noise and augmentation.** For tabular data: small Gaussian noise on standardized continuous inputs (σ ≈ 0.01–0.1). For images and audio, see the lecture.
8. **Ensembling.** Average models trained with different seeds or on different folds. It reduces variance without touching any single model. Average in the space the metric uses (e.g. `log1p`).

**Gradient-boosted trees (LightGBM)** have their own knobs for the same job: `num_leaves` and `min_child_samples` (tree size), `feature_fraction` and `bagging_fraction` (randomness), `lambda_l1` / `lambda_l2` (penalties), and a lower `learning_rate` with more trees plus early stopping.

## 3. How to Tune

- **Fix validation first** ([Validation_guide.md](Validation_guide.md)). Every choice below is only as good as the split it's measured on.
- **Change one knob at a time**, starting from the defaults above, and keep everything else fixed, including seeds.
- **Use 2–3 seeds** before trusting a difference smaller than about 1% of the metric.
- **Watch both curves.** Good regularization raises training loss slightly and lowers validation loss. If both get worse, you've gone too far.
- **Retrain on all data at the end** with the chosen settings and the best epoch count found in validation.

## 4. Regularizing Against Seasonal Shift

A seasonal shift means the data the model trains on and the data it's judged on come from different conditions: e.g. training on January–September and validating on October–December. Adversarial validation detects it: a classifier separates the two periods with a high AUC, mostly through weather or date features.

The aim is different from ordinary regularization: make the model rely on relationships that hold in **every** season, instead of fitting whatever pattern the training months happen to show.

**Validation that exposes the shift**
- Split the year into blocks of consecutive months (e.g. four 3-month blocks), train on the others, validate on each block in turn. This is the single most important step: without it you can't see whether any of the ideas below help.
- Choose settings by the **mean** across folds, and prefer settings whose **spread** across folds is small. A setting that wins in one season and loses in another is fitting that season.
- Early-stop on the season-covering folds, not on one season.

**Features that hold across seasons**
- **Physical transforms instead of raw values.** Heating and cooling respond to temperature beyond a comfort band: heating degree-hours `max(0, 18 − T)` and cooling degree-hours `max(0, T − 22)`. These behave the same in winter and summer; raw temperature doesn't.
- **Smoothed and lagged weather** (rolling 24-hour and 7-day means per site). Buildings react to recent weather, not just the current hour, and smoothed values are harder to memorize than exact hourly readings.
- **Avoid features that act as a date ID**, such as day of year, or month when the validation months never occur in training. The model can only memorize these; it can't generalize from them.
- **Normalize per series.** Predict the deviation from each building's typical level (e.g. `log1p(reading)` minus the series mean on training data) so the model learns how weather *changes* consumption, not each building's absolute level in the training months.

**Model-side**
- **Regularize identity features harder**, since they're what memorizes a season: smaller embedding dimension, more weight decay on the embedding table, or embedding dropout.
- **Separate outputs per group that reacts differently**, e.g. one output per meter type, so heating and cooling don't share one temperature response.
- **For trees:** monotone constraints where physics is clear (e.g. cooling load non-decreasing in temperature above the comfort band).
- **Ensemble the fold models.** Each fold's model saw a different mix of seasons; averaging them smooths out season-specific quirks.

**What not to do**
- Don't tune on a single season and trust the result for a test set that covers all seasons.
- Don't reweight training rows by adversarial probability when the test set covers the same seasons as the full training data; the final model trained on all months already matches it.

## 5. Checklist

- [ ] Learning curves plotted, and the pattern diagnosed (§1)
- [ ] Label errors removed before any model-side regularization
- [ ] Validation covers every season/group the test set contains
- [ ] Early stopping on that validation, best epoch recorded
- [ ] Weight decay, dropout and capacity tuned one at a time, 2–3 seeds
- [ ] Identity embeddings regularized separately
- [ ] Final model retrained on all data with the chosen settings

---

## Worked Example: ASHRAE Great Energy Predictor III

Homework 1: predict hourly energy use of ~1,450 buildings; see [EDA.ipynb](../Homeworks/Homework%201/eda/EDA.ipynb), [mlp.ipynb](../Homeworks/Homework%201/models/mlp.ipynb) and [adversarial.ipynb](../Homeworks/Homework%201/validation/adversarial.ipynb).

- **Diagnosis:** the best MLP (4 meter outputs + a 16-dimensional building embedding) reaches training RMSLE 0.80 but validation 1.11: clear overfitting.
- **Label cleaning came first, and paid most.** Before zero-electricity readings were removed, the building-identity models got *worse* on validation after epoch 1: they had memorized that some buildings use no power. After cleaning, validation improved every epoch and RMSLE fell from 1.365 to 1.108.
- **The shift:** adversarial validation separates our training months (Jan–Sep) from our validation months (Oct–Dec) with AUC 0.92, almost all through weather. Training vs Kaggle test only reaches 0.75, and that comes from weather fingerprints (exact value combinations that identify one hour), not a real shift.
- **Plan, in order:**
  1. four 3-month folds, and pick settings by mean RMSLE across them;
  2. early stopping on those folds;
  3. AdamW weight decay `1e-4` → `1e-2`, with extra decay on the building embedding;
  4. dropout 0.1–0.3 after each hidden layer;
  5. heating/cooling degree-hours and 24-hour / 7-day rolling temperature per site;
  6. retrain on all of 2016 and average the fold models.
- **Success looks like:** mean fold RMSLE falls, the spread across folds shrinks, and the train–validation gap narrows.
- **Result** ([regularization_test.ipynb](../Homeworks/Homework%201/models/regularization_test.ipynb)): on four seasonal folds, weight decay, dropout, a smaller embedding and their combination all failed to improve validation (best: none, 1.200). Validation was still improving at epoch 10, so the gap wasn't classic overfitting but the difference between seasons (1.05–1.42 across folds). Dropout alone would have looked like a win on October–December only. Lesson: diagnose with §1 first — here the fix is seasonal features (§4), not a penalty.

---

**Next:** [Pipeline §4.6 — Learning Rate Scheduling](DL_Development_Pipeline.md#46-learning-rate-scheduling-lecture-4)
