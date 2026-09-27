# Stage 1 Guide: Exploratory Data Analysis & Metric Analysis

This expands **Stage 1** of the [DL Development Pipeline](DL_Development_Pipeline.md). It combines the course material ([Formalization](../Module_1/Lecture_1/Intro_to_Deep_Learning.ipynb), [Evaluation Metrics](../Module_1/Lecture_3/Validation.ipynb)) with general EDA best practice. Goal: by the end of this stage you should understand your data well enough to design a validation strategy (Stage 2) and know exactly what "good performance" means for this task.

> **Ground rule:** perform EDA on the **train split only** (or train + unlabeled test features, never test labels). If you look at test-set statistics or targets before splitting, any conclusion you draw can leak into modeling decisions and silently inflate your validation score later.

---

## 1. Data Overview

- **Shape & memory footprint** — row/column counts, memory usage per column; decide if you need chunked loading or downcasting (`float64`→`float32`, categoricals).
- **Schema** — per-column dtype, and whether the *inferred* dtype matches the *semantic* type (e.g. a zip code stored as `int64` is categorical, not numeric).
- **Identifier / key columns** — anything that uniquely identifies a row or a group (user ID, session ID, timestamp). These drive your Stage 2 split choice (Group K-fold, time-based split) — flag them now.
- **Train vs. test schema diff** — do train and test have the same columns? Any column present in one but not the other is an immediate red flag (often the target, sometimes an accidental leak).

## 2. Data Quality

- **Missing values** — percentage per column, and whether missingness is random (MCAR) or informative (e.g. "missing because not applicable" — itself a signal, often worth encoding as a separate indicator).
- **Duplicates** — full-row duplicates and duplicate keys; decide whether duplicates are legitimate (repeated events) or a data bug.
- **Constant / near-constant columns** — zero or near-zero variance features carry no signal and add noise to feature importance later.
- **Inconsistent categories** — typos, mixed casing, or encoding differences in categorical values that should map to the same category.
- **Outliers & impossible values** — values outside physically/logically plausible ranges (negative ages, timestamps in the future). Decide per-feature: clip, transform, or investigate as a labeling error.

## 3. Target Variable Analysis

- **Distribution** — histogram for regression targets (check skew — may motivate a log/Box-Cox transform); class counts for classification (**imbalance ratio** directly affects both your metric choice and validation strategy — see §5).
- **Target vs. time** — plot the target over time if a time column exists; a drifting target is an early sign you'll need a time-based split, not a random one.
- **Sanity bounds** — do target values fall inside the range you'd expect from the problem definition? Values outside plausible bounds may indicate label noise.

## 4. Univariate Feature Analysis

For every feature (or a representative sample of many features):
- Numeric: histogram/KDE, summary stats (mean/median/std/min/max/quantiles), skewness.
- Categorical: value counts / cardinality — is it low-cardinality (one-hot friendly) or high-cardinality (needs target/frequency encoding or an embedding, per Stage 4)?
- Flag features with heavy-tailed distributions early — they usually need a transform before either a tree model (less critical) or a neural net (more critical, since raw scale affects optimization).

## 5. Feature–Target Relationships (Bivariate Analysis)

- Correlation (Pearson/Spearman) or mutual information between each numeric feature and the target; group-wise target means for categorical features.
- **Watch for suspiciously strong predictors.** A single feature with near-perfect correlation to the target, or a naive baseline reaching accuracy far beyond what the domain would suggest, is the classic signature of **target leakage** — a feature that encodes information only available *after* the target is known (e.g. a "resolution time" column when predicting whether a ticket will be resolved). Always ask: *would this value actually be available at prediction time in production?*
- Feature-feature correlation / redundancy — near-duplicate features inflate dimensionality without adding signal and can destabilize both tree-based feature importance and gradient-based training.

## 6. Early Train/Test Distribution Check

This is a lightweight preview of the full **adversarial validation** you'll formalize in Stage 2 — do it now, while it's still cheap, so it can inform feature engineering:
- Overlay train vs. test histograms for key features (or, for many features at once, compare summary statistics side by side).
- If you have both train and test feature tables, a quick way to scan *all* features at once: train a simple classifier (Logistic Regression / shallow tree) to predict "is this row from train or test." An AUC close to 0.5 means the two sets look alike; an AUC well above 0.5 (rule of thumb: >0.55) means there's a **distribution shift**, and the classifier's feature importances tell you *which* features are shifting.
- A shifted feature isn't automatically disqualified — but you now know it needs closer inspection (drop it, bucket it, or reweight training samples) before you trust any model's validation score.
- This preview does not replace the formal adversarial validation over your *actual* validation split in Stage 2 — repeat it there once the split is fixed.

## 7. Target Metric Analysis (course-specific requirement)

Beyond exploring the data, explicitly analyze the metric you'll be optimized/graded on (see [Validation.ipynb §2](../Module_1/Lecture_3/Validation.ipynb)):
- **Separate optimization loss from evaluation metric.** They often differ (e.g. optimizing log-loss while being evaluated on F1) — know why, and whether that gap could bite you.
- **List the metric's edge cases and failure modes** — e.g. accuracy misleading under class imbalance, MAPE blowing up near zero targets, ROC-AUC insensitive to calibration, F1 being threshold-dependent.
- **Propose a complementary metric** that captures an aspect of performance the primary metric misses (e.g. pair accuracy with PR-AUC under imbalance, or pair RMSE with MAE to separately check outlier sensitivity).
- **Propose (and justify) an alternative primary metric**, if you believe the given one isn't the best fit for the business/task goal.

## 8. Deliverables Checklist

By the end of Stage 1 you should be able to produce:

- [ ] Data profiling summary (shape, dtypes, missingness, duplicates, constant columns)
- [ ] Target distribution plot + imbalance/skew assessment
- [ ] Feature distribution plots for key/high-signal features
- [ ] Feature–target relationship analysis, with any leakage suspects flagged and investigated
- [ ] Early train/test distribution comparison (full adversarial validation deferred to Stage 2)
- [ ] Written metric analysis: strengths, weaknesses, edge cases, proposed complementary/alternative metric
- [ ] A short list of modeling implications (e.g. "target is heavily imbalanced → need stratified split + PR-AUC", "feature X leaks → drop it", "feature Y is high-cardinality categorical → embed in Stage 4")

## 9. Common Pitfalls

- Running EDA on the full dataset (train+test combined) and letting test statistics influence preprocessing decisions.
- Treating a high-cardinality ID-like column as a normal categorical feature and one-hot-encoding it.
- Dropping outliers without checking if they're a meaningful, rare-but-real subpopulation (fraud, rare disease) rather than noise.
- Stopping at "the data looks fine" without explicitly interrogating the metric — a technically correct model can still be judged by a metric that doesn't reflect what matters.
- Spending disproportionate time on exhaustive plots for every feature instead of prioritizing the few that show highest signal or highest risk (leakage, shift, imbalance) — EDA should reduce uncertainty, not just produce plots.

## 10. Suggested Tools

- `pandas` / `polars` for profiling (`.describe()`, `.info()`, `.isna().mean()`).
- `ydata-profiling` or `sweetviz` for an automated first-pass report.
- `missingno` for missing-value pattern visualization.
- `seaborn` / `matplotlib` for distribution and relationship plots.
- A shallow `LightGBM`/`sklearn` classifier for the quick train/test distinguishability check in §6.

---

## Worked Example: IEEE-CIS Fraud Detection

Concrete walkthrough on [ieee-fraud-detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Appendix 1 of Assignment #1) — binary classification, target `isFraud`, transaction + identity tables joined on `TransactionID`. (The same task continues in the [Validation_guide.md worked example](Validation_guide.md#worked-example-ieee-cis-fraud-detection).)

- **§1 Data overview:** ~590k train rows, 434 columns, mixed numeric (`TransactionAmt`, anonymized `V1`–`V339`) and categorical (`ProductCD`, `card4`, `card6`, `P_emaildomain`, `DeviceType`) features. The `identity` table only covers ~24% of transactions — a join, not a guaranteed 1:1 match, so most rows will have all `id_*` columns missing by construction, not by data quality issues.
- **§2 Data quality:** many `id_*` and `V*` columns are >90% missing; several `V*` blocks are near-duplicates of each other (they come from a handful of underlying aggregations) — a redundancy, not 339 independent signals.
- **§3 Target:** `isFraud` is ≈3.5% positive → strongly imbalanced. This alone tells you: use stratified splits, and don't trust accuracy — prioritize ROC-AUC/PR-AUC.
- **§4 Univariate:** `TransactionAmt` is heavily right-skewed (a few very large transactions) → log-transform before feeding a neural net.
- **§5 Feature–target relationships (leakage smell):** `TransactionDT` (seconds elapsed since a fixed reference time, i.e. essentially a row index in time) shows up as one of the *strongest* predictors in naive feature importance. That's a red flag, not a discovery — a monotonically increasing "time since start" column can look predictive purely because fraud *rates* drift over the collection period, not because it encodes anything about the transaction itself. Treat it as a candidate for exclusion or transformation (e.g. hour-of-day, day-of-week) rather than a genuine causal feature.
- **§6 Early train/test distribution check:** train covers roughly the first ~182 days of `TransactionDT` and test the next ~182 days — the ranges are **disjoint**. A quick classifier trained to distinguish train vs. test rows using `TransactionDT` alone will look almost perfectly separable. That's expected here (the competition *is* split by time by design) — the useful takeaway is *why* it happens, which sets up exactly the validation strategy in Stage 2.
- **§7 Metric analysis:** the competition metric is ROC-AUC. Weakness: it's insensitive to calibration and treats all ranking errors equally, while in production a false positive (blocking a legitimate purchase) and a false negative (missing real fraud) usually have very different business costs. A reasonable complementary metric: PR-AUC (more informative under 3.5% positive prevalence) or a cost-weighted metric reflecting the asymmetry.

---

## Further Reading

- [A Data Scientist's Essential Guide to Exploratory Data Analysis](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/)
- [Kaggle Project Best Practices 101: Exploratory Data Analysis](https://medium.com/@TheKaggler/kaggle-project-best-practices-101-exploratory-data-analysis-cc0b34718280)
- [Exploratory Data Analysis Checklist: What to Look for Every Time](https://medium.com/codetodeploy/exploratory-data-analysis-checklist-what-to-look-for-every-time-b922da6090ed)
- [Feature leakage in ML: Detect, Prevent, and Fix It (Hex)](https://hex.tech/blog/feature-leakage/)
- [Managing dataset shift by adversarial validation for credit scoring](https://arxiv.org/pdf/2112.10078)

**Next:** [Stage 2 — Validation Strategy](Validation_guide.md)
