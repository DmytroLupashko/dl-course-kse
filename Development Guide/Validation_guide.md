# Stage 2 Guide: Validation Strategy Design

This expands **Stage 2** of the [DL Development Pipeline](DL_Development_Pipeline.md). It combines course material ([Validation.ipynb §3–5](../Module_1/Lecture_3/Validation.ipynb)) with general best practice for cross-validation design and adversarial validation. Goal: end this stage with a **fixed, code-frozen validation split** that you trust to correlate with real-world/leaderboard performance, and reuse identically for every model in Stages 3–6.

> **Why this stage exists:** you can drive training loss to ~0 by adding enough capacity (nonlinear features, more layers) — that tells you nothing about generalization. Validation exists to model unseen data using only the data you already have, so you can tell a genuinely better model from an overfit one before it's too late to find out.

---

## 1. Understand *Why* You're Splitting

- Train and test/eval on the same data and you can't tell memorization from generalization — this is the entire justification for holding data out.
- The estimate is only as good as the assumption that your held-out data resembles the unseen data you actually care about. Every choice below is really an exercise in making that assumption as safe as possible.

## 2. Pick a Split Scheme Matched to the Data's Structure

Don't default to a random split — inspect your data (from Stage 1) for time and group structure first, then choose:

| Data structure | Strategy | Notes |
|---|---|---|
| i.i.d., no leakage risk, plenty of data | **Holdout** (train/val, or train/val/test) | Simple, fast, but a noisier out-of-sample estimate than CV. Use a 3-way split when you need to both pick a checkpoint/early-stop *and* get an unbiased final estimate — selecting the best epoch on the same set you report on is itself a (mild) form of overfitting to that set. |
| Imbalanced target, small dataset, or multiclass | **Stratified** holdout/K-fold | Keeps target distribution stable across splits — important whenever the average target is far from uniform. |
| Small dataset, or metric is noisy fold-to-fold | **K-fold / Repeated K-fold** | Averages K estimates → lower-variance performance estimate, at K× the compute cost. Prefer this whenever you have little data and enough time. |
| Any temporal dependency (the target or features drift over time, or the real deployment always predicts forward in time) | **Time-based split / Time K-fold** (sliding or expanding window) | **Never shuffle time series.** Mimic how the data will actually be split at inference time — usually all-before-date-X → train, all-after → test. Prefer multiple time splits over one, since a single cutoff can land on an atypical period. |
| Data grouped by an entity that must not leak across the split (user, patient, author, session, document) | **Group K-fold** | All rows sharing a group key stay on one side of the split. Skipping this is one of the most common sources of an inflated, unrealistic validation score. |
| Both temporal and grouped structure | **Combined time + group split** | E.g. group by user *and* respect chronological order — needed whenever both dependencies are present. |

**Rule of thumb:** replicate how the *real* evaluation (production traffic, competition Private LB) will actually be split. If you don't know, reason from first principles about what the deployed model will see at inference time, and validate accordingly — do not just take "random split" as a safe default. If time/group metadata that you'd need for a correct split is *missing* from the data, treat that as a limitation to state explicitly, not something to silently ignore (see §6).

## 3. Justify the Choice in Writing

For your report, don't just state the split — argue for it:
- Why do you expect this split to correlate with Private/Public LB or production performance?
- What could cause *low* correlation (wrong split type, leakage, distribution shift — see §5) or *high* correlation you should be suspicious of (e.g. accidental leakage inflating both val and test identically)?
- If evidence later contradicts your assumption (val goes up, LB goes down), that's a signal to revisit the split, not to keep tuning on it.

## 4. Model Checkpoint Selection & Nested Validation

- For gradient-based training (neural nets, boosting), each epoch/iteration produces a different checkpoint. Picking the epoch with best validation score is a legitimate use of the validation set — but it means that set is no longer purely "unseen": your checkpoint choice is now optimized against it, and reported performance on it will be mildly optimistic.
- Use a proper **train / validation / test** three-way split (or nested CV) when you need to *both* select checkpoints/hyperparameters *and* report an unbiased final number: tune and pick checkpoints on val, report only on the untouched test set.
- The same logic applies to hyperparameter search: an inner loop searches hyperparameters, an outer loop (data never touched by the inner loop) gives the unbiased final estimate. Full **nested cross-validation** is the rigorous version of this and is worth the extra compute specifically when the dataset is small enough that hyperparameter search could otherwise fit noise in a single validation split.

## 5. Detect Validation ↔ Real-World Mismatch

Two failure modes account for most cases where local validation and the real (LB/production) metric diverge:

**5.1 Wrong local validation strategy**
- Missed time component → didn't use a time-based split.
- Missed group component → didn't use group K-fold.
- Needed a combined time+group split and used only one.
- **Big val/LB gap while train/val scores look fine → suspect a data leak.** Re-open your Stage 1 EDA, look for suspiciously predictive features, and try removing the most predictive ones to see if the gap closes.
- Try a different metric (e.g. ROC-AUC vs. PR-AUC) — sometimes the discrepancy is metric-specific, not a validation-design bug.
- Competition/production hosts sometimes under-specify the split or metric — be willing to test assumptions empirically rather than trust the description alone.

**5.2 Shifted feature distributions**
- Common causes: naturally evolving data (market conditions, evolving user behavior, drifting language/topics over time) or a data collection/processing bug that makes train, test, and live data subtly different under the hood.
- A shift as small as losing part of an input field between train and serving can visibly move your metric (the course example: a text-truncation bug caused a ~5% ROC-AUC drop) — don't assume shifts are always dramatic or obviously visible in raw feature stats.

## 6. Run Adversarial Validation

Formalize the check you previewed in Stage 1, now on your *actual*, fixed validation split:

1. Combine train and validation/test feature rows; label train=0, test/val=1 (drop the real target).
2. Train a binary classifier — Logistic/Linear Regression is enough, but **match its complexity to your main model's**, since the adversarial AUC is only meaningful relative to a comparably powerful classifier.
3. Read the result:
   - **AUC ≈ 0.5** → train and test are indistinguishable, no significant distribution shift, your standard validation approach should transfer.
   - **AUC well above 0.5** → the classifier can tell them apart → real distribution shift. Feature importances from this classifier point at *which* features are shifting; inspect and decide whether to drop, transform, or reweight by them.
- If the task is code-only (no test features available), simulate a held-out test set by carving it out of the training data with the same split logic you plan to use for real, and adversarially validate that instead.
- An adversarial AUC that's unexpectedly high can also be a leakage signal in disguise (e.g. an engineered feature that encodes row order or collection date) — treat a surprising result as something to explain, not just report.

## 7. Know the Limitations — and Say So

Textbook validation assumes you have enough, well-labeled, representative, richly-annotated data. In practice you often don't:

- **Not enough data** — even K-fold can't give a robust estimate from a handful of samples.
- **Severe class imbalance** — a large dataset can still have a vanishingly small positive class, making any single split's estimate noisy.
- **Missing metadata for a correct split** — no `date`/`group` column available (e.g. de-identified medical data) means you may not be able to do the time/group split the data structurally needs — call this out rather than silently defaulting to random.
- **Insufficient data diversity** — training data drawn from a narrower population/source than deployment (e.g. one clinic's images vs. multi-clinic deployment) means even a "correct" split won't reveal that generalization gap.
- **Business requirements that don't reduce cleanly to one metric**, or whose priorities shift over time (e.g. which error type is more costly changes) — a fixed validation metric can become a moving target; state this as a known limitation of the evaluation rather than pretending the metric is timeless.

## 8. Deliverables Checklist

- [ ] Split scheme selected and justified against the data's actual time/group/i.i.d. structure (§2–3)
- [ ] Train/val(/test) split implemented once and frozen — reused identically for every subsequent model
- [ ] If checkpoints/hyperparameters are tuned on validation, a separate untouched test set (or nested CV) used for the final unbiased number (§4)
- [ ] Adversarial validation run on the real split, AUC reported, and any shifted features investigated (§6)
- [ ] Written discussion of expected correlation with Private/Public LB or production, and what would cause it to break down (§3, §5)
- [ ] Explicit list of validation limitations that apply to this task (§7)

## 9. Common Pitfalls

- Re-splitting the data differently for each model you try — this destroys the ability to compare models fairly and invites cherry-picking a split that flatters a given model.
- Using the same held-out set for both hyperparameter tuning and final reporting, then being surprised when the reported number doesn't reproduce.
- Assuming random-split correlation with LB/production without testing it — check it empirically once real evaluation numbers come in (Stage 3 onward), and revisit the split if they diverge.
- Running adversarial validation once and never again — re-run it if you add new features or if the deployment data source changes.
- Treating a clean adversarial validation AUC (~0.5) as proof there's no leakage — it only checks for *distributional* separability, not for leakage that doesn't shift the marginal feature distributions (e.g. a target-derived feature with the same distribution across train/test).

---

## Worked Example: IEEE-CIS Fraud Detection

Continuing the [EDA_guide.md worked example](EDA_guide.md#worked-example-ieee-cis-fraud-detection) on [ieee-fraud-detection](https://www.kaggle.com/competitions/ieee-fraud-detection) — the EDA already surfaced a disjoint `TransactionDT` range between train and test. Here's how that finding drives every choice below.

- **§2 Split scheme:** `TransactionDT` shows a clear, unavoidable time component (train = first ~182 days, test = next ~182 days) — this rules out a random or plain stratified K-fold outright. The correct local scheme mimics the organizers': hold out the *last* chunk of days (chronologically) as validation, train on everything before it. A single time-based holdout, or a few sliding/expanding time-window folds, both beat any i.i.d. split here.
- **Group consideration:** `card1`/`card2`/`addr1` combinations approximate a recurring "customer." If the same implied customer appears in both train and validation, that leaks customer-specific behavior across the split — worth checking as a combined time+group scheme if a stable customer key can be constructed.
- **§3 Justification:** local validation is expected to correlate with the Public/Private LB *only* if it replicates the organizers' time-based split — this is exactly why public solutions that used random K-fold widely reported CV scores that looked great locally but did not track the leaderboard, while a chronological holdout tracked it much better.
- **§6 Adversarial validation:** combine train and test transaction rows (drop `isFraud`), label train=0/test=1, train a LightGBM classifier of comparable complexity to the main model. Result: a very high AUC (often reported near-perfect in public write-ups), with `TransactionDT` and several `D*` columns (which are themselves time-deltas) dominating feature importance. Interpretation here is *not* "you have a bug" — it's "the shift is real and by design," which confirms the time-based split from §2 is the right one, and flags `TransactionDT`/`D*` as features to engineer carefully (e.g. relative/cyclical transforms) rather than feed in raw.
- **§7 Limitations:** the `identity` table covers only ~24% of transactions, so any validation approach relying on identity-derived grouping is only ever partially applicable — worth stating explicitly rather than assuming full coverage. The ROC-AUC evaluation metric also doesn't capture the real asymmetric cost of false positives (blocking a legitimate customer) vs. false negatives (missing fraud) — a limitation of the *metric*, not the split, but one to note alongside it in the report.

---

## Further Reading

- [Time Series Cross-Validation: Best Practices](https://medium.com/@pacosun/respect-the-order-cross-validation-in-time-series-7d12beab79a1)
- [4 Things to Do When Applying Cross-Validation with Time Series](https://towardsdatascience.com/4-things-to-do-when-applying-cross-validation-with-time-series-c6a5674ebf3a/)
- [Why Nested Cross-Validation Deserves Your Attention](https://medium.com/@pacosun/nested-cross-validation-your-weapon-against-overfitting-17401c851593)
- [Nested Cross-Validation for Machine Learning with Python (MachineLearningMastery)](https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/)

**Previous:** [Stage 1 — EDA & Metric Analysis](EDA_guide.md) · **Next:** [Stage 3 — Classical ML Baseline](DL_Development_Pipeline.md#3-classical-ml-baseline)
