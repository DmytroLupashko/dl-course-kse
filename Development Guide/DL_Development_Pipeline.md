# Deep Learning Development Pipeline

A synthesis of Module 1 ([Intro](../Module_1/Lecture_1/Intro_to_Deep_Learning.ipynb), [Gradients](../Module_1/Lecture_2/gradients_are_our_best_friends.ipynb), [Validation](../Module_1/Lecture_3/Validation.ipynb), [Regularization](../Module_1/Lecture_4/regularization_and_friends.ipynb), [Frameworks](../Module_1/Lecture_5/Frameworks.ipynb)) into a single, ordered pipeline for approaching a new deep learning problem — from raw task to a defensible final report. Each stage lists *what to do* and *what can go wrong if you skip it*.

---

## 0. Problem Formalization

Before writing any code, define the ML problem formally (Lecture 1, *Formalization*):

- **Data distribution** `D` you are sampling from vs. the actual **dataset** `X = {(x_i, y_i)}` you were given — they are not the same thing, and the gap between them is the root cause of most train/production mismatches later in the pipeline.
- **Hypothesis space** `H` — the family of functions your model can express (a linear model, an MLP, a CNN, …). Pick it deliberately; it bounds what you can possibly learn.
- **Training algorithm** — how you search `H` for a good hypothesis (closed-form, gradient descent, etc.) and what "good" means (loss function).
- **Empirical vs. theoretical risk** — you optimize empirical risk `Q_emp` on a finite sample; you actually care about theoretical risk `Q` over `D`. Everything from here on (validation, regularization) exists to keep these two close.

**Output of this stage:** a one-paragraph problem statement — task type (classification/regression/ranking/etc.), inputs, target, and the optimization loss.

---

## 1. Exploratory Data Analysis (EDA) & Metric Analysis

> Detailed walkthrough: [EDA_guide.md](EDA_guide.md)

- Explore feature distributions, missing values, target balance, leakage risks, train/test schema differences.
- **Analyze the target metric itself** (Lecture 3, *Evaluation metrics*), not just the data:
  - Distinguish **optimization loss** (what the model minimizes) from the **evaluation metric** (what you're judged on) — they're often different and that's fine as long as you understand the mismatch.
  - Know what "good" and "bad" properties a metric can have: sensitivity to class imbalance, scale-dependence, robustness to outliers, whether it's differentiable, whether it degenerates in edge cases (e.g. accuracy under heavy imbalance, MAPE with values near zero).
  - Classification: continuous metrics (ROC-AUC, PR-AUC, log-loss) vs. discrete/threshold metrics (precision/recall/F1, accuracy) — pick continuous metrics for model comparison, threshold metrics for deployment decisions.
  - Regression: MAE vs. RMSE vs. MAPE vs. quantile losses — each penalizes error distribution differently.
  - **Propose a complementary/alternative metric** with justification, and reason about edge cases where the primary metric could mislead you.

**Output of this stage:** an EDA notebook/report + a written critique of the target metric, its weaknesses, and a proposed complementary metric.

---

## 2. Validation Strategy

> Detailed walkthrough: [Validation_guide.md](Validation_guide.md)

This is arguably the most important engineering decision in the whole pipeline (Lecture 3, *Validation Strategies*) — a good model with a broken validation split is worse than a mediocre model with a trustworthy one.

1. **Choose a split scheme based on the data's structure**, not by default:
   - Random holdout / stratified split — i.i.d. tabular data with no leakage risk.
   - K-fold / Stratified K-fold / Repeated K-fold — small datasets, need low-variance estimates.
   - Time-based split / time K-fold — any temporal dependency (never shuffle time series).
   - Group K-fold — grouped data (same user/patient/session must not appear in both train and val).
2. **Justify the choice** in terms of expected correlation with the real (Public/Private LB or production) evaluation, and explicitly reason about why correlation could be *low* (distribution shift, leakage, wrong grouping) or *high*.
3. **Watch for the two failure modes that break local validation ↔ real-world consistency**:
   - Wrong validation strategy for the data structure (e.g. random split on time series or grouped data).
   - Shifted feature distributions between train and test/production.
4. **Run adversarial validation**: train a classifier (Logistic/Linear Regression is enough) to distinguish train vs. test rows. A high AUC means train and test are distinguishable → your validation split is not representative and needs to be redesigned (or the shifted features need to be dropped/fixed). If the task is code-only competition without test features, simulate a held-out test set from training data instead.

**Output of this stage:** a fixed, code-frozen validation split + adversarial validation report, used identically for every model trained afterward.

---

## 3. Classical ML Baseline

Always establish a non-deep baseline before reaching for a neural network (this both sanity-checks the task and gives you a floor to beat):

- Dedicated feature engineering (encodings, aggregations, interactions) suited to a tree-based model.
- Train a tree-based model (Random Forest / Gradient Boosting / sklearn equivalent).
- Log everything: metrics on your validation split, feature importances, hyperparameter tuning trace.
- Submit/evaluate on the real held-out target and compare against local validation — this is your first real check of whether Stage 2's validation strategy actually correlates with ground truth.
- Explicitly conclude: is this baseline adequate, or is there clear signal a more expressive model could capture?

---

## 4. Deep Learning Model

### 4.1 Architecture
- Don't default to a plain MLP — consider multiple input/output branches, auxiliary losses, and embeddings for categorical features where relevant.
- Perform DL-specific feature engineering/preprocessing (normalization, embedding cardinalities, sequence padding, etc.) — different from the tree-model feature engineering in Stage 3.

### 4.2 Weight Initialization (Lecture 4)
Initialization affects convergence speed and gradient stability from step 0:
- **Why it matters:** symmetry breaking, avoiding exploding/vanishing signals, stable gradient flow, faster convergence.
- **Match the scheme to the activation:** LeCun (Sigmoid/Tanh/SELU), Xavier/Glorot (Sigmoid/Tanh), He/Kaiming (ReLU/LeakyReLU); Orthogonal or LSUV for deeper/recurrent nets.

### 4.3 Custom Building Blocks (Assignment requirement)
- Implement **at least one custom layer** as `nn.Module` + `nn.Parameter` (original or a reimplementation).
- Implement **at least one custom optimizer** via `torch.optim.Optimizer` (original, modified, or a reimplementation of Momentum/RMSProp/Adam/RAdam — see Lecture 2 for the exact update rules to reimplement).

### 4.4 Optimization Algorithm (Lecture 2)
Understand what you're actually optimizing with:
- **Gradient Descent variants:** Batch vs. Stochastic vs. Mini-Batch — the standard bias/variance/compute trade-off.
- **Momentum / Nesterov Momentum:** smooth the update direction using past gradients; Nesterov looks ahead before computing the gradient.
- **Adaptive methods:** AdaGrad → RMSProp → Adam → RAdam, each fixing a specific failure of its predecessor (AdaGrad's decaying LR, Adam's initialization bias, RAdam's variance instability early in training). Know each method's practical default hyperparameters and when adaptive optimizers can *underperform* plain SGD (near-convex problems).
- Verify **descent conditions**: for convex problems any local minimum is global, and the negative gradient is guaranteed to be a descent direction — use this as a sanity anchor when debugging a training run that isn't converging.

### 4.5 Regularization (Lecture 4)
> Detailed walkthrough: [Regularization_guide.md](Regularization_guide.md)

Pick based on the bias-variance trade-off you observe (train/val gap, capacity vs. dataset size, Occam's razor):
- **L1/L2 parameter norm penalties** — L2 (weight decay, note: decoupled weight decay ≠ L2-as-loss-term, they differ in the update rule) shrinks weights smoothly; L1 induces sparsity.
- **Dropout** (inverted dropout at train time, scaled correctly so eval-time expectation matches).
- **Batch Normalization / Layer Normalization** — stabilize internal activation statistics; know when each is preferable (BatchNorm needs large-enough batches and behaves differently in eval mode; LayerNorm is batch-size-independent, standard for sequence models).
- Other techniques as appropriate (data augmentation, label smoothing, early stopping) — mentioned in Lecture 4 as complementary tools.

### 4.6 Learning Rate Scheduling (Lecture 4)
A fixed LR is rarely optimal for the whole run:
- Step decay, exponential decay, polynomial decay, cosine annealing (with warm restarts), cyclical LR (CLR), OneCycle, warmup.
- `ReduceLROnPlateau` reacts to a monitored metric directly — but watch for overfitting to the validation metric it plateaus on.

---

## 5. Training Infrastructure & Engineering (Lecture 5)
> Detailed walkthrough: [Frameworks_guide.md](Frameworks_guide.md)

### 5.1 Data pipeline
Structure data loading as the standard three-layer stack:
- **Dataset** — indexable access to a single example.
- **DataCollator** — turns a list of examples into a batch (padding, stacking, on-the-fly augmentation).
- **DataLoader** — batching, shuffling, parallel workers.

### 5.2 Training loop framework
- Prefer a structured framework (e.g. **PyTorch Lightning**) over a hand-rolled loop once the project grows: it separates model logic from the training loop and standardizes checkpointing, logging, and multi-device training.
- Wire up experiment logging/tracking (TensorBoard, Weights & Biases, etc.) from the start — pick based on team needs (self-hosted vs. cloud, collaboration features).
- The Hugging Face `Trainer` stack is a viable alternative when working with `transformers`/`datasets`-native models.

### 5.3 Training tricks
- **Gradient clipping** — caps gradient norm to prevent exploding gradients, especially in RNNs/deep nets or early training.
- **Gradient accumulation** — simulate a larger effective batch size than fits in memory by accumulating gradients over several micro-batches before stepping the optimizer.
- **Stochastic Weight Averaging (SWA)** — average weights over the tail of training for a flatter, better-generalizing minimum.
- **Distributed training** — DDP (Distributed Data Parallel) for standard multi-GPU data parallelism; FSDP (Fully Sharded Data Parallel) when the model itself doesn't fit on one device.
- **Mixed precision training** — fp16/bf16 compute for speed and memory savings, with loss scaling to avoid gradient underflow.

### 5.4 Health checks on optimization (Assignment requirement)
While training, continuously monitor, don't just wait for the final metric:
- Loss/metric curves for both train and validation (diverging curves → overfitting; both flat/high → underfitting or bad LR/init).
- **Gradient flow visualization** per layer — catches vanishing/exploding gradients and dead units early, and is your main debugging tool when a custom layer/optimizer misbehaves.
- Parameter/hyperparameter tuning logs — keep every run's config and result, including failed runs (they are evidence of systematic exploration, not noise to hide).

---

## 6. Model Comparison & Ensembling

- Compare the DL model against the Stage 3 classical baseline on the **same, fixed validation split** — conclude explicitly whether the added complexity of the DL model is justified by the metric gain.
- Combine the classical and DL models into an **ensemble** (blend or stacked two-stage pipeline).
- Re-validate the ensemble with the same validation strategy — an ensemble is a new model and can overfit the validation set just like any other.
- Compare local validation results against the real (Public/Private LB or held-out production) metric at every stage — this is how you detect if Stage 2's validation design has silently broken down as models got more complex.

---

## 7. Final Report

Package the whole pipeline into a single narrative, not just a metrics table:

- Data insights (from EDA).
- Metric analysis and any proposed alternative/complementary metric.
- Validation strategy and its justification + adversarial validation results.
- Feature engineering (separately for the classical and DL tracks).
- All tried models, chosen hyperparameters, and tuning logs.
- Metrics and optimization curves (including gradient flow plots).
- Ensemble results.
- Model insights, conclusions, and — importantly — **failed experiments**: they demonstrate systematic exploration and often carry as much insight as what worked.

---

## Pipeline at a Glance

```
0. Formalize the problem (data vs. distribution, hypothesis space, loss vs. metric)
        │
1. EDA + metric analysis (strengths/weaknesses, propose alternatives)
        │
2. Validation strategy (split scheme → justification → adversarial validation)
        │
3. Classical ML baseline (feature engineering → train → log → compare vs. real metric)
        │
4. Deep learning model
   ├─ architecture (+ embeddings/branches/aux losses)
   ├─ weight init (matched to activation)
   ├─ custom layer (nn.Module/nn.Parameter) + custom optimizer (torch.optim.Optimizer)
   ├─ optimization algorithm (SGD/Momentum/Adam/RAdam/…)
   ├─ regularization (L1/L2, Dropout, Batch/LayerNorm)
   └─ LR schedule (cosine/OneCycle/warmup/plateau)
        │
5. Training infrastructure (Dataset/Collator/DataLoader → Lightning/logging →
   gradient clipping/accumulation, SWA, DDP/FSDP, mixed precision)
   + continuous health checks (loss curves, gradient flow, tuning logs)
        │
6. Compare DL vs. baseline → ensemble → re-validate
        │
7. Final report (insights, curves, conclusions, failed experiments included)
```

---

## Cross-check with Assignment #1

| Assignment #1 requirement | Pipeline stage |
|---|---|
| EDA, metric strengths/weaknesses, alternative metric | Stage 1 |
| Validation strategy + motivation + adversarial validation | Stage 2 |
| Tree-based baseline + feature importance + LB comparison | Stage 3 |
| DL model with custom layer + custom optimizer + regularization | Stage 4 |
| Loss/metric curves, gradient flow, tuning logs | Stage 5.4 |
| Ensemble of baseline + DL model, two-stage pipeline | Stage 6 |
| Comprehensive report incl. failed ideas | Stage 7 |
