# Stage 5 Guide: Frameworks, Training Loop & Logging

This expands **§5 Training Infrastructure & Engineering** of the [DL Development Pipeline](DL_Development_Pipeline.md), based on [Frameworks.ipynb](../Module_1/Lecture_5/Frameworks.ipynb) (Lecture 5). Goal: a training setup where data loading, the training loop, logging and training tricks are standard pieces, so that every experiment is comparable, reproducible and cheap to run.

> **Rule of thumb:** a hand-written loop is fine for a first experiment. Switch to a framework once you need checkpoints, early stopping, logging or several runs to compare — that's usually by the second week of a project.

---

## 1. Choose the Loop

| Option | Use when | Cost |
| --- | --- | --- |
| **Plain PyTorch loop** | one quick experiment, data already in memory, full control needed | you write checkpointing, logging, early stopping yourself |
| **PyTorch Lightning** | most projects: several models, folds, callbacks, logging, multi-GPU later | learn its structure (`LightningModule`, `Trainer`) |
| **Hugging Face `Trainer`** | models and datasets from the `transformers` / `datasets` ecosystem | the model must return a dict with `loss`; less flexible outside NLP/vision |

Imports: `import lightning.pytorch as L` is the current name; the lecture's `import pytorch_lightning as pl` is the same library and also works.

## 2. Data Flow: Dataset → DataCollator → DataLoader

| Part | Job | Keep out of it |
| --- | --- | --- |
| **Dataset** | read raw data, return **one** sample by index | batching, GPU work |
| **DataCollator** | turn a list of samples into a batch: to tensors, pad/normalize, return a dict (`{"pixel_values": ..., "labels": ...}`) | `.to(device)` — collators run inside CPU worker processes |
| **DataLoader** | batching, shuffling, parallel workers (`num_workers`), `collate_fn=collator` | model logic |

Practical rules from the lecture:

- Move batches to the GPU in the **main process**: `x.to(device, non_blocking=True)`, paired with `pin_memory=True` in the DataLoader (CUDA only).
- `num_workers = os.cpu_count() - 1` is a reasonable start for data read from disk.
- Put transforms in the collator when different loaders need different policies (e.g. augmentation for training only).
- **Tabular data that fits in memory is the exception.** A DataLoader that fetches and collates one row at a time is slow for millions of rows. Keep the whole feature tensor on the device and index a random permutation per batch, or pass a batch-level sampler.

## 3. PyTorch Lightning

Split the code into three pieces:

- **`LightningModule`**: the model plus `training_step`, `validation_step` and `configure_optimizers` (optimizer and LR scheduler). Call `self.save_hyperparameters()` so every run records its settings.
- **`LightningDataModule`**: `setup()` builds the datasets; `train_dataloader()` / `val_dataloader()` return loaders. Folds become one DataModule per fold.
- **`Trainer`**: runs the loop. Everything else is configuration: `max_epochs`, `accelerator="auto"`, `logger=...`, `callbacks=[...]`.

Callbacks worth using from day one:

- `ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)` — keeps the best epoch, not the last.
- `EarlyStopping(monitor="val_loss", patience=3)` — stops when validation stops improving.
- `LearningRateMonitor(logging_interval="step")` — logs the LR so schedules can be checked.

Metrics: use `torchmetrics`. The pattern is `update()` in every step, `compute()` at epoch end, then `reset()` — this gives exact epoch-level metrics instead of an average of batch averages.

## 4. Hugging Face `Trainer`

- The model's `forward` takes named tensors (`pixel_values`, `labels`, ...) and returns `{"loss": ..., "logits": ...}`.
- `TrainingArguments` holds everything else: batch sizes, epochs, `eval_strategy="epoch"`, `save_strategy`, `learning_rate`, `weight_decay`, `lr_scheduler_type`, `optim="adamw_torch"`, `report_to="wandb"`.
- `compute_metrics(eval_pred)` receives logits and labels as NumPy arrays and returns a dict of metrics.
- The same `data_collator` from §2 plugs straight in.

## 5. Logging and Experiment Tracking

| Tool | Strengths | Weaknesses | Best for |
| --- | --- | --- | --- |
| **Weights & Biases** | rich UI, sweeps, artifacts, team features | account needed; data leaves your machine unless self-hosted | research and team projects |
| **TensorBoard** | ships with PyTorch, simple scalars and images | weak experiment management, file-based | quick local runs |
| **MLflow** | open source, self-hostable, model registry | plainer UI | self-hosted MLOps |
| **CSV/JSON** | no dependencies, full control | you build every view yourself | small projects, restricted environments |

Log, for every run: the full config (hyperparameters, data version, seed), training and validation loss per epoch, the learning rate, the metric per important group (e.g. per class or per meter type), and the run's result — **including failed runs**, which show the search was systematic.

## 6. Training Tricks

All are one `Trainer` argument or callback in Lightning.

| Trick | What it does | Use when | Lightning |
| --- | --- | --- | --- |
| **Gradient clipping** | rescales the gradient when its norm exceeds τ: g′ = g · τ / ‖g‖ | loss spikes, RNNs, deep nets, early training | `gradient_clip_val=1.0` |
| **Gradient accumulation** | averages gradients over N mini-batches before one optimizer step | the batch you want doesn't fit in memory | `accumulate_grad_batches=8` |
| **SWA** | averages weights from the last part of training | extra generalization at the end of training | `callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)]` |
| **DDP** | one process per GPU, gradients synchronized | several GPUs, model fits on one | `strategy="ddp", devices=N` |
| **FSDP** | shards parameters, gradients and optimizer state across GPUs | the model doesn't fit on one GPU | `strategy=FSDPStrategy(...)` |
| **Mixed precision** | 16-bit compute, 32-bit master weights | NVIDIA GPUs with Tensor Cores; bf16 on A100/H100 | `precision="16-mixed"` or `"bf16-mixed"` |

bf16 keeps float32's range, so it rarely needs loss scaling; fp16 needs it (Lightning handles this automatically).

## 7. Reproducibility

- One `set_seed(seed)` for `random`, NumPy and PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`); in Lightning, `L.seed_everything(seed)`.
- `torch.backends.cudnn.deterministic = True` and `benchmark = False` make CUDA runs repeatable, at some speed cost.
- Some backends are never fully deterministic (e.g. Apple MPS). Measure the run-to-run noise once and treat smaller differences as ties.

## 8. Checklist

- [ ] Loop chosen deliberately (§1); framework adopted once you need callbacks or several runs
- [ ] Dataset / collator / loader responsibilities kept separate; no GPU work in workers
- [ ] Best-epoch checkpoint and early stopping on validation
- [ ] Every run logged with its config, per-epoch curves and result, including failures
- [ ] Seeds fixed; run-to-run noise measured
- [ ] Tricks (clipping, accumulation, SWA, mixed precision) added only for a problem they solve

---

## Worked Example: ASHRAE Great Energy Predictor III

Homework 1 ([mlp.ipynb](../Homeworks/Homework%201/models/mlp.ipynb)) trains its MLPs with Lightning and tracks them in MLflow. How it maps to this guide:

- **Data flow:** 3M rows × 33 features fit in memory — the tabular exception in §2. The Dataset keeps them as tensors and implements `__getitems__`, so each batch is one indexing step while the DataLoader still shuffles and batches; the collator just passes the batch through.
- **One module, many networks:** every network takes the same inputs `(x, meter, building)`, so a single `LightningModule` trains all five MLPs with `torchmetrics` RMSLE.
- **Tracking:** each run logs its settings, per-epoch training and validation RMSLE, and per-meter scores to a local MLflow SQLite file; the results table and learning curves are read back from MLflow.
- **Shared code:** the Dataset, networks, `LightningModule` and an MLflow `Experiment` helper live in a small package ([ashrae/](../Homeworks/Homework%201/ashrae)) that every notebook imports, so each experiment notebook contains only what it changes.
- **Next:** four seasonal folds are one `Experiment.fit()` call per fold's loaders, and early stopping one callback.
- **Reproducibility:** training on Apple MPS isn't fully deterministic. The same model and seed scored 1.1075 and 1.1084 in two runs, so differences below ~0.01 are treated as noise.
- **Framework pitfall:** in the same process, PyTorch and LightGBM each load their own OpenMP runtime, and multi-threaded LightGBM crashed the kernel on macOS. The fix was `n_jobs=1` for LightGBM, or running it in a separate notebook.
- **Tricks:** none are needed at this size. The network is small (about 30K parameters, most of them the building embedding), batches fit easily, and training takes seconds per epoch; clipping or mixed precision would add complexity without solving a problem.

---

**Next:** [Pipeline §5.4 — Health checks on optimization](DL_Development_Pipeline.md#54-health-checks-on-optimization-assignment-requirement)
