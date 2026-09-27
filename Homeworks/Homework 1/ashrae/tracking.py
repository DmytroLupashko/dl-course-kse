"""MLflow bookkeeping: one `Experiment` per notebook run, tagging every run with that run's session."""
from time import strftime

import lightning.pytorch as L
import mlflow
from lightning.pytorch.loggers import MLFlowLogger

from .config import MLFLOW_URI
from .modeling import EnergyModel


class Experiment:
    def __init__(self, name):
        self.name, self.session = name, strftime("%Y-%m-%d %H:%M:%S")  # session: tag shared by this execution's runs
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(name)
        self.client = mlflow.MlflowClient()
        print(f"MLflow: {MLFLOW_URI}, experiment '{name}', session {self.session}")

    def fit(self, make_net, train_loader, val_loader=None, *, run_name, epochs, tags=None, **model_kw):
        """Seeds, builds the net, trains it for `epochs` and logs the run. Returns the model and its run id."""
        L.seed_everything(0, verbose=False)
        model = EnergyModel(make_net(), **model_kw)
        logger = MLFlowLogger(self.name, tracking_uri=MLFLOW_URI, run_name=run_name,
                              tags={"session": self.session, **(tags or {})})
        logger.log_hyperparams({"epochs": epochs, "batch_size": train_loader.batch_size,
                                "n_inputs": train_loader.dataset.x.shape[1]})
        trainer = L.Trainer(max_epochs=epochs, logger=logger, enable_checkpointing=False,
                            enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(model, train_loader, val_loader)
        return model, logger.run_id

    def log_metrics(self, run_id, metrics):
        for key, value in metrics.items():
            self.client.log_metric(run_id, key, value)

    def history(self, run_id, key):
        """Per-epoch values of a metric, in epoch order."""
        return [m.value for m in sorted(self.client.get_metric_history(run_id, key), key=lambda m: m.step)]

    def runs(self):
        """This session's runs, as a DataFrame."""
        return mlflow.search_runs(experiment_names=[self.name], filter_string=f"tags.session = '{self.session}'")
