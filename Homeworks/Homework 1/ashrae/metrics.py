"""RMSLE, computed from log1p values. Kept free of torch so LightGBM notebooks can import it (EDA, adversarial)."""
import numpy as np

from .config import METERS


def rmsle(y_log, pred_log):
    """Negative predictions count as 0, since readings can't be negative."""
    return float(np.sqrt(np.mean((np.clip(pred_log, 0, None) - y_log) ** 2)))


def rmsle_by_meter(y_log, pred_log, meter, prefix="val_rmsle"):
    """Overall RMSLE and one per meter type, keyed like the MLflow metrics."""
    by_meter = {f"{prefix}_{name}": rmsle(y_log[meter == m], pred_log[meter == m]) for m, name in METERS.items()}
    return {prefix: rmsle(y_log, pred_log), **by_meter}
