"""Loading raw and processed data, weather cleaning, anomaly flags and train/validation splits."""
import numpy as np
import pandas as pd

from .config import DATA, PROCESSED

WEATHER_COLS = ["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr",
                "sea_level_pressure", "wind_direction", "wind_speed"]


def read_building():
    dtype = {"site_id": "int8", "building_id": "int16", "primary_use": "category", "square_feet": "int32",
             "year_built": "float32", "floor_count": "float32"}
    return pd.read_csv(DATA / "building_metadata.csv", dtype=dtype)


def read_readings(name):
    """`train.csv` or `test.csv`."""
    dtype = {"row_id": "int32", "building_id": "int16", "meter": "int8", "meter_reading": "float32"}
    return pd.read_csv(DATA / name, parse_dates=["timestamp"], dtype=dtype)


def read_weather(name, columns=WEATHER_COLS):
    """`weather_train.csv` or `weather_test.csv`, with only the given weather columns."""
    dtype = {"site_id": "int8", **{c: "float32" for c in columns}}
    return pd.read_csv(DATA / name, usecols=["site_id", "timestamp", *columns], parse_dates=["timestamp"], dtype=dtype)


def clean_weather(w):
    """Each site on a full hourly grid, gaps filled by linear interpolation within the site; a column a site never
    reports gets the other sites' mean at that hour. Wind direction is interpolated as sin/cos."""
    columns = [c for c in w.columns if c not in ("site_id", "timestamp")]
    hours = pd.date_range(w.timestamp.min(), w.timestamp.max(), freq="h")
    grid = pd.MultiIndex.from_product([w.site_id.unique(), hours], names=["site_id", "timestamp"])
    w = w.set_index(["site_id", "timestamp"]).reindex(grid).reset_index()

    rad = np.deg2rad(w.pop("wind_direction"))
    w["wd_sin"], w["wd_cos"] = np.sin(rad), np.cos(rad)

    cols = w.columns.drop(["site_id", "timestamp"])
    w[cols] = w.groupby("site_id")[cols].transform(lambda s: s.interpolate(limit_direction="both"))
    w[cols] = w[cols].fillna(w.groupby("timestamp")[cols].transform("mean"))

    w["wind_direction"] = np.rad2deg(np.arctan2(w.pop("wd_sin"), w.pop("wd_cos"))) % 360
    return w.astype({c: "float32" for c in columns})


def read_processed(name="train_merged.parquet", columns=None):
    """Output of preprocessing/data_aggregation.ipynb."""
    return pd.read_parquet(PROCESSED / name, columns=columns)


def anomaly_flags(df):
    """The two anomaly rules of EDA §9, per row of `df` (needs building_id, meter, timestamp, meter_reading,
    air_temperature). Also returns the length in hours of the zero run each zero reading belongs to."""
    s = df.sort_values(["building_id", "meter", "timestamp"])
    series = s.building_id.astype(int) * 4 + s.meter.astype(int)
    zero = s.meter_reading.eq(0)
    zero_run = (zero.ne(zero.shift()) | series.ne(series.shift())).cumsum()  # consecutive zeros in one series
    run_hours = zero.groupby(zero_run).transform("size").where(zero, 0)
    run_temp = s.air_temperature.groupby(zero_run).transform("mean")
    flags = pd.DataFrame({
        "zero_electricity": zero & (s.meter == 0),
        "off_when_needed": (run_hours >= 168)
        & ((s.meter.isin([2, 3]) & (run_temp < 10)) | ((s.meter == 1) & (run_temp > 20))),
        "zero_run_hours": run_hours,
    })
    return flags.reindex(df.index)


def split(df, val_months, n_train, n_val):
    """Sampled training rows outside `val_months`, without anomalies, and validation rows inside, all kept
    (they are scored in full)."""
    is_val = df.timestamp.dt.month.isin(val_months)
    return df[~is_val & ~df.anomaly].sample(n_train, random_state=0), df[is_val].sample(n_val, random_state=0)
