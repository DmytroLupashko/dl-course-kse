"""Model inputs, built the same way for training, validation and test rows."""
import numpy as np
import pandas as pd

from .config import METERS


def cyclic(values, period):
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def build_features(df, year_median):
    """`year_median` comes from the training rows, so validation and test rows are filled the same way."""
    X = pd.DataFrame(index=df.index)
    X["log_square_feet"] = np.log1p(df.square_feet)
    X["year_built_missing"] = df.year_built.isna()
    X["year_built"] = df.year_built.fillna(year_median)
    for col in ["air_temperature", "dew_temperature", "sea_level_pressure", "wind_speed"]:
        X[col] = df[col]
    X["wind_dir_sin"], X["wind_dir_cos"] = cyclic(df.wind_direction, 360)
    X["hour_sin"], X["hour_cos"] = cyclic(df.local_timestamp.dt.hour, 24)
    X["dow_sin"], X["dow_cos"] = cyclic(df.local_timestamp.dt.dayofweek, 7)
    meter = pd.get_dummies(pd.Categorical(df.meter, categories=list(METERS)), prefix="meter")
    use = pd.get_dummies(df.primary_use, prefix="use")
    meter.index = use.index = df.index
    return pd.concat([X, meter, use], axis=1).astype("float32")


def add_series_mean(X, df, train_df):
    """Adds the mean log1p reading of each building × meter series in `train_df`; series with no training rows
    get their meter type's mean."""
    y = np.log1p(train_df.meter_reading)
    series_mean = y.groupby([train_df.building_id, train_df.meter]).mean()
    meter_mean = y.groupby(train_df.meter).mean()
    values = series_mean.reindex(pd.MultiIndex.from_arrays([df.building_id, df.meter])).to_numpy()
    unseen = np.isnan(values)
    values[unseen] = df.meter.map(meter_mean).to_numpy()[unseen]
    return X.assign(series_mean=values.astype("float32"))
