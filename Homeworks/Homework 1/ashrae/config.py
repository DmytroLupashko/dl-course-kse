"""Paths and constants shared by all notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the Homework 1 folder
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"  # one tracking database for every notebook

METERS = {0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"}
KBTU_TO_KWH = 0.2931  # site 0 electricity is in kBTU

# Standard-time offset of each site, local time = UTC + offset (EDA §6.1)
UTC_OFFSET = {0: -5, 1: 0, 2: -7, 3: -5, 4: -8, 5: 0, 6: -5, 7: -5,
              8: -5, 9: -6, 10: -7, 11: -5, 12: 0, 13: -6, 14: -5, 15: -5}
# Weather is shifted to the meters' clock; site 14's meters are in UTC too, so it isn't shifted
WEATHER_SHIFT_HOURS = {**UTC_OFFSET, 14: 0}

# 2016 in four 3-month blocks: each fold validates on one block and trains on the other nine months
FOLDS = {"Jan–Mar": [1, 2, 3], "Apr–Jun": [4, 5, 6], "Jul–Sep": [7, 8, 9], "Oct–Dec": [10, 11, 12]}
