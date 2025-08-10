import nfl_data_py as nfl
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

NFL_PBP_YEARS = [
    2024
]

# Load NFL play-by-play data
df = nfl.import_pbp_data(NFL_PBP_YEARS, cache=False, alt_path=None)
pd.set_option('display.max_columns', None)

# Clean the NFL play-by-play data
df = df[
    [
        "half_seconds_remaining",
        "ydstogo",
        "defteam_timeouts_remaining",
        "posteam_timeouts_remaining",
        "posteam_score",
        "defteam_score",
        "qtr",
        "down",
        "yardline_100",
        "play_type",
        "desc"
    ]
]
df["score_diff"] = df["posteam_score"] - df["defteam_score"]
df = df.drop("posteam_score", axis=1)
df = df.drop("defteam_score", axis=1)
df = df[df['half_seconds_remaining'].notna()]
df = df[df['defteam_timeouts_remaining'].notna()]
df["down"] = df["down"].fillna(0)
df = df[df["yardline_100"].notna()]
df = df[df["play_type"].notna()]
train, test = train_test_split(df)

# Initialize and train a logistic regression model
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=5000
)
model.fit(
    train[
        [
            "half_seconds_remaining",
            "ydstogo",
            "defteam_timeouts_remaining",
            "posteam_timeouts_remaining",
            "score_diff",
            "qtr",
            "down",
            "yardline_100"
        ]
    ],
    train[["play_type"]]
)

# Test the logistic regression model
pred = model.predict(
    test[
        [
            "half_seconds_remaining",
            "ydstogo",
            "defteam_timeouts_remaining",
            "posteam_timeouts_remaining",
            "score_diff",
            "qtr",
            "down",
            "yardline_100"
        ]
    ]
)
print(confusion_matrix(test[["play_type"]], pred))
print(classification_report(test[["play_type"]], pred))

# Demo the logistic regression model
demo = model.predict(
    [
        [
            900,
            10,
            3,
            3,
            0,
            2,
            1,
            35
        ]
    ]
)
print(demo)
