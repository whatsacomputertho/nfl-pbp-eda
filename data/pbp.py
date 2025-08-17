import nfl_data_py as nfl
import pandas as pd
import numpy as np

NFL_PBP_YEARS = [
    2024,
    2023,
    2022,
    2021,
    2020,
    2019,
    2018,
    2017,
    2016,
    2015
]

def load_clean_nfl_pbp_data(
        years: list[int]=NFL_PBP_YEARS
    ) -> pd.DataFrame:
    """
    Loads historical NFL play-by-play data and cleans it for training

    Args:
        years (list[int]): The years of play-by-play data to load

    Returns:
        pd.DataFrame: The loaded & cleaned historical NFL play-by-play data
    """
    # Load NFL play-by-play data
    df = nfl.import_pbp_data(NFL_PBP_YEARS, cache=False, alt_path=None)
    pd.set_option('display.max_columns', None)

    # Clean the NFL play-by-play data
    df = df[
        [
            "qtr",
            "half_seconds_remaining",
            "down",
            "ydstogo",
            "yardline_100",
            "posteam_score",
            "defteam_score",
            "defteam_timeouts_remaining",
            "posteam_timeouts_remaining",
            "play_type",
            "posteam",
            "goal_to_go",
            "timeout",
            "timeout_team",
            "run_location",
            "pass_length",
            "desc"
        ]
    ]
    # Construct score differential column
    df["score_diff"] = df["posteam_score"] - df["defteam_score"]
    df = df.drop("posteam_score", axis=1)
    df = df.drop("defteam_score", axis=1)

    # Clean null values
    df = df[df['half_seconds_remaining'].notna()]
    df = df[df['defteam_timeouts_remaining'].notna()]
    df["down"] = df["down"].fillna(0)
    df = df[df["yardline_100"].notna()]
    df = df[df["play_type"].notna()]

    # Clean goal to go and no huddle columns
    df.loc[df["goal_to_go"] == 1, "goal_to_go"] = True
    df.loc[df["goal_to_go"] == 0, "goal_to_go"] = False
    # TODO: Eventually re-introduce
    #df.loc[df["no_huddle"] == 1.0, "no_huddle"] = True
    #df.loc[df["no_huddle"] == 0.0, "no_huddle"] = False
    df.loc[df["timeout"] == 1.0, "timeout"] = True
    df.loc[df["timeout"] == 0.0, "timeout"] = False

    # Combine pass length, run direction, and timeout into play type column
    df.loc[(df["play_type"] == "pass") & (df["pass_length"] == "short"), "play_type"] = "short_pass"
    df.loc[(df["play_type"] == "pass") & (df["pass_length"] == "deep"), "play_type"] = "deep_pass"
    df.loc[(df["play_type"] == "run") & (df["run_location"] == "left"), "play_type"] = "run_left"
    df.loc[(df["play_type"] == "run") & (df["run_location"] == "middle"), "play_type"] = "run_middle"
    df.loc[(df["play_type"] == "run") & (df["run_location"] == "right"), "play_type"] = "run_right"
    df.loc[(df["timeout"] == True) & (df["timeout_team"] == df["posteam"]), "play_type"] = "offense_timeout"
    df.loc[(df["timeout"] == True) & ~(df["timeout_team"] == df["posteam"]), "play_type"] = "defense_timeout"
    df = df[~(df["play_type"] == "no_play")]
    df = df.drop("run_location", axis=1)
    df = df.drop("pass_length", axis=1)
    df = df.drop("timeout", axis=1)
    df = df.drop("timeout_team", axis=1)
    df = df.drop("posteam", axis=1)

    # Sort remaining passes into short and deep using the observed proportion
    counts = df["play_type"].value_counts()
    num_passes = counts["pass"]
    num_short = counts["short_pass"]
    num_deep = counts["deep_pass"]
    proportion = num_short / (num_short + num_deep)
    df.loc[df["play_type"] == "pass", "play_type"] = np.where(
        np.random.rand(num_passes) < proportion,
        'short_pass',
        'deep_pass'
    )

    # Return the cleaned dataframe
    return df
