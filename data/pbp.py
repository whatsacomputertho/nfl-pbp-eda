import nfl_data_py as nfl
import pandas as pd

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

    # Return the cleaned dataframe
    return df
