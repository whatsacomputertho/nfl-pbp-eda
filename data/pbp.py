import nfl_data_py as nfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

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

def load_clean_nfl_pbp_between_play_data(
        years: list[int]=NFL_PBP_YEARS
    ) -> pd.DataFrame:
    """
    Loads historical NFL play-by-play data and cleans it for training the
    between-play model

    Args:
        years (list[int]): The years of play-by-play data to load

    Returns:
        pd.DataFrame: The loaded & cleaned historical NFL play-by-play data
    """
    # Load NFL play-by-play data
    df = nfl.import_pbp_data(years, cache=False, alt_path=None)

    # Derive play duration and drop outliers
    df['game_seconds_next'] = df.groupby('game_id')['game_seconds_remaining'].shift(-1)
    df['play_duration'] = df['game_seconds_remaining'] - df['game_seconds_next']
    df = df.query('play_duration < 69.0')
    df = df.query('play_duration >= 0')
    df['prev_play_duration'] = df['play_duration'].shift(-1)
    df['prev_play_out_of_bounds'] = df['out_of_bounds'].shift(-1)
    df['prev_play_incomplete_pass'] = df['incomplete_pass'].shift(-1)
    df['prev_play_timeout'] = df['timeout'].shift(-1)

    season_posteam_groups = df.groupby(["season", "posteam"])
    for groups, group in season_posteam_groups:
        # Average play duration group
        total_plays = len(group)
        total_duration = group["play_duration"].sum()
        average_play_duration = total_duration / total_plays
        df.loc[
            (df['season'] == groups[0]) & (df['posteam'] == groups[1]),
            "average_play_duration"
        ] = average_play_duration

    # Normalize average play duration
    min_avg_play_duration = df["average_play_duration"].min()
    max_avg_play_duration = df["average_play_duration"].max()
    df["norm_average_play_duration"] = (df["average_play_duration"] - min_avg_play_duration) / \
        (max_avg_play_duration - min_avg_play_duration)

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
            "no_huddle",
            "average_play_duration",
            "norm_average_play_duration",
            "posteam",
            "goal_to_go",
            "timeout",
            "timeout_team",
            "play_duration",
            "prev_play_duration",
            "prev_play_out_of_bounds",
            "prev_play_incomplete_pass",
            "prev_play_timeout",
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

    # Clean goal to go and no huddle columns
    df.loc[df["goal_to_go"] == 1, "goal_to_go"] = True
    df.loc[df["goal_to_go"] == 0, "goal_to_go"] = False
    df.loc[df["no_huddle"] == 1.0, "no_huddle"] = True
    df.loc[df["no_huddle"] == 0.0, "no_huddle"] = False
    df.loc[df["timeout"] == 1.0, "timeout"] = True
    df.loc[df["timeout"] == 0.0, "timeout"] = False

    # Return the cleaned dataframe
    return df

def load_clean_nfl_pbp_playcall_data(
        years: list[int]=NFL_PBP_YEARS
    ) -> pd.DataFrame:
    """
    Loads historical NFL play-by-play data and cleans it for training the
    playcalling model

    Args:
        years (list[int]): The years of play-by-play data to load

    Returns:
        pd.DataFrame: The loaded & cleaned historical NFL play-by-play data
    """
    # Load NFL play-by-play data
    df = nfl.import_pbp_data(years, cache=False, alt_path=None)

    season_posteam_groups = df.groupby(["season", "posteam"])
    for groups, group in season_posteam_groups:
        # Run playcall percent
        run_percent = group["play_type"].value_counts(normalize=True)["run"]
        df.loc[
            (df['season'] == groups[0]) & (df['posteam'] == groups[1]),
            "run_percent"
        ] = run_percent

        # Go for it on fourth percent
        fourth_downs = group[group["down"] == 4]
        total_fourth_downs = len(fourth_downs)
        go_for_it_count = len(fourth_downs[(fourth_downs["play_type"] == "run") | (fourth_downs["play_type"] == "pass")])
        p_go_for_it = go_for_it_count / total_fourth_downs
        df.loc[
            (df['season'] == groups[0]) & (df['posteam'] == groups[1]),
            "go_for_it_percent"
        ] = p_go_for_it

    # Normalize run percent
    min_run_percent = df["run_percent"].min()
    max_run_percent = df["run_percent"].max()
    df["norm_run_percent"] = (df["run_percent"] - min_run_percent) / \
        (max_run_percent - min_run_percent)
    min_go_for_it_percent = df["go_for_it_percent"].min()
    max_go_for_it_percent = df["go_for_it_percent"].max()
    df["norm_go_for_it_percent"] = (df["go_for_it_percent"] - min_go_for_it_percent) / \
        (max_go_for_it_percent - min_go_for_it_percent)

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
            "no_huddle",
            "play_type",
            "run_percent",
            "norm_run_percent",
            "go_for_it_percent",
            "norm_go_for_it_percent",
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
    df.loc[df["no_huddle"] == 1.0, "no_huddle"] = True
    df.loc[df["no_huddle"] == 0.0, "no_huddle"] = False
    df.loc[df["timeout"] == 1.0, "timeout"] = True
    df.loc[df["timeout"] == 0.0, "timeout"] = False

    # Combine pass length, run direction, and timeout into play type column
    df = df[~(df["play_type"] == "no_play")]
    df = df.drop("run_location", axis=1)
    df = df.drop("pass_length", axis=1)
    df = df.drop("timeout", axis=1)
    df = df.drop("timeout_team", axis=1)
    df = df.drop("posteam", axis=1)

    # Return the cleaned dataframe
    return df

def load_clean_nfl_pbp_fieldgoal_data(
        years: list[int]=NFL_PBP_YEARS,
        clean_columns: bool=True
    ):
    """
    Loads historical NFL play-by-play data and cleans it for training the
    field goal result model

    Args:
        years (list[int]): The years of play-by-play data to load
        clean_columns (bool): Whether to drop irrelevant columns
    
    Returns:
        pd.DataFrame: The loaded & cleaned historicla NFL field goal data
    """
    # Load NFL play-by-play data
    df = nfl.import_pbp_data(NFL_PBP_YEARS, cache=False, alt_path=None)

    # Derive play duration and drop outliers
    df['game_seconds_next'] = df.groupby('game_id')['game_seconds_remaining'].shift(-1)
    df['play_duration'] = df['game_seconds_remaining'] - df['game_seconds_next']
    df = df.query('play_duration < 69.0')
    df = df.query('play_duration >= 0')

    # Derive the field goal kicking and field goal defense properties
    field_goal_attempts = df.query("field_goal_attempt == 1")
    season_posteam_groups = field_goal_attempts.groupby(["season", "posteam"])
    season_defteam_groups = field_goal_attempts.groupby(["season", "defteam"])

    # Loop through each group, get field goal proportion for and against, label
    for groups, group in season_posteam_groups:
        # Field goal percent for
        fg_percent = group["field_goal_result"].value_counts(normalize=True)["made"]
        field_goal_attempts.loc[
            (field_goal_attempts['season'] == groups[0]) & (field_goal_attempts['posteam'] == groups[1]),
            "field_goal_percent"
        ] = fg_percent

        # Field goal blocked percent against
        blocked_percent = group["field_goal_result"].value_counts(normalize=True).get("blocked", 0)
        field_goal_attempts.loc[
            (field_goal_attempts['season'] == groups[0]) & (field_goal_attempts['posteam'] == groups[1]),
            "blocked_percent_against"
        ] = blocked_percent
    for groups, group in season_defteam_groups:
        # Field goal percent against
        fg_percent_against = group["field_goal_result"].value_counts(normalize=True)["made"]
        field_goal_attempts.loc[
            (field_goal_attempts['season'] == groups[0]) & (field_goal_attempts['defteam'] == groups[1]),
            "field_goal_percent_against"
        ] = fg_percent_against

        # Field goal blocked percent for
        blocked_percent = group["field_goal_result"].value_counts(normalize=True).get("blocked", 0)
        field_goal_attempts.loc[
            (field_goal_attempts['season'] == groups[0]) & (field_goal_attempts['defteam'] == groups[1]),
            "blocked_percent"
        ] = blocked_percent

    # Normalize the field goal percent for and field goal percent against column
    min_fg_percent = field_goal_attempts["field_goal_percent"].min()
    max_fg_percent = field_goal_attempts["field_goal_percent"].max()
    field_goal_attempts["norm_field_goal_percent"] = (field_goal_attempts["field_goal_percent"] - min_fg_percent) \
        / (max_fg_percent - min_fg_percent)
    min_fg_percent_against = field_goal_attempts["field_goal_percent_against"].min()
    max_fg_percent_against = field_goal_attempts["field_goal_percent_against"].max()
    field_goal_attempts["norm_field_goal_percent_against"] = (field_goal_attempts["field_goal_percent_against"] - min_fg_percent_against) \
        / (max_fg_percent_against - min_fg_percent_against)
    
    # Normalize the field goal blocked percent for and against columns
    min_blocked_percent_against = field_goal_attempts["blocked_percent_against"].min()
    max_blocked_percent_against = field_goal_attempts["blocked_percent_against"].max()
    field_goal_attempts["norm_blocked_percent_against"] = (field_goal_attempts["blocked_percent_against"] - min_blocked_percent_against) \
        / (max_blocked_percent_against - min_blocked_percent_against)
    min_blocked_percent = field_goal_attempts["blocked_percent"].min()
    max_blocked_percent = field_goal_attempts["blocked_percent"].max()
    field_goal_attempts["norm_blocked_percent"] = (field_goal_attempts["blocked_percent"] - min_blocked_percent) \
        / (max_blocked_percent - min_blocked_percent)

    # Calculate and normalize the field goal diffs
    field_goal_attempts["diff_field_goal_percent"] = field_goal_attempts["norm_field_goal_percent"] \
        - field_goal_attempts["norm_field_goal_percent_against"]
    min_diff_fg_percent = field_goal_attempts["diff_field_goal_percent"].min()
    max_diff_fg_percent = field_goal_attempts["diff_field_goal_percent"].max()
    field_goal_attempts["norm_diff_field_goal_percent"] = (field_goal_attempts["diff_field_goal_percent"] - min_diff_fg_percent) \
        / (max_diff_fg_percent - min_diff_fg_percent)
    field_goal_attempts["diff_blocked_percent"] = field_goal_attempts["norm_blocked_percent_against"] \
        - field_goal_attempts["norm_blocked_percent"]
    min_diff_blocked_percent = field_goal_attempts["diff_blocked_percent"].min()
    max_diff_blocked_percent = field_goal_attempts["diff_blocked_percent"].max()
    field_goal_attempts["norm_diff_blocked_percent"] = (field_goal_attempts["diff_blocked_percent"] - min_diff_blocked_percent) \
        / (max_diff_blocked_percent - min_diff_blocked_percent)

    # Drop irrelevant columns if requested
    if clean_columns:
        field_goal_attempts = field_goal_attempts[
            [
                "yardline_100",
                "norm_field_goal_percent",
                "norm_diff_field_goal_percent",
                "norm_diff_blocked_percent",
                "field_goal_attempt",
                "desc",
                "field_goal_result",
                "return_yards",
                "play_duration"
            ]
        ]
    return field_goal_attempts

def load_clean_nfl_pbp_run_data(
        years: list[int]=NFL_PBP_YEARS,
        clean_columns: bool=True
    ):
    """
    Loads historical NFL play-by-play data and cleans it for training the
    run result model

    Args:
        years (list[int]): The years of play-by-play data to load
        clean_columns (bool): Whether to drop irrelevant columns
    
    Returns:
        pd.DataFrame: The loaded & cleaned historicla NFL rushing data
    """
    # Load NFL play-by-play data
    df = nfl.import_pbp_data(years, cache=False, alt_path=None)

    # Derive score differential column
    df["score_diff"] = df["posteam_score"] - df["defteam_score"]

    # Derive play duration and drop outliers
    df['game_seconds_next'] = df.groupby('game_id')['game_seconds_remaining'].shift(-1)
    df['play_duration'] = df['game_seconds_remaining'] - df['game_seconds_next']
    df = df.query('play_duration < 69.0')
    df = df.query('play_duration >= 0')

    # Derive whether the penalty was committed by the posteam
    df['posteam_penalty'] = 0.0
    df.loc[(df["penalty"] == True) & (df["penalty_team"] == df["posteam"]), "posteam_penalty"] = 1.0

    # Clean null penalty yards values
    df['penalty_yards'] = df['penalty_yards'].fillna(0)
    df = df.dropna(subset=['down'])

    # Derive the rushing, rush defense, blocking, blitzing, turnover properties
    rush_attempts = df.query("rush_attempt == 1")
    season_posteam_groups = rush_attempts.groupby(["season", "posteam"])
    season_defteam_groups = rush_attempts.groupby(["season", "defteam"])

    # Loop through each group, get rushing, blocking, turnover properties
    for groups, group in season_posteam_groups:
        # Rushing
        total_rush_attempts = len(group)
        total_rushing_yards = group['rushing_yards'].sum()
        rushing = total_rushing_yards / total_rush_attempts
        rush_attempts.loc[
            (rush_attempts['season'] == groups[0]) & (rush_attempts['posteam'] == groups[1]),
            "rushing"
        ] = rushing

        # Run blocking
        tackled_for_loss = len(
            group.query('tackled_for_loss == 1')
        )
        blocking = 1 - (tackled_for_loss / total_rush_attempts)
        rush_attempts.loc[
            (rush_attempts['season'] == groups[0]) & (rush_attempts['posteam'] == groups[1]),
            "run_blocking"
        ] = blocking

        # Ball handling
        total_rushing_turnovers = len(
            group.query('fumble == 1')
        )
        ball_handling = 1 - (total_rushing_turnovers / total_rush_attempts)
        rush_attempts.loc[
            (rush_attempts['season'] == groups[0]) & (rush_attempts['posteam'] == groups[1]),
            "ball_handling"
        ] = ball_handling

        # Rushing penalties
        total_rushing_penalties = len(
            group.query(f'penalty == 1 and penalty_team == "{groups[1]}"')
        )
        rushing_penalties = 1 - (total_rushing_penalties / total_rush_attempts)
        rush_attempts.loc[
            (rush_attempts['season'] == groups[0]) & (rush_attempts['posteam'] == groups[1]),
            "rushing_penalties"
        ] = rushing_penalties
    for groups, group in season_defteam_groups:
        # Rush defense
        total_rush_attempts_against = len(group)
        total_rushing_yards_against = group['rushing_yards'].sum()
        rush_defense = total_rushing_yards_against / total_rush_attempts_against
        rush_attempts.loc[
            (rush_attempts['season'] == groups[0]) & (rush_attempts['defteam'] == groups[1]),
            "rush_defense"
        ] = rush_defense

        # Rush blitzing
        rushing_tackle_for_loss = len(
            group.query('tackled_for_loss == 1')
        )
        rush_blitzing = rushing_tackle_for_loss / total_rush_attempts_against
        rush_attempts.loc[
            (rush_attempts['season'] == groups[0]) & (rush_attempts['defteam'] == groups[1]),
            "rush_blitzing"
        ] = rush_blitzing

        # Forced fumbles
        total_forced_fumbles = len(
            group.query('fumble == 1')
        )
        forced_fumbles = total_forced_fumbles / total_rush_attempts_against
        rush_attempts.loc[
            (rush_attempts['season'] == groups[0]) & (rush_attempts['defteam'] == groups[1]),
            "forced_fumbles"
        ] = forced_fumbles

        # Rush defense penalties
        total_defensive_penalties = len(
            group.query(f'penalty == 1 and penalty_team == "{groups[1]}"')
        )
        rush_defense_penalties = 1 - (total_defensive_penalties / total_rush_attempts_against)
        rush_attempts.loc[
            (rush_attempts['season'] == groups[0]) & (rush_attempts['defteam'] == groups[1]),
            "rush_defense_penalties"
        ] = rush_defense_penalties
    
    # Normalize each offensive skill property
    min_rushing = rush_attempts["rushing"].min()
    max_rushing = rush_attempts["rushing"].max()
    rush_attempts["norm_rushing"] = (rush_attempts["rushing"] - min_rushing) \
        / (max_rushing - min_rushing)
    min_blocking = rush_attempts["run_blocking"].min()
    max_blocking = rush_attempts["run_blocking"].max()
    rush_attempts["norm_run_blocking"] = (rush_attempts["run_blocking"] - min_blocking) \
        / (max_blocking - min_blocking)
    min_handling = rush_attempts["ball_handling"].min()
    max_handling = rush_attempts["ball_handling"].max()
    rush_attempts["norm_ball_handling"] = (rush_attempts["ball_handling"] - min_handling) \
        / (max_handling - min_handling)
    min_off_penalties = rush_attempts["rushing_penalties"].min()
    max_off_penalties = rush_attempts["rushing_penalties"].max()
    rush_attempts["norm_rushing_penalties"] = (rush_attempts["rushing_penalties"] - min_off_penalties) \
        / (max_off_penalties - min_off_penalties)
    
    # Normalize each defensive skill property
    min_rush_defense = rush_attempts["rush_defense"].min()
    max_rush_defense = rush_attempts["rush_defense"].max()
    rush_attempts["norm_rush_defense"] = 1 \
        - ((rush_attempts["rush_defense"] - min_rush_defense) \
        / (max_rush_defense - min_rush_defense))
    min_blitzing = rush_attempts["rush_blitzing"].min()
    max_blitzing = rush_attempts["rush_blitzing"].max()
    rush_attempts["norm_rush_blitzing"] = (rush_attempts["rush_blitzing"] - min_blitzing) \
        / (max_blitzing - min_blitzing)
    min_ff = rush_attempts["forced_fumbles"].min()
    max_ff = rush_attempts["forced_fumbles"].max()
    rush_attempts["norm_forced_fumbles"] = (rush_attempts["forced_fumbles"] - min_ff) \
        / (max_ff - min_ff)
    min_def_penalties = rush_attempts["rush_defense_penalties"].min()
    max_def_penalties = rush_attempts["rush_defense_penalties"].max()
    rush_attempts["norm_rush_defense_penalties"] = (rush_attempts["rush_defense_penalties"] - min_def_penalties) \
        / (max_def_penalties - min_def_penalties)

    # Calculate the normalized skill diffs for relevant properties
    rush_attempts["diff_rushing"] = rush_attempts["rushing"] - rush_attempts["rush_defense"]
    min_diff_rushing = rush_attempts["diff_rushing"].min()
    max_diff_rushing = rush_attempts["diff_rushing"].max()
    rush_attempts["norm_diff_rushing"] = (rush_attempts["diff_rushing"] - min_diff_rushing) \
        / (max_diff_rushing - min_diff_rushing)
    rush_attempts["diff_blocking_blitzing"] = rush_attempts["run_blocking"] - rush_attempts["rush_blitzing"]
    min_diff_blocking = rush_attempts["diff_blocking_blitzing"].min()
    max_diff_blocking = rush_attempts["diff_blocking_blitzing"].max()
    rush_attempts["norm_diff_blocking_blitzing"] = (rush_attempts["diff_blocking_blitzing"] - min_diff_blocking) \
        / (max_diff_blocking - min_diff_blocking)
    rush_attempts["diff_ball_handling"] = rush_attempts["norm_ball_handling"] - rush_attempts["norm_forced_fumbles"]
    min_diff_handling = rush_attempts["diff_ball_handling"].min()
    max_diff_handling = rush_attempts["diff_ball_handling"].max()
    rush_attempts["norm_diff_ball_handling"] = (rush_attempts["diff_ball_handling"] - min_diff_handling) \
        / (max_diff_handling - min_diff_handling)
    
    # Drop irrelevant columns if requested
    if clean_columns:
        rush_attempts = rush_attempts[
            [
                # Game context
                "qtr",
                "half_seconds_remaining",
                "down",
                "ydstogo",
                "yardline_100",
                "defteam_timeouts_remaining",
                "posteam_timeouts_remaining",
                "score_diff",
                "goal_to_go",

                # Team skill levels and skill diffs
                "norm_diff_rushing",
                "norm_diff_blocking_blitzing",
                "norm_diff_ball_handling",
                "norm_rushing_penalties",
                "norm_rush_defense_penalties",

                # Rush result
                "play_duration",
                "yards_gained",
                "penalty",
                "posteam_penalty",
                "penalty_yards",
                "fumble",
                "return_yards",
                "touchdown"
            ]
        ]
    return rush_attempts

def load_clean_nfl_pbp_pass_data(
        years: list[int]=NFL_PBP_YEARS,
        clean_columns: bool=True
    ):
    df = nfl.import_pbp_data(years, cache=False, alt_path=None)

    # Derive play duration and drop outliers
    df['game_seconds_next'] = df.groupby('game_id')['game_seconds_remaining'].shift(-1)
    df['play_duration'] = df['game_seconds_remaining'] - df['game_seconds_next']
    df = df.query('play_duration < 69.0')
    df = df.query('play_duration >= 0')

    pass_attempts = df.query("pass_attempt == 1 or qb_scramble == 1")

    # Label with raw skill levels
    season_posteam_groups = pass_attempts.groupby(["season", "posteam"])
    season_defteam_groups = pass_attempts.groupby(["season", "defteam"])
    for groups, group in season_posteam_groups:
        # Pass blocking
        total_pass_attempts = len(group)
        pressures = len(
            group.query('qb_hit == 1 or sack == 1 or tackled_for_loss == 1')
        )
        blocking = 1 - (pressures / total_pass_attempts)
        pass_attempts.loc[
            (pass_attempts['season'] == groups[0]) & (pass_attempts['posteam'] == groups[1]),
            "pass_blocking"
        ] = blocking

        # Scrambling
        scrambles = len(
            group.query('qb_scramble == 1')
        )
        scrambling = scrambles / total_pass_attempts
        pass_attempts.loc[
            (pass_attempts['season'] == groups[0]) & (pass_attempts['posteam'] == groups[1]),
            "scrambling"
        ] = scrambling

        # Passing
        completions = group.query("complete_pass == 1")
        total_pass_completions = len(completions)
        total_pass_yards = group['passing_yards'].sum()
        total_pass_touchdowns = completions['pass_touchdown'].sum()
        total_pass_interceptions = group['interception'].sum()
        passing = ((
            ( # a
                ((total_pass_completions / total_pass_attempts) - 0.3) * 5
            ) +
            ( # b
                ((total_pass_yards / total_pass_attempts) - 3) * 0.25
            ) +
            ( # c
                (total_pass_touchdowns / total_pass_attempts) * 20
            ) +
            ( # d
                2.375 - ((total_pass_interceptions / total_pass_attempts) * 25)
            )
        ) / 6) * 100
        pass_attempts.loc[
            (pass_attempts['season'] == groups[0]) & (pass_attempts['posteam'] == groups[1]),
            "passing"
        ] = passing

        # Receiving
        total_yac = group["yards_after_catch"].sum()
        yac_per_completion = total_yac / total_pass_completions
        pass_attempts.loc[
            (pass_attempts['season'] == groups[0]) & (pass_attempts['posteam'] == groups[1]),
            "receiving"
        ] = yac_per_completion

        # Interceptions
        interceptions = len(group.query("interception == 1"))
        pass_interceptions = 1 - (interceptions / total_pass_attempts)
        pass_attempts.loc[
            (pass_attempts['season'] == groups[0]) & (pass_attempts['posteam'] == groups[1]),
            "pass_interceptions"
        ] = pass_interceptions
    for groups, group in season_defteam_groups:
        # Pass rushing
        total_pass_attempts_against = len(group)
        pressures_against = len(
            group.query('qb_hit == 1 or sack == 1 or tackled_for_loss == 1')
        )
        blitzing = pressures_against / total_pass_attempts_against
        pass_attempts.loc[
            (pass_attempts['season'] == groups[0]) & (pass_attempts['defteam'] == groups[1]),
            "pass_rushing"
        ] = blitzing

        # Pass defense
        completions = group.query("complete_pass == 1")
        total_pass_completions = len(completions)
        total_pass_yards = group['passing_yards'].sum()
        total_pass_touchdowns = completions['pass_touchdown'].sum()
        total_pass_interceptions = group['interception'].sum()
        pass_defense = ((
            ( # a
                ((total_pass_completions / total_pass_attempts) - 0.3) * 5
            ) +
            ( # b
                ((total_pass_yards / total_pass_attempts) - 3) * 0.25
            ) +
            ( # c
                (total_pass_touchdowns / total_pass_attempts) * 20
            ) +
            ( # d
                2.375 - ((total_pass_interceptions / total_pass_attempts) * 25)
            )
        ) / 6) * 100
        pass_attempts.loc[
            (pass_attempts['season'] == groups[0]) & (pass_attempts['defteam'] == groups[1]),
            "pass_defense"
        ] = pass_defense

        # Receiving
        total_yac_against = group["yards_after_catch"].sum()
        yac_per_completion_against = total_yac_against / total_pass_completions
        pass_attempts.loc[
            (pass_attempts['season'] == groups[0]) & (pass_attempts['defteam'] == groups[1]),
            "coverage"
        ] = yac_per_completion_against

        # Interceptions
        interceptions = len(group.query("interception == 1"))
        def_interceptions = interceptions / total_pass_attempts
        pass_attempts.loc[
            (pass_attempts['season'] == groups[0]) & (pass_attempts['defteam'] == groups[1]),
            "def_interceptions"
        ] = def_interceptions

    # Label with normalized skill levels
    min_passing = pass_attempts["passing"].min()
    max_passing = pass_attempts["passing"].max()
    pass_attempts["norm_passing"] = (pass_attempts["passing"] - min_passing) \
        / (max_passing - min_passing)
    min_receiving = pass_attempts["receiving"].min()
    max_receiving = pass_attempts["receiving"].max()
    pass_attempts["norm_receiving"] = (pass_attempts["receiving"] - min_receiving) \
        / (max_receiving - min_receiving)
    min_blocking = pass_attempts["pass_blocking"].min()
    max_blocking = pass_attempts["pass_blocking"].max()
    pass_attempts["norm_pass_blocking"] = (pass_attempts["pass_blocking"] - min_blocking) \
        / (max_blocking - min_blocking)
    min_scrambling = pass_attempts["scrambling"].min()
    max_scrambling = pass_attempts["scrambling"].max()
    pass_attempts["norm_scrambling"] = (pass_attempts["scrambling"] - min_scrambling) \
        / (max_scrambling - min_scrambling)
    min_pass_int = pass_attempts["pass_interceptions"].min()
    max_pass_int = pass_attempts["pass_interceptions"].max()
    pass_attempts["norm_pass_interceptions"] = (pass_attempts["pass_interceptions"] - min_pass_int) \
        / (max_pass_int - min_pass_int)
    min_pass_defense = pass_attempts["pass_defense"].min()
    max_pass_defense = pass_attempts["pass_defense"].max()
    pass_attempts["norm_pass_defense"] = 1 - (
        (pass_attempts["pass_defense"] - min_pass_defense) \
            / (max_pass_defense - min_pass_defense)
    )
    min_coverage = pass_attempts["coverage"].min()
    max_coverage = pass_attempts["coverage"].max()
    pass_attempts["norm_coverage"] = 1 - (
        (pass_attempts["coverage"] - min_coverage) \
            / (max_coverage - min_coverage)
    )
    min_blitzing = pass_attempts["pass_rushing"].min()
    max_blitzing = pass_attempts["pass_rushing"].max()
    pass_attempts["norm_pass_rushing"] = (pass_attempts["pass_rushing"] - min_blitzing) \
        / (max_blitzing - min_blitzing)
    min_def_int = pass_attempts["def_interceptions"].min()
    max_def_int = pass_attempts["def_interceptions"].max()
    pass_attempts["norm_def_interceptions"] = (pass_attempts["def_interceptions"] - min_def_int) \
        / (max_def_int - min_def_int)

    # Label with normalized skill differentials
    pass_attempts["diff_passing"] = pass_attempts["norm_passing"] - pass_attempts["norm_pass_defense"]
    min_diff_passing = pass_attempts["diff_passing"].min()
    max_diff_passing = pass_attempts["diff_passing"].max()
    pass_attempts["norm_diff_passing"] = (pass_attempts["diff_passing"] - min_diff_passing) \
        / (max_diff_passing - min_diff_passing)
    pass_attempts["diff_receiving"] = pass_attempts["norm_receiving"] - pass_attempts["norm_coverage"]
    min_diff_receiving = pass_attempts["diff_receiving"].min()
    max_diff_receiving = pass_attempts["diff_receiving"].max()
    pass_attempts["norm_diff_receiving"] = (pass_attempts["diff_receiving"] - min_diff_receiving) \
        / (max_diff_receiving - min_diff_receiving)
    pass_attempts["diff_pass_blocking_rushing"] = pass_attempts["norm_pass_blocking"] - pass_attempts["norm_pass_rushing"]
    min_diff_blocking = pass_attempts["diff_pass_blocking_rushing"].min()
    max_diff_blocking = pass_attempts["diff_pass_blocking_rushing"].max()
    pass_attempts["norm_diff_pass_blocking_rushing"] = (pass_attempts["diff_pass_blocking_rushing"] - min_diff_blocking) \
        / (max_diff_blocking - min_diff_blocking)
    pass_attempts["diff_interceptions"] = pass_attempts["norm_pass_interceptions"] - pass_attempts["norm_def_interceptions"]
    min_diff_int = pass_attempts["diff_interceptions"].min()
    max_diff_int = pass_attempts["diff_interceptions"].max()
    pass_attempts["norm_diff_interceptions"] = (pass_attempts["diff_interceptions"] - min_diff_int) \
        / (max_diff_int - min_diff_int)

    # Return the dataframe
    if clean_columns:
        return pass_attempts[
            [
                # Play context
                "yardline_100",
                
                # Skill levels
                "norm_passing",
                "norm_receiving",
                "norm_pass_blocking",
                "norm_scrambling",
                "norm_pass_interceptions",
                "norm_pass_defense",
                "norm_coverage",
                "norm_pass_rushing",
                "norm_def_interceptions",

                # Skill differentials
                "norm_diff_passing",
                "norm_diff_receiving",
                "norm_diff_pass_blocking_rushing",
                "norm_diff_interceptions",

                # Pass result columns
                "qb_hit",
                "sack",
                "tackled_for_loss",
                "qb_scramble",
                "pass_attempt",
                "incomplete_pass",
                "interception",
                "fumble",
                "air_yards",
                "yards_after_catch",
                "return_yards",
                "pass_length",
                "play_duration",
                "yards_gained",

                # Debug
                "desc"
            ]
        ]
    return pass_attempts

def load_clean_nfl_pbp_punt_data(
        years: list[int]=NFL_PBP_YEARS
    ):
    """
    Loads historical NFL play-by-play data and cleans it for training the
    punt result model
    """
    # Load NFL play-by-play data
    df = nfl.import_pbp_data(NFL_PBP_YEARS, cache=False, alt_path=None)

    # Derive play duration and drop outliers
    df['game_seconds_next'] = df.groupby('game_id')['game_seconds_remaining'].shift(-1)
    df['play_duration'] = df['game_seconds_remaining'] - df['game_seconds_next']
    df = df.query('play_duration < 69.0')
    df = df.query('play_duration >= 0')

    punt_plays = df.query("punt_attempt == 1")

    # Punting and punt return skill levels
    # Punting skill => percentage of punts inside twenty
    # Return defense skill => Return yards per punt return against
    # Returning skill => Return yards per punt return
    season_posteam_groups = punt_plays.groupby(["season", "posteam"])
    season_defteam_groups = punt_plays.groupby(["season", "defteam"])
    for groups, group in season_posteam_groups:
        # Derive punting skill
        total_punt_attempts = len(group)
        punts_inside_twenty = len(
            group.query('punt_inside_twenty == 1')
        )
        punting = 1 - (punts_inside_twenty / total_punt_attempts)
        punt_plays.loc[
            (punt_plays['season'] == groups[0]) & (punt_plays['posteam'] == groups[1]),
            "punting"
        ] = punting

        # Derive return defense skill
        punt_returns = group.query(
            'punt_downed == 0 and punt_fair_catch == 0 and touchback == 0 and punt_out_of_bounds == 0'
        )
        total_punt_returns = len(punt_returns)
        total_return_yards = punt_returns['return_yards'].sum()
        return_defense = total_return_yards / total_punt_returns
        punt_plays.loc[
            (punt_plays['season'] == groups[0]) & (punt_plays['posteam'] == groups[1]),
            "return_defense"
        ] = return_defense
    for groups, group in season_defteam_groups:
        # Derive blitzing skill
        total_punts = len(group)
        total_punt_blocks = len(group.query('punt_blocked == 1'))
        blitzing = total_punt_blocks / total_punts
        punt_plays.loc[
            (punt_plays['season'] == groups[0]) & (punt_plays['posteam'] == groups[1]),
            "blitzing"
        ] = blitzing

        # Derive returning skill
        punt_returns = group.query(
            'punt_downed == 0 and punt_fair_catch == 0 and touchback == 0 and punt_out_of_bounds == 0'
        )
        total_punt_returns = len(punt_returns)
        total_return_yards = punt_returns['return_yards'].sum()
        returning = total_return_yards / total_punt_returns
        punt_plays.loc[
            (punt_plays['season'] == groups[0]) & (punt_plays['defteam'] == groups[1]),
            "returning"
        ] = returning

    # Normalize punting, returning, return defense
    min_punting = punt_plays["punting"].min()
    max_punting = punt_plays["punting"].max()
    punt_plays["norm_punting"] = 1 - (
        (punt_plays["punting"] - min_punting) \
        / (max_punting - min_punting)
    )
    min_blitzing = punt_plays["blitzing"].min()
    max_blitzing = punt_plays["blitzing"].max()
    punt_plays["norm_blitzing"] = 1 - (
        (punt_plays["blitzing"] - min_blitzing) \
        / (max_blitzing - min_blitzing)
    )
    min_returning = punt_plays["returning"].min()
    max_returning = punt_plays["returning"].max()
    punt_plays["norm_returning"] = (punt_plays["returning"] - min_returning) \
        / (max_returning - min_returning)
    min_return_defense = punt_plays["return_defense"].min()
    max_return_defense = punt_plays["return_defense"].max()
    punt_plays["norm_return_defense"] = 1 - (
        (punt_plays["return_defense"] - min_return_defense) \
        / (max_return_defense - min_return_defense)
    )

    # Derive norm diff returning
    punt_plays["diff_returning"] = punt_plays["norm_returning"] - punt_plays["norm_return_defense"]
    min_diff_returning = punt_plays["diff_returning"].min()
    max_diff_returning = punt_plays["diff_returning"].max()
    punt_plays["norm_diff_returning"] = (punt_plays["diff_returning"] - min_diff_returning) \
        / (max_diff_returning - min_diff_returning)

    return punt_plays[
        [
            # Play context
            "yardline_100",

            # Skill levels
            "norm_blitzing",
            "norm_punting",
            "norm_diff_returning",

            # Punt result columns
            "kick_distance",
            "return_yards",
            "out_of_bounds",
            "fumble",
            "punt_out_of_bounds",
            "punt_blocked",
            "punt_in_endzone",
            "touchback",
            "punt_inside_twenty",
            "punt_fair_catch",
            "punt_downed",
            "play_duration",

            # Debug
            "desc"
        ]
    ]

def load_clean_nfl_pbp_kickoff_data():
    """
    Loads historical NFL play-by-play data and cleans it for training the
    kickoff result model

    Args:
        years (list[int]): The years of kickoff data to load
    
    Returns:
        pd.DataFrame: The loaded & cleaned historical NFL kickoff data
    """
    # Load NFL play-by-play data
    df = nfl.import_pbp_data(NFL_PBP_YEARS, cache=False, alt_path=None)
    
    # Derive play duration and drop outliers
    df['game_seconds_next'] = df.groupby('game_id')['game_seconds_remaining'].shift(-1)
    df['play_duration'] = df['game_seconds_remaining'] - df['game_seconds_next']
    df = df.query('play_duration < 69.0')
    df = df.query('play_duration >= 0')
    kickoff_plays = df.query("kickoff_attempt == 1")

    # Kickoff and kick return skill groups
    season_posteam_groups = kickoff_plays.groupby(["season", "posteam"])
    season_defteam_groups = kickoff_plays.groupby(["season", "defteam"])
    for groups, group in season_posteam_groups:
        # Derive returning skill
        total_return_attempts = len(group)
        total_return_yards = group["return_yards"].sum()
        returning = total_return_yards / total_return_attempts
        kickoff_plays.loc[
            (kickoff_plays['season'] == groups[0]) & (kickoff_plays['posteam'] == groups[1]),
            "returning"
        ] = returning
    for groups, group in season_defteam_groups:
        # Derive kicking skill
        total_kickoff_attempts = len(group)
        total_touchbacks = len(
            group.query('touchback == 1')
        )
        kicking = total_touchbacks / total_kickoff_attempts
        kickoff_plays.loc[
            (kickoff_plays['season'] == groups[0]) & (kickoff_plays['defteam'] == groups[1]),
            "kicking"
        ] = kicking

        # Derive return defense skill
        total_return_yards_against = group["return_yards"].sum()
        return_defense = total_return_yards_against / total_kickoff_attempts
        kickoff_plays.loc[
            (kickoff_plays['season'] == groups[0]) & (kickoff_plays['defteam'] == groups[1]),
            "return_defense"
        ] = return_defense

    # Normalize kicking, returning, return defense
    min_kicking = kickoff_plays["kicking"].min()
    max_kicking = kickoff_plays["kicking"].max()
    kickoff_plays["norm_kicking"] = (kickoff_plays["kicking"] - min_kicking) / \
        (max_kicking - min_kicking)
    min_returning = kickoff_plays["returning"].min()
    max_returning = kickoff_plays["returning"].max()
    kickoff_plays["norm_returning"] = (kickoff_plays["returning"] - min_returning) / \
        (max_returning - min_returning)
    min_return_defense = kickoff_plays["return_defense"].min()
    max_return_defense = kickoff_plays["return_defense"].max()
    kickoff_plays["norm_return_defense"] = 1 - (
        (kickoff_plays["return_defense"] - min_return_defense) / \
        (max_return_defense - min_return_defense)
    )

    # Calculate norm diff returning
    kickoff_plays["diff_returning"] = kickoff_plays["norm_returning"] - kickoff_plays["norm_return_defense"]
    min_diff_returning = kickoff_plays["diff_returning"].min()
    max_diff_returning = kickoff_plays["diff_returning"].max()
    kickoff_plays["norm_diff_returning"] = (kickoff_plays["diff_returning"] - min_diff_returning) / \
        (max_diff_returning - min_diff_returning)
    return kickoff_plays[
        [
            # Skill levels
            "norm_kicking",
            "norm_diff_returning",

            # Kickoff result columns
            "kick_distance",
            "kickoff_inside_twenty",
            "kickoff_in_endzone",
            "kickoff_out_of_bounds",
            "kickoff_downed",
            "kickoff_fair_catch",
            "return_yards",
            "fumble",
            "touchback",
            "play_duration",

            # Debug
            "desc"
        ]
    ]

def load_clean_nfl_pbp_playresult_data(
        years: list[int]=NFL_PBP_YEARS
    ):
    """
    Loads historical NFL play-by-play data and cleans it for training the
    play result model

    Args:
        years (list[int]): The years of play-by-play data to load

    Returns:
        pd.DataFrame: The loaded & cleaned historical NFL play-by-play data
    """
    # Load NFL play-by-play data
    df = nfl.import_pbp_data(NFL_PBP_YEARS, cache=False, alt_path=None)

    # Loop through each season and team
    for s in df['season'].unique():
        print(f"Calculating performance properties for season {s}")
        for p in df.query(f'season == {s}')['posteam'].unique():
            if p == None:
                continue

            # Derive the blocking property
            total_offensive_plays = len(
                df.query(f'season == {s} and posteam == "{p}"')
            )
            tackled_for_loss = len(
                df.query(f'season == {s} and posteam == "{p}" and (tackled_for_loss == 1 or sack == 1 or qb_hit == 1)')
            )
            blocking = tackled_for_loss / total_offensive_plays
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "blocking"
            ] = blocking

            # Derive the rushing property
            rush_att = df.query(f'season == {s} and posteam == "{p}" and rush_attempt == 1')
            total_rush_attempts = len(rush_att)
            total_rushing_yards = rush_att['rushing_yards'].sum()
            rushing = total_rushing_yards / total_rush_attempts
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "rushing"
            ] = rushing

            # Derive the passing property
            pass_att = df.query(f'season == {s} and posteam == "{p}" and pass_attempt == 1')
            completions = df.query(f'season == {s} and posteam == "{p}" and pass_attempt == 1 and complete_pass == 1')
            total_pass_attempts = len(pass_att)
            total_pass_completions = len(completions)
            total_pass_yards = pass_att['passing_yards'].sum()
            total_pass_touchdowns = completions['pass_touchdown'].sum()
            total_pass_interceptions = pass_att['interception'].sum()
            passing = ((
                ( # a
                    ((total_pass_completions / total_pass_attempts) - 0.3) * 5
                ) +
                ( # b
                    ((total_pass_yards / total_pass_attempts) - 3) * 0.25
                ) +
                ( # c
                    (total_pass_touchdowns / total_pass_attempts) * 20
                ) +
                ( # d
                    2.375 - ((total_pass_interceptions / total_pass_attempts) * 25)
                )
            ) / 6) * 100
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "passing"
            ] = passing

            # Derive the receiving property
            total_incompletions = total_pass_attempts - total_pass_completions
            incompletions_per_attempt = total_incompletions / total_pass_attempts
            total_yards_after_catch = completions['yards_after_catch'].sum()
            yac_per_completion = total_yards_after_catch / total_pass_completions
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "incompletions_per_attempt"
            ] = incompletions_per_attempt
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "yac_per_completion"
            ] = yac_per_completion

            # Derive the scrambling property
            scramble_att = df.query(f'season == {s} and posteam == "{p}" and qb_scramble == 1')
            total_qb_scrambles = len(scramble_att)
            total_qb_scramble_rushing = scramble_att['rushing_yards'].sum()
            scrambles_per_play = total_qb_scrambles / total_offensive_plays
            yards_per_scramble = total_qb_scramble_rushing / total_qb_scrambles
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "scrambles_per_play"
            ] = scrambles_per_play
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "yards_per_scramble"
            ] = yards_per_scramble

            # Derive the offensive_turnovers property
            total_offensive_turnovers = len(
                df.query(f'season == {s} and posteam == "{p}" and (fumble == 1 or interception == 1)')
            )
            offensive_turnovers = total_offensive_turnovers / total_offensive_plays
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "offensive_turnovers"
            ] = offensive_turnovers

            # Derive the offensive_penalties property
            total_offensive_penalties = len(
                df.query(f'season == {s} and posteam == "{p}" and penalty == 1 and penalty_team == "{p}"')
            )
            offensive_penalties = total_offensive_penalties / total_offensive_plays
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "offensive_penalties"
            ] = offensive_penalties

            # Derive the blitzing property
            total_defensive_plays = len(
                df.query(f'season == {s} and defteam == "{p}"')
            )
            made_tackle_for_loss = len(
                df.query(f'season == {s} and defteam == "{p}" and (tackled_for_loss == 1 or sack == 1 or qb_hit == 1)')
            )
            blitzing = made_tackle_for_loss / total_defensive_plays
            df.loc[
                (df['season'] == s) & (df['defteam'] == p),
                "blitzing"
            ] = blitzing

            # Derive the rush defense property
            rush_att_against = df.query(f'season == {s} and defteam == "{p}" and rush_attempt == 1')
            total_rush_attempts_against = len(rush_att_against)
            total_rushing_yards_against = rush_att['rushing_yards'].sum()
            rush_defense = total_rushing_yards_against / total_rush_attempts_against
            df.loc[
                (df['season'] == s) & (df['defteam'] == p),
                "rush_defense"
            ] = rush_defense

            # Derive the pass defense property
            pass_att_against = df.query(f'season == {s} and defteam == "{p}" and pass_attempt == 1')
            completions_against = df.query(f'season == {s} and defteam == "{p}" and pass_attempt == 1 and complete_pass == 1')
            total_pass_attempts_against = len(pass_att_against)
            total_pass_completions_against = len(completions_against)
            total_pass_yards_against = pass_att_against['passing_yards'].sum()
            total_pass_touchdowns_against = completions_against['pass_touchdown'].sum()
            total_interceptions_for = pass_att_against['interception'].sum()
            pass_defense = ((
                ( # a
                    ((total_pass_completions_against / total_pass_attempts_against) - 0.3) * 5
                ) +
                ( # b
                    ((total_pass_yards_against / total_pass_attempts_against) - 3) * 0.25
                ) +
                ( # c
                    (total_pass_touchdowns_against / total_pass_attempts_against) * 20
                ) +
                ( # d
                    2.375 - ((total_interceptions_for / total_pass_attempts_against) * 25)
                )
            ) / 6) * 100
            df.loc[
                (df['season'] == s) & (df['defteam'] == p),
                "pass_defense"
            ] = pass_defense

            # Derive the coverage property
            total_incompletions_against = total_pass_attempts_against - total_pass_completions_against
            incompletions_per_attempt_against = total_incompletions_against / total_pass_attempts_against
            total_yards_after_catch_against = completions_against['yards_after_catch'].sum()
            yac_per_completion_against = total_yards_after_catch_against / total_pass_completions_against
            df.loc[
                (df['season'] == s) & (df['defteam'] == p),
                "incompletions_per_attempt_against"
            ] = incompletions_per_attempt_against
            df.loc[
                (df['season'] == s) & (df['defteam'] == p),
                "yac_per_completion_against"
            ] = yac_per_completion_against

            # Derive the defensive_turnovers property
            total_defensive_turnovers = len(
                df.query(f'season == {s} and defteam == "{p}" and (fumble == 1 or interception == 1)')
            )
            defensive_turnovers = total_defensive_turnovers / total_defensive_plays
            df.loc[
                (df['season'] == s) & (df['defteam'] == p),
                "defensive_turnovers"
            ] = defensive_turnovers

            # Derive the defensive_penalties property
            total_defensive_penalties = len(
                df.query(f'season == {s} and defteam == "{p}" and penalty == 1 and penalty_team == "{p}"')
            )
            defensive_penalties = total_defensive_penalties / total_defensive_plays
            df.loc[
                (df['season'] == s) & (df['defteam'] == p),
                "defensive_penalties"
            ] = defensive_penalties

    # Normalize the blocking property
    min_blocking = df['blocking'].min()
    max_blocking = df['blocking'].max()
    df['norm_blocking'] = 1 - ((df['blocking'] - min_blocking) / (max_blocking - min_blocking))

    # Normalize the rushing property
    min_rushing = df['rushing'].min()
    max_rushing = df['rushing'].max()
    df['norm_rushing'] = (df['rushing'] - min_rushing) / (max_rushing - min_rushing)

    # Normalize the passing property
    min_passing = df['passing'].min()
    max_passing = df['passing'].max()
    df['norm_passing'] = (df['passing'] - min_passing) / (max_passing - min_passing)

    # Normalize the receiving properties and derive normalized receiving
    min_incompletions_per_attempt = df['incompletions_per_attempt'].min()
    max_incompletions_per_attempt = df['incompletions_per_attempt'].max()
    df['norm_incompletions_per_attempt'] = 1 - ((df['incompletions_per_attempt'] - min_incompletions_per_attempt) / (max_incompletions_per_attempt - min_incompletions_per_attempt))
    min_yac_per_completion = df['yac_per_completion'].min()
    max_yac_per_completion = df['yac_per_completion'].max()
    df['norm_yac_per_completion'] = (df['yac_per_completion'] - min_yac_per_completion) / (max_yac_per_completion - min_yac_per_completion)
    df['receiving'] = (df['norm_yac_per_completion'] + df['norm_incompletions_per_attempt']) / 2
    min_receiving = df['receiving'].min()
    max_receiving = df['receiving'].max()
    df['norm_receiving'] = (df['receiving'] - min_receiving) / (max_receiving - min_receiving)

    # Normalize the scrambling properties and derive normalized scrambling
    min_scrambles_per_play = df['scrambles_per_play'].min()
    max_scrambles_per_play = df['scrambles_per_play'].max()
    df['norm_scrambles_per_play'] = (df['scrambles_per_play'] - min_scrambles_per_play) / (max_scrambles_per_play - min_scrambles_per_play)
    min_yards_per_scramble = df['yards_per_scramble'].min()
    max_yards_per_scramble = df['yards_per_scramble'].max()
    df['norm_yards_per_scramble'] = (df['yards_per_scramble'] - min_yards_per_scramble) / (max_yards_per_scramble - min_yards_per_scramble)
    df['scrambling'] = (df['norm_scrambles_per_play'] + df['norm_yards_per_scramble']) / 2
    min_scrambling = df['scrambling'].min()
    max_scrambling = df['scrambling'].max()
    df['norm_scrambling'] = (df['scrambling'] - min_scrambling) / (max_scrambling - min_scrambling)

    # Normalize the offensive turnovers property
    min_offensive_turnovers = df['offensive_turnovers'].min()
    max_offensive_turnovers = df['offensive_turnovers'].max()
    df['norm_offensive_turnovers'] = 1 - ((df['offensive_turnovers'] - min_offensive_turnovers) / (max_offensive_turnovers - min_offensive_turnovers))

    # Normalize the offensive penalties property
    min_offensive_penalties = df['offensive_penalties'].min()
    max_offensive_penalties = df['offensive_penalties'].max()
    df['norm_offensive_penalties'] = 1 - ((df['offensive_penalties'] - min_offensive_penalties) / (max_offensive_penalties - min_offensive_penalties))

    # Normalize the blitzing property
    min_blitzing = df['blitzing'].min()
    max_blitzing = df['blitzing'].max()
    df['norm_blitzing'] = (df['blitzing'] - min_blitzing) / (max_blitzing - min_blitzing)

    # Normalize the rush_defense property
    min_rush_defense = df['rush_defense'].min()
    max_rush_defense = df['rush_defense'].max()
    df['norm_rush_defense'] = 1 - ((df['rush_defense'] - min_rush_defense) / (max_rush_defense - min_rush_defense))

    # Normalize the pass_defense property
    min_pass_defense = df['pass_defense'].min()
    max_pass_defense = df['pass_defense'].max()
    df['norm_pass_defense'] = 1 - ((df['pass_defense'] - min_pass_defense) / (max_pass_defense - min_pass_defense))

    # Normalize the coverage properties and derive normalized coverage
    min_incompletions_per_attempt_against = df['incompletions_per_attempt_against'].min()
    max_incompletions_per_attempt_against = df['incompletions_per_attempt_against'].max()
    df['norm_incompletions_per_attempt_against'] = (df['incompletions_per_attempt_against'] - min_incompletions_per_attempt_against) / (max_incompletions_per_attempt_against - min_incompletions_per_attempt_against)
    min_yac_per_completion_against = df['yac_per_completion_against'].min()
    max_yac_per_completion_against = df['yac_per_completion_against'].max()
    df['norm_yac_per_completion_against'] = 1 - ((df['yac_per_completion_against'] - min_yac_per_completion_against) / (max_yac_per_completion_against - min_yac_per_completion_against))
    df['coverage'] = (df['norm_yac_per_completion_against'] + df['norm_incompletions_per_attempt_against']) / 2
    min_coverage = df['coverage'].min()
    max_coverage = df['coverage'].max()
    df['norm_coverage'] = (df['coverage'] - min_coverage) / (max_coverage - min_coverage)

    # Normalize the defensive turnovers property
    min_defensive_turnovers = df['defensive_turnovers'].min()
    max_defensive_turnovers = df['defensive_turnovers'].max()
    df['norm_defensive_turnovers'] = (df['defensive_turnovers'] - min_defensive_turnovers) / (max_defensive_turnovers - min_defensive_turnovers)

    # Normalize the defensive penalties property
    min_defensive_penalties = df['defensive_penalties'].min()
    max_defensive_penalties = df['defensive_penalties'].max()
    df['norm_defensive_penalties'] = 1 - ((df['defensive_penalties'] - min_defensive_penalties) / (max_defensive_penalties - min_defensive_penalties))

    # Loop through each season and team again
    grouped_off = df.groupby(['season', 'posteam'])
    grouped_def = df.groupby(['season', 'defteam'])
    for s in df['season'].unique():
        print(f"Calculating unit overall properties for season {s}")
        for p in df.query(f'season == {s}')['posteam'].unique():
            if p == None:
                continue

            # Calculate the team's offensive overall rating
            norm_blocking = grouped_off.get_group((s, p))['norm_blocking'].mean()
            norm_rushing = grouped_off.get_group((s, p))['norm_rushing'].mean()
            norm_passing = grouped_off.get_group((s, p))['norm_passing'].mean()
            norm_receiving = grouped_off.get_group((s, p))['norm_receiving'].mean()
            norm_offensive_turnovers = grouped_off.get_group((s, p))['norm_offensive_turnovers'].mean()
            norm_offensive_penalties = grouped_off.get_group((s, p))['norm_offensive_penalties'].mean()
            offense_overall = np.average(
                [
                    norm_blocking,
                    norm_rushing,
                    norm_passing,
                    norm_receiving,
                    norm_offensive_turnovers,
                    norm_offensive_penalties
                ]
            )
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "offense_overall"
            ] = offense_overall

            # Calculate the team's defensive overall rating
            norm_blitzing = grouped_def.get_group((s, p))['norm_blitzing'].mean()
            norm_rush_defense = grouped_def.get_group((s, p))['norm_rush_defense'].mean()
            norm_pass_defense = grouped_def.get_group((s, p))['norm_pass_defense'].mean()
            norm_coverage = grouped_def.get_group((s, p))['norm_coverage'].mean()
            norm_defensive_turnovers = grouped_def.get_group((s, p))['norm_defensive_turnovers'].mean()
            norm_defensive_penalties = grouped_def.get_group((s, p))['norm_defensive_penalties'].mean()
            defense_overall = np.average(
                [
                    norm_blitzing,
                    norm_rush_defense,
                    norm_pass_defense,
                    norm_coverage,
                    norm_defensive_turnovers,
                    norm_defensive_penalties
                ]
            )
            df.loc[
                (df['season'] == s) & (df['defteam'] == p),
                "defense_overall"
            ] = defense_overall

    # Normalize the offensive and defensive overall properties
    min_offense_overall = df['offense_overall'].min()
    max_offense_overall = df['offense_overall'].max()
    df['norm_offense_overall'] = (df['offense_overall'] - min_offense_overall) / (max_offense_overall - min_offense_overall)
    min_defense_overall = df['defense_overall'].min()
    max_defense_overall = df['defense_overall'].max()
    df['norm_defense_overall'] = (df['defense_overall'] - min_defense_overall) / (max_defense_overall - min_defense_overall)

    # Loop through each season and team a third time
    grouped_off = df.groupby(['season', 'posteam'])
    grouped_def = df.groupby(['season', 'defteam'])
    for s in df['season'].unique():
        print(f"Calculating overall properties for season {s}")
        for p in df.query(f'season == {s}')['posteam'].unique():
            if p == None:
                continue

            # Get the team's offense and defense overall
            offense_overall = grouped_off.get_group((s, p))['norm_offense_overall'].mean()
            defense_overall = grouped_def.get_group((s, p))['norm_defense_overall'].mean()

            # Calculate the team's overall rating
            overall = np.average(
                [
                    offense_overall,
                    defense_overall
                ]
            )
            df.loc[
                (df['season'] == s) & (df['posteam'] == p),
                "overall"
            ] = overall

    # Normalize the team overall properties
    min_overall = df['overall'].min()
    max_overall = df['overall'].max()
    df['norm_overall'] = (df['overall'] - min_overall) / (max_overall - min_overall)

    ###
    # Model outputs
    ###

    # Derive play duration and drop outliers
    df['game_seconds_next'] = df.groupby('game_id')['game_seconds_remaining'].shift(-1)
    df['play_duration'] = df['game_seconds_remaining'] - df['game_seconds_next']
    df = df.query('play_duration < 69.0')
    df = df.query('play_duration >= 0')

    # Derive change of possession
    df['change_of_possession'] = 0.0
    df.loc[
        (df['interception'] == 1.0) |
        (df['fumble'] == 1.0) |
        (
            (df['down'] == 4.0) &
            (df['first_down'] != 1.0) & 
            (
                (df['field_goal_attempt'] == 0.0) |
                (
                    (df['field_goal_attempt'] == 1.0) &
                    (df['field_goal_result'] != 'made')
                )
            )
        ) |
        (df['punt_attempt'] == 1.0) |
        (df['kickoff_attempt'] == 1.0),
        'change_of_possession'
    ] = 1.0

    # Derive whether the penalty was committed by the posteam
    df['posteam_penalty'] = 0.0
    df.loc[(df["penalty"] == True) & (df["penalty_team"] == df["posteam"]), "posteam_penalty"] = 1.0

    # Derive whether the timeout was called by the posteam
    df['posteam_timeout'] = 0.0
    df.loc[(df["timeout"] == True) & (df["timeout_team"] == df["posteam"]), "posteam_timeout"] = 1.0

    # One-hot encode field goal result
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_fg_results = encoder.fit_transform(df[['field_goal_result']])
    encoded_fg_result_df = pd.DataFrame(encoded_fg_results, columns=encoder.get_feature_names_out(['field_goal_result']))
    df = pd.concat([df.drop('field_goal_result', axis=1), encoded_fg_result_df], axis=1)

    ###
    # Game context
    ###

    # Derive score differential column
    df["score_diff"] = df["posteam_score"] - df["defteam_score"]

    # Clean null values
    df = df[df['half_seconds_remaining'].notna()]
    df = df[df['defteam_timeouts_remaining'].notna()]
    df["down"] = df["down"].fillna(0)
    df = df[df["yardline_100"].notna()]
    df = df[df["play_type"].notna()]
    
    # Clean goal to go column
    df.loc[df["goal_to_go"] == 1, "goal_to_go"] = 1.0
    df.loc[df["goal_to_go"] == 0, "goal_to_go"] = 0.0

    # Combine pass length, run direction, and timeout into play type column
    df.loc[(df["play_type"] == "pass") & (df["pass_length"] == "short"), "play_type"] = "short_pass"
    df.loc[(df["play_type"] == "pass") & (df["pass_length"] == "deep"), "play_type"] = "deep_pass"
    df.loc[(df["play_type"] == "run") & (df["run_location"] == "left"), "play_type"] = "run_left"
    df.loc[(df["play_type"] == "run") & (df["run_location"] == "middle"), "play_type"] = "run_middle"
    df.loc[(df["play_type"] == "run") & (df["run_location"] == "right"), "play_type"] = "run_right"
    df.loc[(df["timeout"] == True) & (df["timeout_team"] == df["posteam"]), "play_type"] = "offense_timeout"
    df.loc[(df["timeout"] == True) & ~(df["timeout_team"] == df["posteam"]), "play_type"] = "defense_timeout"
    df = df[~(df["play_type"] == "no_play")]

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

    # One-hot encode play type
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_play_types = encoder.fit_transform(df[['play_type']])
    encoded_play_type_df = pd.DataFrame(encoded_play_types, columns=encoder.get_feature_names_out(['play_type']))
    df = pd.concat([df.drop('play_type', axis=1), encoded_play_type_df], axis=1)

    # Drop & clean null values
    df = df.drop(df[df['norm_rushing'].isnull()].index)
    df.loc[df['penalty_yards'].isnull(), 'penalty_yards'] = 0.0
    df.loc[df['field_goal_result_made'].isnull(), 'field_goal_result_made'] = 0.0
    df.loc[df['field_goal_result_missed'].isnull(), 'field_goal_result_missed'] = 0.0
    df.loc[df['field_goal_result_blocked'].isnull(), 'field_goal_result_blocked'] = 0.0
    df.loc[df['play_type_defense_timeout'].isnull(), 'play_type_defense_timeout'] = 0.0
    df.loc[df['play_type_offense_timeout'].isnull(), 'play_type_offense_timeout'] = 0.0
    df.loc[df['play_type_qb_spike'].isnull(), 'play_type_qb_spike'] = 0.0
    df.loc[df['play_type_qb_kneel'].isnull(), 'play_type_qb_kneel'] = 0.0
    df.loc[df['play_type_field_goal'].isnull(), 'play_type_field_goal'] = 0.0
    df.loc[df['play_type_extra_point'].isnull(), 'play_type_extra_point'] = 0.0
    df.loc[df['play_type_punt'].isnull(), 'play_type_punt'] = 0.0
    df.loc[df['play_type_kickoff'].isnull(), 'play_type_kickoff'] = 0.0
    df.loc[df['play_type_run_right'].isnull(), 'play_type_run_right'] = 0.0
    df.loc[df['play_type_run_middle'].isnull(), 'play_type_run_middle'] = 0.0
    df.loc[df['play_type_run_left'].isnull(), 'play_type_run_left'] = 0.0
    df.loc[df['play_type_deep_pass'].isnull(), 'play_type_deep_pass'] = 0.0
    df.loc[df['play_type_short_pass'].isnull(), 'play_type_short_pass'] = 0.0

    ###
    # Removing unnecessary properties
    ###
    df = df[
        [
            # Game context
            "qtr",
            "half_seconds_remaining",
            "down",
            "ydstogo",
            "yardline_100",
            "defteam_timeouts_remaining",
            "posteam_timeouts_remaining",
            "score_diff",
            "goal_to_go",

            # Play call (one-hot encoded)
            "play_type_short_pass",
            "play_type_deep_pass",
            "play_type_run_left",
            "play_type_run_middle",
            "play_type_run_right",
            "play_type_kickoff",
            "play_type_punt",
            "play_type_extra_point",
            "play_type_field_goal",
            "play_type_qb_kneel",
            "play_type_qb_spike",
            "play_type_offense_timeout",
            "play_type_defense_timeout",

            # Offense skill
            "norm_blocking",
            "norm_rushing",
            "norm_passing",
            "norm_receiving",
            "norm_scrambling",
            "norm_offensive_turnovers",
            "norm_offensive_penalties",

            # Defense skill
            "norm_blitzing",
            "norm_rush_defense",
            "norm_pass_defense",
            "norm_coverage",
            "norm_defensive_turnovers",
            "norm_defensive_penalties",

            # Play result
            "play_duration",
            "yards_gained",
            "first_down",
            "touchdown",
            "complete_pass",
            "out_of_bounds",
            "qb_scramble",
            "qb_hit",
            "sack",
            "tackled_for_loss",
            "fumble",
            "interception",
            "field_goal_result_blocked",
            "field_goal_result_made",
            "field_goal_result_missed",
            "penalty",
            # TODO: Eventually re-introduce
            #"penalty_type",
            "posteam_penalty",
            "penalty_yards",
            "timeout",
            "posteam_timeout",
            
            # Debug
            "desc"
        ]
    ]
    return df

def visualize_nfl_pbp_playresult_data(
        df: pd.DataFrame
    ):
    """
    Visualizes the distributions of the normalized properties derived by
    load_clean_nfl_pbp_playresult_data.  Writes the resulting matplotlib
    plots to the figures subdirectory as PNGs.

    Args:
        DataFrame: The DataFrame from load_clean_nfl_pbp_playresult_data
    """
    # Visualize the derived properties
    off_plot_df = df[
        [
            'season',
            'posteam',
            'norm_overall',
            'norm_offense_overall',
            'norm_blocking',
            'norm_rushing',
            'norm_passing',
            'norm_receiving',
            'norm_scrambling',
            'norm_offensive_turnovers',
            'norm_offensive_penalties'
        ]
    ].drop_duplicates()
    def_plot_df = df[
        [
            'season',
            'defteam',
            'norm_defense_overall',
            'norm_blitzing',
            'norm_rush_defense',
            'norm_pass_defense',
            'norm_coverage',
            'norm_defensive_turnovers',
            'norm_defensive_penalties'
        ]
    ].drop_duplicates()

    # Visualize the overall property
    off_plot_df['norm_overall'].hist(bins=20)
    plt.title("Team overall distribution")
    plt.savefig('./figures/overall.png')
    plt.clf()

    # Visualize the offense overall property
    off_plot_df['norm_offense_overall'].hist(bins=20)
    plt.title("Offense overall distribution")
    plt.savefig('./figures/offense_overall.png')
    plt.clf()

    # Visualize the defense overall property
    def_plot_df['norm_defense_overall'].hist(bins=20)
    plt.title("Defense overall distribution")
    plt.savefig('./figures/defense_overall.png')
    plt.clf()

    # Visualize the blocking property
    off_plot_df['norm_blocking'].hist(bins=20)
    plt.title("Blocking distribution")
    plt.savefig('./figures/blocking.png')
    plt.clf()

    # Visualize the rushing property
    off_plot_df['norm_rushing'].hist(bins=20)
    plt.title("Rushing distribution")
    plt.savefig('./figures/rushing.png')
    plt.clf()

    # Visualize the passing property
    off_plot_df['norm_passing'].hist(bins=20)
    plt.title("Passing distribution")
    plt.savefig('./figures/passing.png')
    plt.clf()

    # Visualize the receiving property
    off_plot_df['norm_receiving'].hist(bins=20)
    plt.title("Receiving distribution")
    plt.savefig('./figures/receiving.png')
    plt.clf()

    # Visualize the scrambling property
    off_plot_df['norm_scrambling'].hist(bins=20)
    plt.title("Scrambling distribution")
    plt.savefig('./figures/scrambling.png')
    plt.clf()

    # Visualize the offensive turnovers property
    off_plot_df['norm_offensive_turnovers'].hist(bins=20)
    plt.title("Offensive turnovers distribution")
    plt.savefig('./figures/off_turnovers.png')
    plt.clf()

    # Visualize the offensive penalties property
    off_plot_df['norm_offensive_penalties'].hist(bins=20)
    plt.title("Offensive penalties distribution")
    plt.savefig('./figures/off_penalties.png')
    plt.clf()

    # Visualize the blitzing property
    def_plot_df['norm_blitzing'].hist(bins=20)
    plt.title("Blitzing distribution")
    plt.savefig('./figures/blitzing.png')
    plt.clf()

    # Visualize the rush defense property
    def_plot_df['norm_rush_defense'].hist(bins=20)
    plt.title("Rush defense distribution")
    plt.savefig('./figures/rush_defense.png')
    plt.clf()

    # Visualize the pass defense property
    def_plot_df['norm_pass_defense'].hist(bins=20)
    plt.title("Pass defense distribution")
    plt.savefig('./figures/pass_defense.png')
    plt.clf()

    # Visualize the coverage property
    def_plot_df['norm_coverage'].hist(bins=20)
    plt.title("Coverage distribution")
    plt.savefig('./figures/coverage.png')
    plt.clf()

    # Visualize the defensive turnovers property
    def_plot_df['norm_defensive_turnovers'].hist(bins=20)
    plt.title("Defensive turnovers distribution")
    plt.savefig('./figures/def_turnovers.png')
    plt.clf()

    # Visualize the defensive penalties property
    def_plot_df['norm_defensive_penalties'].hist(bins=20)
    plt.title("Defensive penalties distribution")
    plt.savefig('./figures/def_penalties.png')
    plt.clf()
