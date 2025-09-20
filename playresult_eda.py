from data.pbp import load_clean_nfl_pbp_playresult_data, \
    visualize_nfl_pbp_playresult_data

# Load the NFL data and split into training and test data
print("Loading NFL play-by-play data")
df = load_clean_nfl_pbp_playresult_data()
#visualize_nfl_pbp_playresult_data(df)

columns = [
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
    "posteam_penalty",
    # TODO: Eventually re-introduce
    #"penalty_type",
    "penalty_yards",
    "timeout",
    "posteam_timeout",
]
for column in columns:
    print(f"Analyzing column: {column}")
    print(f"Null values: {df[column].isnull().sum()}")
    print(f"Value counts: {df[column].value_counts()}")
