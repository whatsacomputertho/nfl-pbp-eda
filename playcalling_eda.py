from data.pbp import load_clean_nfl_pbp_playcall_data

# Load the NFL data and split into training and test data
print("Loading NFL play-by-play data")
df = load_clean_nfl_pbp_playcall_data()

# Get the value counts of the play call column
print(df["play_type"].value_counts())
