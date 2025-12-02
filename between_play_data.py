from data.pbp import load_clean_nfl_pbp_between_play_data

df = load_clean_nfl_pbp_between_play_data()
df.to_csv('./data/between_play.csv')
