from data.pbp import load_clean_nfl_pbp_pass_data

df = load_clean_nfl_pbp_pass_data()
df.to_csv('./data/passing.csv')
