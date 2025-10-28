from data.pbp import load_clean_nfl_pbp_run_data

df = load_clean_nfl_pbp_run_data()
df.to_csv('./data/rushing.csv')
