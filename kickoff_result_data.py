from data.pbp import load_clean_nfl_pbp_kickoff_data

df = load_clean_nfl_pbp_kickoff_data()
df.to_csv('./data/kickoffs.csv', index=False)
