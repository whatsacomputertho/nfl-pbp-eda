from data.pbp import load_clean_nfl_pbp_fieldgoal_data

df = load_clean_nfl_pbp_fieldgoal_data()
df.to_csv('./data/fgs.csv', index=False)
