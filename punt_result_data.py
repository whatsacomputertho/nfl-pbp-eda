from data.pbp import load_clean_nfl_pbp_punt_data

df = load_clean_nfl_pbp_punt_data()
df.to_csv('./data/punts.csv', index=False)
