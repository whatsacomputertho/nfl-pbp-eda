from data.pbp import load_clean_nfl_pbp_playcall_data

df = load_clean_nfl_pbp_playcall_data()
df.to_csv("./data/playcall.csv", index=False)
