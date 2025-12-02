import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

###
# Between play model flow
# 1. Offense clock management strategy
#   - Up-tempo / no-huddle
#   - Normal
#   - Drain the clock
# 2. Is the clock running?
#   - Was the last play out of bounds?
#   - Was the last play an incomplete pass?
#   - Was the last play a change of possession?
#   - Does the offense call a timeout?
#   - Does the defense call a timeout?
# 3. Is the clock running immediately after the play?
#   - Was the last play a first down?
#   - Was the last play out of bounds outside of 2 mins?
# 4. Seconds between the play
###

df = pd.read_csv('./data/between_play.csv')
df["norm_average_play_duration"] = 1 - df["norm_average_play_duration"]
df["up_tempo_group"] = pd.cut(df["norm_average_play_duration"], bins=4)

# Offense clock management strategy
grouped_up_tempo = df.groupby("up_tempo_group")
def no_huddle_count(s):
    return (s == 1).sum()
grouped_up_tempo_averages = grouped_up_tempo["no_huddle"].agg(["count", no_huddle_count])
grouped_up_tempo_averages["p_up_tempo"] = grouped_up_tempo_averages["no_huddle_count"] / grouped_up_tempo_averages["count"]
print(grouped_up_tempo_averages)
print()
# NOTE: This is for normal plays
# NOTE: Offense will conditionally always call up-tempo / drain clock
# NOTE: Up-tempo scenarios:
# NOTE: - Losing by 2 or fewer scores and less than 3 minutes in the half
# NOTE: Drain clock scenarios:
# NOTE: - 4Q and clock is less than ((number of scores up by) * 4) mins

up_tempo_midpoints = [[(name.left + name.right)/2] for name in grouped_up_tempo_averages.index]
p_up_tempo_model = LinearRegression()
p_up_tempo_model.fit(
    up_tempo_midpoints,
    np.log(grouped_up_tempo_averages["p_up_tempo"])
)
zero_to_one = pd.DataFrame(np.linspace(0, 1))
p_up_tempo_pred = np.exp(p_up_tempo_model.predict(zero_to_one))
print("Up tempo probability model")
print(f"Coef: {p_up_tempo_model.coef_}")
print(f"Intr: {p_up_tempo_model.intercept_}")
print()

plt.scatter(up_tempo_midpoints, grouped_up_tempo_averages["p_up_tempo"], color='g')
plt.plot(zero_to_one, p_up_tempo_pred, color='b')
plt.title("Up-tempo probability by normalized average play duration")
plt.xlabel("Normalized average play duration")
plt.ylabel("Up-tempo probability")
plt.savefig("./figures/up_tempo_probability.png")
plt.clf()

# Difference in average seconds between the play when not immediate clock start
oob_non_clock_stop_plays = df[(df["prev_play_out_of_bounds"] == 1) & (df["half_seconds_remaining"] > 120)]
print("Difference in play duration when clock does not immediately start")
print(f"OOB:    {oob_non_clock_stop_plays['play_duration'].agg('mean')}")
print(f"Normal: {df['play_duration'].agg('mean')}")
print()
# NOTE: Not much of a difference, we'll ignore this

# Seconds between the play if clock running (no strategy)
print("Play duration distribution for normal plays")
print(df["play_duration"].agg(['mean', 'std', 'skew']))
print()
# NOTE: Not much skew, we will stick to mean & std
# NOTE: Going to adjust this
# NOTE: Mean: 20
# NOTE: Std:  5

# Seconds between the play if clock running and up-tempo
print("Play duration distribution for up-tempo plays")
print(df[df["no_huddle"] == 1]["prev_play_duration"].agg(['mean', 'std', 'skew']))
print()
# NOTE: Going to adjust this
# NOTE: Mean: 6
# NOTE: Std:  2

# Seconds between the play if clock running and draining the clock
# NOTE: 40 - exponential distribution centered at 1
