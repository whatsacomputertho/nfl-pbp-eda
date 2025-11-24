import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./data/kickoffs.csv')

###
# Kickoff model flow
# 1. Touchback?
# 2. Out of bounds?
# 3. Inside 20?
# 4. Kick distance
# 5. Fair catch?
# 6. Return yards
# 7. Fumble?
# 8. Duration
###

# 1. Touchback?
df["kicking_group"] = pd.cut(df["norm_kicking"], bins=4)
grouped_kicking = df.groupby("kicking_group")
def touchback_count(s):
    return (s == 1).sum()
grouped_touchback_averages = grouped_kicking["touchback"].agg(["count", touchback_count])
grouped_touchback_averages["p_touchback"] = grouped_touchback_averages["touchback_count"] / grouped_touchback_averages["count"]
print(grouped_touchback_averages)
print()

touchback_midpoints = [[(name.left + name.right)/2] for name in grouped_touchback_averages.index]
p_touchback_model = LinearRegression()
p_touchback_model.fit(
    touchback_midpoints,
    grouped_touchback_averages["p_touchback"]
)
zero_to_one = pd.DataFrame(np.linspace(0, 1))
p_touchback_pred = p_touchback_model.predict(zero_to_one)
print("Kickoff touchback probability by kicking skill")
print(f"Coef: {p_touchback_model.coef_}")
print(f"Intr: {p_touchback_model.intercept_}")
print()

plt.scatter(touchback_midpoints, grouped_touchback_averages["p_touchback"], color='g')
plt.plot(zero_to_one, p_touchback_pred, color='b')
plt.title("Probability of kickoff touchback by kicking skill")
plt.xlabel("Normalized kicking skill")
plt.ylabel("Probability of kickoff touchback")
plt.savefig('./figures/p_kickoff_touchback.png')
plt.clf()

# 2. Out of bounds?
def oob_count(s):
    return (s == 1).sum()
grouped_oob_averages = grouped_kicking["kickoff_out_of_bounds"].agg(["count", oob_count])
grouped_oob_averages["p_oob"] = grouped_oob_averages["oob_count"] / grouped_oob_averages["count"]
print(grouped_oob_averages)
print()

oob_midpoints = [[(name.left + name.right)/2] for name in grouped_oob_averages.index]
p_oob_model = LinearRegression()
p_oob_model.fit(
    oob_midpoints,
    grouped_oob_averages["p_oob"]
)
p_oob_pred = p_oob_model.predict(zero_to_one)
print("Kickoff out of bounds probability by kicking skill")
print(f"Coef: {p_oob_model.coef_}")
print(f"Intr: {p_oob_model.intercept_}")
print()

plt.scatter(oob_midpoints, grouped_oob_averages["p_oob"], color='g')
plt.plot(zero_to_one, p_oob_pred, color='b')
plt.title("Probability of kickoff out of bounds by kicking skill")
plt.xlabel("Normalized kicking skill")
plt.ylabel("Probability of kickoff out of bounds")
plt.savefig('./figures/p_kickoff_oob.png')
plt.clf()

# 3. Inside 20?
non_oob_tb_kicks = df[(df["touchback"] == 0) & (df["kickoff_out_of_bounds"] == 0)]
grouped_non_oob_tb_kicks = non_oob_tb_kicks.groupby("kicking_group")
def inside_20_count(s):
    return (s == 1).sum()
grouped_inside_20_averages = grouped_non_oob_tb_kicks["kickoff_inside_twenty"].agg(["count", inside_20_count])
grouped_inside_20_averages["p_inside_20"] = grouped_inside_20_averages["inside_20_count"] / grouped_inside_20_averages["count"]
print(grouped_inside_20_averages)
print()

inside_20_midpoints = [[(name.left + name.right)/2] for name in grouped_inside_20_averages.index]
p_inside_20_model = LinearRegression()
p_inside_20_model.fit(
    inside_20_midpoints,
    grouped_inside_20_averages["p_inside_20"]
)
p_inside_20_pred = p_inside_20_model.predict(zero_to_one)
print("Kickoff inside 20 probability model by kicking skill")
print(f"Coef: {p_inside_20_model.coef_}")
print(f"Intr: {p_inside_20_model.intercept_}")
print()
# NOTE: Constant 0.2 kickoff inside 20 probability

plt.scatter(inside_20_midpoints, grouped_inside_20_averages["p_inside_20"], color='g')
plt.plot(zero_to_one, p_inside_20_pred, color='b')
plt.title("Probability of kickoff inside 20 by kicking skill")
plt.xlabel("Normalized kicking skill")
plt.ylabel("Probability of kickoff inside 20")
plt.savefig('./figures/p_kickoff_inside_20.png')
plt.clf()

# 4. Kick distance
grouped_kicks_inside_20 = df[df["kickoff_inside_twenty"] == 1].groupby("kicking_group")
grouped_inside_20_distance_averages = grouped_kicks_inside_20["kick_distance"].agg(["mean", "std", "skew"])
print(grouped_inside_20_distance_averages)
print()

inside_20_group_midpoints = [[(name.left + name.right)/2] for name in grouped_inside_20_distance_averages.index]
mean_inside_20_dist_model = LinearRegression()
mean_inside_20_dist_model.fit(
    inside_20_group_midpoints,
    grouped_inside_20_distance_averages["mean"]
)
mean_inside_20_dist_pred = mean_inside_20_dist_model.predict(zero_to_one)
print("Mean kick distance model for kicks inside 20")
print(f"Coef: {mean_inside_20_dist_model.coef_}")
print(f"Intr: {mean_inside_20_dist_model.intercept_}")
print()
# NOTE: Constant 64.3 mean

std_inside_20_dist_model = LinearRegression()
std_inside_20_dist_model.fit(
    inside_20_group_midpoints,
    grouped_inside_20_distance_averages["std"]
)
std_inside_20_dist_pred = std_inside_20_dist_model.predict(zero_to_one)
print("Std kick distance model for kicks inside 20")
print(f"Coef: {std_inside_20_dist_model.coef_}")
print(f"Intr: {std_inside_20_dist_model.intercept_}")
print()

skew_inside_20_dist_model = LinearRegression()
skew_inside_20_dist_model.fit(
    inside_20_group_midpoints,
    grouped_inside_20_distance_averages["skew"]
)
skew_inside_20_dist_pred = skew_inside_20_dist_model.predict(zero_to_one)
print("Skew kick distance model for kicks inside 20")
print(f"Coef: {skew_inside_20_dist_model.coef_}")
print(f"Intr: {skew_inside_20_dist_model.intercept_}")
print()
# NOTE: Constant -1.7 skew

plt.scatter(inside_20_group_midpoints, grouped_inside_20_distance_averages["mean"], color='g')
plt.scatter(inside_20_group_midpoints, grouped_inside_20_distance_averages["std"], color='r')
plt.scatter(inside_20_group_midpoints, grouped_inside_20_distance_averages["skew"], color='y')
plt.plot(zero_to_one, mean_inside_20_dist_pred, color='b')
plt.plot(zero_to_one, std_inside_20_dist_pred, color='b')
plt.plot(zero_to_one, skew_inside_20_dist_pred, color='b')
plt.title("Kickoff distance by kicking skill for kicks inside 20")
plt.xlabel("Normalized kicking skill")
plt.ylabel("Kickoff distance")
plt.savefig('./figures/kickoff_distance_inside_20.png')
plt.clf()

grouped_kicks_outside_20 = df[(df["kickoff_inside_twenty"] == 0) & (df["touchback"] == 0)].groupby("kicking_group")
grouped_outside_20_distance_averages = grouped_kicks_outside_20["kick_distance"].agg(["mean", "std", "skew"])
print(grouped_outside_20_distance_averages)
print()

outside_20_group_midpoints = [[(name.left + name.right)/2] for name in grouped_outside_20_distance_averages.index]
mean_outside_20_dist_model = LinearRegression()
mean_outside_20_dist_model.fit(
    outside_20_group_midpoints,
    grouped_outside_20_distance_averages["mean"]
)
mean_outside_20_dist_pred = mean_outside_20_dist_model.predict(zero_to_one)
print("Mean kick distance model for kicks outside 20")
print(f"Coef: {mean_outside_20_dist_model.coef_}")
print(f"Intr: {mean_outside_20_dist_model.intercept_}")
print()

std_outside_20_dist_model = LinearRegression()
std_outside_20_dist_model.fit(
    outside_20_group_midpoints,
    grouped_outside_20_distance_averages["std"]
)
std_outside_20_dist_pred = std_outside_20_dist_model.predict(zero_to_one)
print("Std kick distance model for kicks outside 20")
print(f"Coef: {std_outside_20_dist_model.coef_}")
print(f"Intr: {std_outside_20_dist_model.intercept_}")
print()

skew_outside_20_dist_model = LinearRegression()
skew_outside_20_dist_model.fit(
    outside_20_group_midpoints,
    grouped_outside_20_distance_averages["skew"]
)
skew_outside_20_dist_pred = skew_outside_20_dist_model.predict(zero_to_one)
print("Skew kick distance model for kicks outside 20")
print(f"Coef: {skew_outside_20_dist_model.coef_}")
print(f"Intr: {skew_outside_20_dist_model.intercept_}")
print()
# NOTE: Constant -2 skew

plt.scatter(outside_20_group_midpoints, grouped_outside_20_distance_averages["mean"], color='g')
plt.scatter(outside_20_group_midpoints, grouped_outside_20_distance_averages["std"], color='r')
plt.scatter(outside_20_group_midpoints, grouped_outside_20_distance_averages["skew"], color='y')
plt.plot(zero_to_one, mean_outside_20_dist_pred, color='b')
plt.plot(zero_to_one, std_outside_20_dist_pred, color='b')
plt.plot(zero_to_one, skew_outside_20_dist_pred, color='b')
plt.title("Kickoff distance by kicking skill for kicks outside 20")
plt.xlabel("Normalized kicking skill")
plt.ylabel("Kickoff distance")
plt.savefig('./figures/kickoff_distance_outside_20.png')
plt.clf()

# 5. Fair catch?
df["returning_group"] = pd.cut(df["norm_diff_returning"], bins=4)
grouped_returning = df.groupby("returning_group")
def fair_catch_count(s):
    return (s == 1).sum()
grouped_fair_catch_averages = grouped_returning["kickoff_fair_catch"].agg(["count", fair_catch_count])
grouped_fair_catch_averages["p_fair_catch"] = grouped_fair_catch_averages["fair_catch_count"] / grouped_fair_catch_averages["count"]
print(grouped_fair_catch_averages)
print()

fair_catch_group_midpoints = [[(name.left + name.right)/2] for name in grouped_fair_catch_averages.index]
p_fair_catch_model = LinearRegression()
p_fair_catch_model.fit(
    fair_catch_group_midpoints,
    grouped_fair_catch_averages["p_fair_catch"]
)
p_fair_catch_pred = p_fair_catch_model.predict(zero_to_one)
print("Fair catch probability model")
print(f"Coef: {p_fair_catch_model.coef_}")
print(f"Intr: {p_fair_catch_model.intercept_}")
print()

plt.scatter(fair_catch_group_midpoints, grouped_fair_catch_averages["p_fair_catch"], color='g')
plt.plot(zero_to_one, p_fair_catch_pred, color='b')
plt.title("Kickoff fair catch probability by returning skill")
plt.xlabel("Normalized returning skill differential")
plt.ylabel("Fair catch probability")
plt.savefig('./figures/p_kickoff_fair_catch.png')
plt.clf()

# 6. Return yards
grouped_return_yard_averages = grouped_returning["return_yards"].agg(["mean", "std", "skew"])
print(grouped_return_yard_averages)
print()

returning_group_midpoints = [[(name.left + name.right)/2] for name in grouped_return_yard_averages.index]
mean_return_yards_model = LinearRegression()
mean_return_yards_model.fit(
    returning_group_midpoints,
    grouped_return_yard_averages["mean"]
)
mean_return_yards_pred = mean_return_yards_model.predict(zero_to_one)
print("Mean kickoff return yards model")
print(f"Coef: {mean_return_yards_model.coef_}")
print(f"Intr: {mean_return_yards_model.intercept_}")
print()

std_return_yards_model = LinearRegression()
std_return_yards_model.fit(
    returning_group_midpoints,
    grouped_return_yard_averages["std"]
)
std_return_yards_pred = std_return_yards_model.predict(zero_to_one)
print("Std kickoff return yards model")
print(f"Coef: {std_return_yards_model.coef_}")
print(f"Intr: {std_return_yards_model.intercept_}")
print()

skew_return_yards_model = LinearRegression()
skew_return_yards_model.fit(
    returning_group_midpoints,
    grouped_return_yard_averages["skew"]
)
skew_return_yards_pred = skew_return_yards_model.predict(zero_to_one)
print("Skew kickoff return yards model")
print(f"Coef: {skew_return_yards_model.coef_}")
print(f"Intr: {skew_return_yards_model.intercept_}")
print()

plt.scatter(returning_group_midpoints, grouped_return_yard_averages["mean"], color='g')
plt.scatter(returning_group_midpoints, grouped_return_yard_averages["std"], color='r')
plt.scatter(returning_group_midpoints, grouped_return_yard_averages["skew"], color='y')
plt.plot(zero_to_one, mean_return_yards_pred, color='b')
plt.plot(zero_to_one, std_return_yards_pred, color='b')
plt.plot(zero_to_one, skew_return_yards_pred, color='b')
plt.title("Kickoff return yard statistics by returning skill")
plt.xlabel("Normalized returning skill differential")
plt.ylabel("Return yards (mean, std, skew)")
plt.savefig('./figures/kickoff_return_yards.png')
plt.clf()

# 7. Fumble?
def fumble_count(s):
    return (s == 1).sum()
grouped_fumble_averages = grouped_returning["fumble"].agg(["count", fumble_count])
grouped_fumble_averages["p_fumble"] = grouped_fumble_averages["fumble_count"] / grouped_fumble_averages["count"]
print(grouped_fumble_averages)
print()
# NOTE: Constant 0.007 fumble probability

# 8. Duration
kickoff_return_duration_model = LinearRegression()
positive_returns = df[df["return_yards"] >= 0].dropna()
kickoff_return_duration_model.fit(
    np.sqrt(positive_returns[["return_yards"]]),
    positive_returns[["play_duration"]]
)
print("Kickoff return duration model (sqrt activation)")
print(f"Coef: {kickoff_return_duration_model.coef_}")
print(f"Intr: {kickoff_return_duration_model.intercept_}")
print()

zero_to_hundred = pd.DataFrame(np.linspace(0, 100))
kickoff_return_duration_pred = kickoff_return_duration_model.predict(pd.DataFrame(np.sqrt(zero_to_hundred)))
plt.scatter(df["return_yards"], df["play_duration"], color='g')
plt.plot(zero_to_hundred, kickoff_return_duration_pred, color='b')
plt.title("Kickoff return duration by return yards")
plt.xlabel("Return yards")
plt.ylabel("Kickoff return play duration")
plt.savefig('./figures/kickoff_return_duration.png')
plt.clf()
