import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("./data/passing.csv")

###
# Flow
# 1. Is QB pressured?
# 2. If pressured, is QB sacked?
# 3. If not sacked, does QB scramble?
# 4. If scramble, rush result
# 5. Short or deep pass?
# 6. Pass distance (Two separate models for short vs. deep)
# 7. Interception?
# 8. If interception, return yards
# 9. Complete?
# 10. Zero yards after catch?
# 11. If nonzero, yards after catch
# 12. Fumble?
# 13. If fumble, return yards
###

# 1. Is QB pressured?
df["pressured"] = (df["qb_hit"] == 1) | (df["tackled_for_loss"] == 1) | (df["sack"] == 1)
df["norm_diff_pass_blocking_rushing_group"] = pd.cut(df['norm_diff_pass_blocking_rushing'], bins=5)
grouped_blocking = df.groupby("norm_diff_pass_blocking_rushing_group")
def pressured_count(s):
    return (s == 1).sum()
grouped_blocking_averages = grouped_blocking["pressured"].agg(["count", pressured_count])
grouped_blocking_averages["p_pressure"] = grouped_blocking_averages["pressured_count"] / grouped_blocking_averages["count"]
print(grouped_blocking_averages)
print()

blocking_group_midpoints = [[(name.left + name.right)/2] for name in grouped_blocking_averages.index]
p_pressure_model = LinearRegression()
p_pressure_model.fit(
    blocking_group_midpoints,
    grouped_blocking_averages["p_pressure"]
)
print("Pressure probability model")
print(f"Coef: {p_pressure_model.coef_}")
print(f"Intr: {p_pressure_model.intercept_}")
print()

zero_to_one = pd.DataFrame(np.linspace(0, 1))
p_pressure_pred = p_pressure_model.predict(zero_to_one)
plt.scatter(blocking_group_midpoints, grouped_blocking_averages["p_pressure"], color='g')
plt.plot(zero_to_one, p_pressure_pred, color='b')
plt.title('QB pressure probability by blocking skill differential')
plt.xlabel('Normalized blocking / blitzing skill differential')
plt.ylabel('Probability of a QB pressure')
plt.savefig('./figures/qb_pressure_probability.png')
plt.clf()

# 2. If pressured, is QB sacked?
def sacked_count(s):
    return (s == 1).sum()
grouped_sack_averages = grouped_blocking["sack"].agg(["count", sacked_count])
grouped_sack_averages["p_sack"] = grouped_sack_averages["sacked_count"] / grouped_sack_averages["count"]
print(grouped_sack_averages)
print()

p_sack_model = LinearRegression()
p_sack_model.fit(
    blocking_group_midpoints,
    grouped_sack_averages["p_sack"]
)
print("Sack probability model")
print(f"Coef: {p_sack_model.coef_}")
print(f"Intr: {p_sack_model.intercept_}")
print()

p_sack_pred = p_sack_model.predict(zero_to_one)
plt.scatter(blocking_group_midpoints, grouped_sack_averages["p_sack"], color='g')
plt.plot(zero_to_one, p_sack_pred, color='b')
plt.title('QB sack probability by blocking skill differential')
plt.xlabel('Normalized blocking / blitzing skill differential')
plt.ylabel('Probability of a QB sack when under pressure')
plt.savefig('./figures/qb_sack_probability.png')
plt.clf()

# 3. If not sacked, does QB scramble?
df["norm_scrambling_group"] = pd.cut(df['norm_scrambling'], bins=5)
grouped_scrambling = df.groupby("norm_scrambling_group")
def scramble_count(s):
    return (s == 1).sum()
grouped_scrambling_averages = grouped_scrambling["qb_scramble"].agg(["count", scramble_count])
grouped_scrambling_averages["p_scramble"] = grouped_scrambling_averages["scramble_count"] / grouped_scrambling_averages["count"]
print(grouped_scrambling_averages)
print()

scrambling_group_midpoints = [[(name.left + name.right)/2] for name in grouped_scrambling_averages.index]
p_scramble_model = LinearRegression()
p_scramble_model.fit(
    scrambling_group_midpoints,
    grouped_scrambling_averages["p_scramble"]
)
print("Scramble probability model")
print(f"Coef: {p_scramble_model.coef_}")
print(f"Intr: {p_scramble_model.intercept_}")
print()

p_scramble_pred = p_scramble_model.predict(zero_to_one)
plt.scatter(scrambling_group_midpoints, grouped_scrambling_averages["p_scramble"], color='g')
plt.plot(zero_to_one, p_scramble_pred, color='b')
plt.title('QB scramble probability by scrambling skill')
plt.xlabel('Normalized scrambling skill')
plt.ylabel('Probability of a QB scramble when under pressure')
plt.savefig('./figures/qb_scramble_probability.png')
plt.clf()

# 4. If scramble, rush result
# Using rush result model for now
# TODO: Train scrambling model

# 5. Short or deep pass?
# TODO: Model frequency of short vs. deep pass based on yardline
df["yardline_group"] = pd.cut(df['yardline_100'], bins=10)
df["is_short"] = 0
df.loc[df["pass_length"] == "short", "is_short"] = 1
def short_pass_count(s):
    return (s == 1).sum()
grouped_yardline = df.groupby("yardline_group")
grouped_yardline_pass_length_averages = grouped_yardline["is_short"].agg(["count", short_pass_count])
grouped_yardline_pass_length_averages["p_short_pass"] = grouped_yardline_pass_length_averages["short_pass_count"] / \
    grouped_yardline_pass_length_averages["count"]
print(grouped_yardline_pass_length_averages)
print()

yardline_group_midpoints = [[(name.left + name.right) / 2] for name in grouped_yardline_pass_length_averages.index]
p_short_pass_model = LinearRegression()
pf = PolynomialFeatures(degree=2)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    grouped_yardline_pass_length_averages["p_short_pass"]
)
p_short_pass_model.fit(
    transformed_yardline_midpoints,
    grouped_yardline_pass_length_averages["p_short_pass"]
)
zero_to_hundred = pd.DataFrame(np.linspace(0, 100))
p_short_pass_pred = p_short_pass_model.predict(pf.fit_transform(zero_to_hundred))
print("Short pass probability model")
print(f"Coef: {p_short_pass_model.coef_}")
print(f"Intr: {p_short_pass_model.intercept_}")
print()
plt.scatter(yardline_group_midpoints, grouped_yardline_pass_length_averages["p_short_pass"], color='g')
plt.plot(zero_to_hundred, p_short_pass_pred, color='b')
plt.title('Short pass probability by yard line')
plt.xlabel('Yard line')
plt.ylabel('Probability of a short pass')
plt.savefig('./figures/short_pass_probability.png')
plt.clf()

# 6. Pass distance (Two separate models for short vs. deep)
short_passes = df[df["pass_length"] == "short"]
deep_passes = df[df["pass_length"] == "deep"]

grouped_short_passes = short_passes.groupby("yardline_group")
grouped_short_pass_averages = grouped_short_passes["air_yards"].agg(["mean", "std", "skew"])
print(grouped_short_pass_averages)
print()
mean_short_pass_dist_model = LinearRegression()
pf = PolynomialFeatures(degree=3)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    grouped_short_pass_averages["mean"]
)
mean_short_pass_dist_model.fit(
    transformed_yardline_midpoints,
    grouped_short_pass_averages["mean"]
)
mean_short_pass_dist_pred = mean_short_pass_dist_model.predict(pf.fit_transform(zero_to_hundred))
print("Mean short pass distance model")
print(f"Coef: {mean_short_pass_dist_model.coef_}")
print(f"Intr: {mean_short_pass_dist_model.intercept_}")
print()

std_short_pass_dist_model = LinearRegression()
pf = PolynomialFeatures(degree=3)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    grouped_short_pass_averages["std"]
)
std_short_pass_dist_model.fit(
    transformed_yardline_midpoints,
    grouped_short_pass_averages["std"]
)
std_short_pass_dist_pred = std_short_pass_dist_model.predict(pf.fit_transform(zero_to_hundred))
print("Std short pass distance model")
print(f"Coef: {std_short_pass_dist_model.coef_}")
print(f"Intr: {std_short_pass_dist_model.intercept_}")
print()

plt.scatter(yardline_group_midpoints, grouped_short_pass_averages["mean"], color='g')
plt.scatter(yardline_group_midpoints, grouped_short_pass_averages["std"], color='r')
plt.plot(zero_to_hundred, mean_short_pass_dist_pred, color='b')
plt.plot(zero_to_hundred, std_short_pass_dist_pred, color='b')
plt.title('Mean, std short pass distance by field position')
plt.xlabel('Field position')
plt.ylabel('Mean, std, skew short pass distance')
plt.savefig('./figures/mean_std_short_pass_distance.png')
plt.clf()

grouped_deep_passes = deep_passes.groupby("yardline_group")
grouped_deep_pass_averages = grouped_deep_passes["air_yards"].agg(["mean", "std", "skew"])
print(grouped_deep_pass_averages)
print()
mean_deep_pass_dist_model = LinearRegression()
pf = PolynomialFeatures(degree=3)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    grouped_deep_pass_averages["mean"]
)
mean_deep_pass_dist_model.fit(
    transformed_yardline_midpoints,
    grouped_deep_pass_averages["mean"]
)
mean_deep_pass_dist_pred = mean_deep_pass_dist_model.predict(pf.fit_transform(zero_to_hundred))
print("Mean deep pass distance model")
print(f"Coef: {mean_deep_pass_dist_model.coef_}")
print(f"Intr: {mean_deep_pass_dist_model.intercept_}")
print()

std_deep_pass_dist_model = LinearRegression()
pf = PolynomialFeatures(degree=3)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    grouped_deep_pass_averages["std"]
)
std_deep_pass_dist_model.fit(
    transformed_yardline_midpoints,
    grouped_deep_pass_averages["std"]
)
std_deep_pass_dist_pred = std_deep_pass_dist_model.predict(pf.fit_transform(zero_to_hundred))
print("Std deep pass distance model")
print(f"Coef: {std_deep_pass_dist_model.coef_}")
print(f"Intr: {std_deep_pass_dist_model.intercept_}")
print()

plt.scatter(yardline_group_midpoints, grouped_deep_pass_averages["mean"], color='g')
plt.scatter(yardline_group_midpoints, grouped_deep_pass_averages["std"], color='r')
plt.plot(zero_to_hundred, mean_deep_pass_dist_pred, color='b')
plt.plot(zero_to_hundred, std_deep_pass_dist_pred, color='b')
plt.title('Mean, std, skew deep pass distance by field position')
plt.xlabel('Field position')
plt.ylabel('Mean, std, skew deep pass distance')
plt.savefig('./figures/mean_std_deep_pass_distance.png')
plt.clf()

# 7. Interception?
df["norm_diff_interceptions_group"] = pd.cut(df["norm_diff_interceptions"], bins=10)
def interception_count(s):
    return (s == 1).sum()
grouped_interceptions = df.groupby("norm_diff_interceptions_group")
grouped_interceptions_averages = grouped_interceptions["interception"].agg(["count", interception_count])
grouped_interceptions_averages["p_interception"] = grouped_interceptions_averages["interception_count"] / grouped_interceptions_averages["count"]
print(grouped_interceptions_averages)
print()

grouped_interceptions_midpoints = [[(name.left + name.right) / 2] for name in grouped_interceptions_averages.index]
p_interception_model = LinearRegression()
p_interception_model.fit(
    grouped_interceptions_midpoints,
    grouped_interceptions_averages["p_interception"]
)
p_interception_pred = p_interception_model.predict(zero_to_one)
print("Interception probability model")
print(f"Coef: {p_interception_model.coef_}")
print(f"Intr: {p_interception_model.intercept_}")
print()

plt.scatter(grouped_interceptions_midpoints, grouped_interceptions_averages["p_interception"], color='g')
plt.plot(zero_to_one, p_interception_pred, color='b')
plt.title('Interception probability by turnover skill differential')
plt.xlabel('Normalized turnover skill differential')
plt.ylabel('Interception probability')
plt.savefig('./figures/interception_probability.png')
plt.clf()

# 8. If interception, return yards
ints = df[df["interception"] == 1]
grouped_int_return_yardline = ints.groupby("yardline_group")
grouped_int_return_averages = grouped_int_return_yardline["return_yards"].agg(["mean", "std", "skew"])
print(grouped_int_return_averages)
print()

mean_int_return_model = LinearRegression()
pf = PolynomialFeatures(degree=3)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    grouped_int_return_averages["mean"]
)
mean_int_return_model.fit(
    transformed_yardline_midpoints,
    grouped_int_return_averages["mean"]
)
mean_int_return_pred = mean_int_return_model.predict(pf.fit_transform(zero_to_hundred))
print("Mean interception return yards model")
print(f"Coef: {mean_int_return_model.coef_}")
print(f"Intr: {mean_int_return_model.intercept_}")
print()

std_int_return_model = LinearRegression()
pf = PolynomialFeatures(degree=3)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    grouped_int_return_averages["std"]
)
std_int_return_model.fit(
    transformed_yardline_midpoints,
    grouped_int_return_averages["std"]
)
std_int_return_pred = std_int_return_model.predict(pf.fit_transform(zero_to_hundred))
print("Std interception return yards model")
print(f"Coef: {std_int_return_model.coef_}")
print(f"Intr: {std_int_return_model.intercept_}")
print()

skew_int_return_model = LinearRegression()
pf = PolynomialFeatures(degree=3)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    grouped_int_return_averages["skew"]
)
skew_int_return_model.fit(
    transformed_yardline_midpoints,
    grouped_int_return_averages["skew"]
)
skew_int_return_pred = skew_int_return_model.predict(pf.fit_transform(zero_to_hundred))
print("Skew interception return yards model")
print(f"Coef: {skew_int_return_model.coef_}")
print(f"Intr: {skew_int_return_model.intercept_}")
print()

plt.scatter(yardline_group_midpoints, grouped_int_return_averages["mean"], color='g')
plt.scatter(yardline_group_midpoints, grouped_int_return_averages["std"], color='r')
plt.scatter(yardline_group_midpoints, grouped_int_return_averages["skew"], color='y')
plt.plot(zero_to_hundred, mean_int_return_pred, color='b')
plt.plot(zero_to_hundred, std_int_return_pred, color='b')
plt.plot(zero_to_hundred, skew_int_return_pred, color='b')
plt.title('Mean, std, skew interception return yards by field position')
plt.xlabel('Field position')
plt.ylabel('Mean, std, skew interception return yards')
plt.savefig('./figures/mean_std_skew_int_return_yards.png')
plt.clf()

# 9. Complete?
df["norm_diff_passing_group"] = pd.cut(df["norm_diff_passing"], bins=5)
grouped_passing = df[df["pass_attempt"] == 1].groupby("norm_diff_passing_group")
def complete_count(s):
    return (s == 0).sum()
grouped_passing_averages = grouped_passing["incomplete_pass"].agg(["count", complete_count])
grouped_passing_averages["p_complete"] = grouped_passing_averages["complete_count"] / grouped_passing_averages["count"]
print(grouped_passing_averages)
print()

passing_group_midpoints = [[(name.left + name.right) / 2] for name in grouped_passing_averages.index]
complete_pass_model = LinearRegression()
complete_pass_model.fit(
    passing_group_midpoints,
    grouped_passing_averages["p_complete"]
)
complete_pass_pred = complete_pass_model.predict(zero_to_one)
print("Complete pass probability model")
print(f"Coef: {complete_pass_model.coef_}")
print(f"Intr: {complete_pass_model.intercept_}")
print()

plt.scatter(passing_group_midpoints, grouped_passing_averages["p_complete"], color='g')
plt.plot(zero_to_one, complete_pass_pred, color='b')
plt.title("Complete pass probability by passing skill differential")
plt.xlabel("Normalized passing skill differential")
plt.ylabel("Probability of a completed pass")
plt.savefig('./figures/completed_pass_probability.png')
plt.clf()

# 10. Zero yards after catch?
df["norm_diff_receiving_group"] = pd.cut(df["norm_diff_receiving"], bins=5)
df["zero_yac"] = 0
df.loc[df["yards_after_catch"] == 0, "zero_yac"] = 1
grouped_receiving = df[df["pass_attempt"] == 1].groupby("norm_diff_receiving_group")
def zero_yac_count(s):
    return (s == 1).sum()
grouped_receiving_zero_yac_averages = grouped_receiving["zero_yac"].agg(["count", zero_yac_count])
grouped_receiving_zero_yac_averages["p_zero_yac"] = grouped_receiving_zero_yac_averages["zero_yac_count"] / grouped_receiving_zero_yac_averages["count"]
print(grouped_receiving_zero_yac_averages)
print()

receiving_group_midpoints = [[(name.left + name.right) / 2] for name in grouped_receiving_zero_yac_averages.index]
p_zero_yac_model = LinearRegression()
p_zero_yac_model.fit(
    receiving_group_midpoints,
    grouped_receiving_zero_yac_averages["p_zero_yac"]
)
p_zero_yac_pred = p_zero_yac_model.predict(zero_to_one)
print("Zero yards after catch probability model")
print(f"Coef: {p_zero_yac_model.coef_}")
print(f"Intr: {p_zero_yac_model.intercept_}")
print()

plt.scatter(receiving_group_midpoints, grouped_receiving_zero_yac_averages["p_zero_yac"], color='g')
plt.plot(zero_to_one, p_zero_yac_pred, color='b')
plt.title("Probability of zero yards after catch by receiving skill differential")
plt.xlabel("Normalized receiving skill differential")
plt.ylabel("Probability of zero yards after catch")
plt.savefig('./figures/zero_yac_probability.png')
plt.clf()

# 11. If nonzero, yards after catch
grouped_receiving_averages = grouped_receiving["yards_after_catch"].agg(["mean", "std", "skew"])
print(grouped_receiving_averages)
print()

receiving_group_midpoints = [[(name.left + name.right) / 2] for name in grouped_receiving_averages.index]
mean_yac_model = LinearRegression()
pf = PolynomialFeatures(degree=2)
transformed_receiving_midpoints = pf.fit_transform(receiving_group_midpoints)
pf.fit(
    transformed_receiving_midpoints,
    grouped_receiving_averages["mean"]
)
mean_yac_model.fit(
    transformed_receiving_midpoints,
    grouped_receiving_averages["mean"]
)
mean_yac_pred = mean_yac_model.predict(pf.fit_transform(zero_to_one))
print("Mean yards after catch model")
print(f"Coef: {mean_yac_model.coef_}")
print(f"Intr: {mean_yac_model.intercept_}")
print()

std_yac_model = LinearRegression()
pf = PolynomialFeatures(degree=2)
transformed_receiving_midpoints = pf.fit_transform(receiving_group_midpoints)
pf.fit(
    transformed_receiving_midpoints,
    grouped_receiving_averages["std"]
)
std_yac_model.fit(
    transformed_receiving_midpoints,
    grouped_receiving_averages["std"]
)
std_yac_pred = std_yac_model.predict(pf.fit_transform(zero_to_one))
print("Std yards after catch model")
print(f"Coef: {std_yac_model.coef_}")
print(f"Intr: {std_yac_model.intercept_}")
print()

skew_yac_model = LinearRegression()
skew_yac_model.fit(
    receiving_group_midpoints,
    grouped_receiving_averages["skew"]
)
skew_yac_pred = skew_yac_model.predict(zero_to_one)
print("Skew yards after catch model")
print(f"Coef: {skew_yac_model.coef_}")
print(f"Intr: {skew_yac_model.intercept_}")
print()

plt.scatter(receiving_group_midpoints, grouped_receiving_averages["mean"], color='g')
plt.scatter(receiving_group_midpoints, grouped_receiving_averages["std"], color='r')
plt.scatter(receiving_group_midpoints, grouped_receiving_averages["skew"], color='y')
plt.plot(zero_to_one, mean_yac_pred, color='b')
plt.plot(zero_to_one, std_yac_pred, color='b')
plt.plot(zero_to_one, skew_yac_pred, color='b')
plt.title("Mean, std, skew yards after catch by receiving skill differential")
plt.xlabel("Normalized receiving skill differential")
plt.ylabel("Mean, std, skew yards after catch")
plt.savefig('./figures/mean_std_skew_yac.png')
plt.clf()

# 12. Fumble?
def fumble_count(s):
    return (s == 1).sum()
total_completions = len(df[(df["pass_attempt"] == 1) & (df["incomplete_pass"] == 0)])
total_fumbles = len(df[(df["pass_attempt"] == 1) & (df["incomplete_pass"] == 0) & (df["fumble"] == 1)])
p_fumble = total_fumbles / total_completions
print(f"Fumble probability: {p_fumble}")
print()

# 13. If fumble, return yards
# TODO: Fumble return yards
# Fumble recovery distribution will be an exponential distrubution with lambda = 1

# TODO: Play duration

# Incomplete pass duration
incomplete_passes = df[df["incomplete_pass"] == 1]
exp_incomplete_duration = incomplete_passes[incomplete_passes['play_duration'] < 8]
incomplete_pass_duration_model = LinearRegression()
incomplete_pass_duration_model.fit(
    pd.DataFrame(exp_incomplete_duration['yards_gained']),
    pd.DataFrame(exp_incomplete_duration['play_duration'])
)
print("Incomplete pass duration model")
print(f"coef: {incomplete_pass_duration_model.coef_}")
print(f"intr: {incomplete_pass_duration_model.intercept_}")
print()

zero_to_hundred = np.linspace(-20, 100)
incomplete_dur_pred = incomplete_pass_duration_model.predict(pd.DataFrame(zero_to_hundred))
plt.scatter(exp_incomplete_duration['air_yards'], exp_incomplete_duration['play_duration'], color='g')
plt.scatter(zero_to_hundred, np.random.normal(loc=incomplete_dur_pred, scale=2), color='b')
plt.title("Incomplete pass play duration by pass distance")
plt.xlabel("Pass distance (yards)")
plt.ylabel("Play duration (seconds)")
plt.savefig('./figures/incomplete_pass_duration.png')
plt.clf()

# Complete pass duration
complete_passes = df[df["incomplete_pass"] == 0]
exp_complete_duration = complete_passes[complete_passes['play_duration'] < 14]
complete_pass_duration_model = LinearRegression()
complete_pass_duration_model.fit(
    pd.DataFrame(exp_complete_duration['yards_gained']),
    pd.DataFrame(exp_complete_duration['play_duration'])
)
print("Complete pass duration model")
print(f"coef: {complete_pass_duration_model.coef_}")
print(f"intr: {complete_pass_duration_model.intercept_}")
print()

zero_to_hundred = np.linspace(-20, 100)
complete_dur_pred = incomplete_pass_duration_model.predict(pd.DataFrame(zero_to_hundred))
plt.scatter(exp_complete_duration['yards_gained'], exp_complete_duration['play_duration'], color='g')
plt.scatter(zero_to_hundred, np.random.normal(loc=complete_dur_pred, scale=2), color='b')
plt.title("Complete pass play duration by yards gained")
plt.xlabel("Yards gained")
plt.ylabel("Play duration (seconds)")
plt.savefig('./figures/complete_pass_duration.png')
plt.clf()
