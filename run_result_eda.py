import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load rushing data
df = pd.read_csv("./data/rushing.csv")

# Sort into 10 subsets based on normalized diff column values
df['norm_diff_rushing_group'] = pd.cut(df['norm_diff_rushing'], bins=5)
df['norm_diff_handling_group'] = pd.cut(df['norm_diff_ball_handling'], bins=5)
big_plays_only = df[df['yards_gained'] > 20]
big_play_non_tds_only = big_plays_only[big_plays_only['touchdown'] == 0]
standard_plays_only = df[df['yards_gained'] < 20]
grouped_rushing = df.groupby('norm_diff_rushing_group')
grouped_handling = df.groupby('norm_diff_handling_group')
grouped_rushing_standard_plays = standard_plays_only.groupby('norm_diff_rushing_group')
grouped_rushing_big_plays = big_plays_only.groupby('norm_diff_rushing_group')
grouped_rushing_non_td_big_plays = big_play_non_tds_only.groupby('norm_diff_rushing_group')

# Define aggregation function for big plays
def big_plays(s):
    return (s > 20).sum()

def touchdowns(s):
    return (s == 1).sum()

# Compute the average and standard deviation rush yards per rushing group
grouped_rushing_averages = grouped_rushing['yards_gained'].agg(['mean', 'std', 'count', big_plays])
grouped_rushing_averages['big_play_proportion'] = grouped_rushing_averages['big_plays'] / grouped_rushing_averages['count']
print(grouped_rushing_averages)
print()

# Compute the average and std rush yards per group for standard plays
grouped_standard_rushing_averages = grouped_rushing_standard_plays['yards_gained'].agg(['mean', 'std'])

# Compute the proportion of touchdowns on big plays
grouped_rushing_big_play_tds = grouped_rushing_big_plays['touchdown'].agg(['count', touchdowns])
grouped_rushing_big_play_tds['touchdown_proportion'] = grouped_rushing_big_play_tds['touchdowns'] / grouped_rushing_big_play_tds['count']
grouped_rushing_big_play_yards = grouped_rushing_non_td_big_plays['yards_gained'].agg(['mean', 'std'])

# Define aggregation function for fumbles
def fumbles(s):
    return (s == 1).sum()

# Compute the proportion of fumbles per handling groups
grouped_handling_averages = grouped_handling['fumble'].agg(['count', fumbles])
grouped_handling_averages['fumble_proportion'] = grouped_handling_averages['fumbles'] / grouped_handling_averages['count']
print(grouped_handling_averages)
print()

# Train a model mapping norm diff rushing to mean & std dev yards gained
rushing_group_midpoints = [[(name.left + name.right)/2] for name in grouped_rushing_averages.index]
mean_model = LinearRegression()
mean_model.fit(
    rushing_group_midpoints,
    grouped_standard_rushing_averages['mean']
)
print("Mean rushing model")
print(f"y = {mean_model.coef_}x + {mean_model.intercept_}")
print()
pf = PolynomialFeatures(degree=2)
transformed_rushing_midpoints = pf.fit_transform(rushing_group_midpoints)
pf.fit(
    transformed_rushing_midpoints,
    grouped_standard_rushing_averages["std"]
)
std_model = LinearRegression()
std_model.fit(
    transformed_rushing_midpoints,
    grouped_standard_rushing_averages["std"]
)
print("Std dev rushing model")
print(f"coef: {std_model.coef_}")
print(f"intr: {std_model.intercept_}")
print()

# Plot the mean & std dev models
zero_to_one = pd.DataFrame(np.linspace(0, 1))
mean_pred = mean_model.predict(zero_to_one)
std_pred = std_model.predict(pf.fit_transform(zero_to_one))
plt.scatter(rushing_group_midpoints, grouped_standard_rushing_averages["mean"], color='g')
plt.scatter(rushing_group_midpoints, grouped_standard_rushing_averages["std"], color='r')
plt.plot(zero_to_one, mean_pred, color='b')
plt.plot(zero_to_one, std_pred, color='b')
plt.title('Mean & std ypc by rushing skill differential')
plt.xlabel('Normalized rushing skill differential')
plt.ylabel('Mean & std yards per carry')
plt.savefig('./figures/mean_std_rushing.png')
plt.clf()

# Train a model mapping norm diff rushing to big play proportion, plot
transformed_averages = np.log(grouped_rushing_averages["big_play_proportion"])
p_big_play_model = LinearRegression()
p_big_play_model.fit(
    rushing_group_midpoints,
    transformed_averages
)
print("Big play probability model")
print(f"coef: {p_big_play_model.coef_}")
print(f"intr: {p_big_play_model.intercept_}")
print()

bp_prop_pred = np.exp(p_big_play_model.predict(zero_to_one))
plt.scatter(rushing_group_midpoints, grouped_rushing_averages["big_play_proportion"], color='g')
plt.plot(zero_to_one, bp_prop_pred, color='b')
plt.title('Big play probability by rushing skill differential')
plt.xlabel('Normalized rushing skill differential')
plt.ylabel('Big play probability')
plt.savefig('./figures/big_rushing_play.png')
plt.clf()

# Train a model mapping norm diff rushing to big play touchdown proportion
transformed_td_props = np.log(grouped_rushing_big_play_tds["touchdown_proportion"])
p_touchdown_model = LinearRegression()
p_touchdown_model.fit(
    rushing_group_midpoints,
    transformed_td_props
)
print("Big play touchdown probability model")
print(f"coef: {p_touchdown_model.coef_}")
print(f"intr: {p_touchdown_model.intercept_}")
print()

td_prop_pred = np.exp(p_touchdown_model.predict(zero_to_one))
plt.scatter(rushing_group_midpoints, grouped_rushing_big_play_tds["touchdown_proportion"], color='g')
plt.plot(zero_to_one, td_prop_pred, color='b')
plt.title('Big play TD probability by rushing skill differential')
plt.xlabel('Normalized rushing skill differential')
plt.ylabel('Big play touchdown probability')
plt.savefig('./figures/big_rushing_td.png')
plt.clf()

# Train a model mapping norm diff rushing to big play non-TD yards
bp_yards_mean_model = LinearRegression()
bp_yards_mean_model.fit(
    rushing_group_midpoints,
    grouped_rushing_big_play_yards["mean"]
)
print("Big play non-touchdown yards mean model")
print(f"y = {bp_yards_mean_model.coef_}x + {bp_yards_mean_model.intercept_}")
print()
pf = PolynomialFeatures(degree=2)
transformed_rushing_midpoints = pf.fit_transform(rushing_group_midpoints)
bp_yards_std_model = LinearRegression()
bp_yards_std_model.fit(
    transformed_rushing_midpoints,
    grouped_rushing_big_play_yards["std"]
)
print("Big play non-touchdown yards std model")
print(f"coef: {bp_yards_std_model.coef_}")
print(f"intr: {bp_yards_std_model.intercept_}")
print()

bp_yards_mean_pred = bp_yards_mean_model.predict(zero_to_one)
bp_yards_std_pred = bp_yards_std_model.predict(pf.fit_transform(zero_to_one))
plt.scatter(rushing_group_midpoints, grouped_rushing_big_play_yards["std"], color='g')
plt.scatter(rushing_group_midpoints, grouped_rushing_big_play_yards["mean"], color='r')
plt.plot(zero_to_one, bp_yards_mean_pred, color='b')
plt.plot(zero_to_one, bp_yards_std_pred, color='b')
plt.title('Mean & std big play non-TD yards by rushing skill differential')
plt.xlabel('Normalized rushing skill differential')
plt.ylabel('Mean & std big play non-TD yards')
plt.savefig('./figures/mean_std_bp_rushing.png')
plt.clf()

# Train a model mapping norm diff ball handling to fumble proportion
handling_group_midpoints = [[(name.left + name.right)/2] for name in grouped_handling_averages.index]
p_fumble_model = LinearRegression()
p_fumble_model.fit(
    handling_group_midpoints,
    grouped_handling_averages['fumble_proportion']
)
print("Fumble probability model")
print(f"coef: {p_fumble_model.coef_}")
print(f"intr: {p_fumble_model.intercept_}")
print()

f_prop_pred = p_fumble_model.predict(zero_to_one)
plt.scatter(rushing_group_midpoints, grouped_handling_averages["fumble_proportion"], color='g')
plt.plot(zero_to_one, f_prop_pred, color='b')
plt.title('Fumble proportion by ball security skill differential')
plt.xlabel('Normalized ball security skill differential')
plt.ylabel('Fumble probability')
plt.savefig('./figures/rushing_fumble.png')
plt.clf()

# Fumble recovery distribution will be an exponential distrubution with lambda = 1
exp_play_duration = df[df['play_duration'] < 12]
play_duration_model = LinearRegression()
pf = PolynomialFeatures(degree=2)
play_duration_model.fit(
    pf.fit_transform(pd.DataFrame(exp_play_duration['yards_gained'])),
    pd.DataFrame(exp_play_duration['play_duration'])
)
print("Play duration model")
print(f"coef: {play_duration_model.coef_}")
print(f"intr: {play_duration_model.intercept_}")
print()

zero_to_hundred = np.linspace(-20, 100)
dur_pred = np.abs(play_duration_model.predict(pf.fit_transform(pd.DataFrame(zero_to_hundred))))
plt.scatter(exp_play_duration['yards_gained'], exp_play_duration['play_duration'], color='g')
plt.scatter(zero_to_hundred, np.random.normal(loc=dur_pred, scale=2), color='b')
plt.title('Play duration by yards gained')
plt.xlabel('Yards gained')
plt.ylabel('Play duration')
plt.savefig('./figures/rushing_play_duration.png')
plt.clf()

# Flow:
# 1. Is this a big play? If so, is this a touchdown? If not, yards gained?
# 2. Sample normal curve to generate yards gained based on mean + std dev
# 3. Is this a fumble? If so, return yards?
