import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('./data/punts.csv')

###
# Punt model flow
# 1. Blocked?
# 2. If blocked, Return yards
# 3. Touchback, inside 20, outside 20?
# 4. Punt distance given punt chunk
# 5. Out of bounds?
# 6. Fair catch? Muffed?
# 7. Return yards
# 8. Fumble?
# 9. Duration
###

# 1. Blocked?
df['norm_blitzing_group'] = pd.cut(df['norm_blitzing'], bins=4)
grouped_blitzing = df.groupby('norm_blitzing_group')
def blocked_count(s):
    return (s == 1).sum()
grouped_blitzing_averages = grouped_blitzing["punt_blocked"].agg(["count", blocked_count])
grouped_blitzing_averages["p_block"] = grouped_blitzing_averages["blocked_count"] / grouped_blitzing_averages["count"]
print(grouped_blitzing_averages)
print()

blocking_group_midpoints = [[(name.left + name.right)/2] for name in grouped_blitzing_averages.index]
p_block_model = LinearRegression()
p_block_model.fit(
    blocking_group_midpoints,
    grouped_blitzing_averages["p_block"]
)
print("Punt block probability model")
print(f"Coef: {p_block_model.coef_}")
print(f"Intr: {p_block_model.intercept_}")
print()

zero_to_one = pd.DataFrame(np.linspace(0, 1))
p_block_pred = p_block_model.predict(zero_to_one)
plt.scatter(blocking_group_midpoints, grouped_blitzing_averages["p_block"], color='g')
plt.plot(zero_to_one, p_block_pred, color='b')
plt.title('Punt block probability by blitzing skill')
plt.xlabel('Blitzing skill')
plt.ylabel('Punt block probability')
plt.savefig('./figures/punt_block_probability.png')
plt.clf()

# 2. If blocked, Return yards
# TODO: punt block return yards

# 3. Punt in endzone, inside 20, short of 20?
df['norm_punting_group'] = pd.cut(df['norm_punting'], bins=4)
grouped_punting = df.groupby('norm_punting_group')
def touchback_count(s):
    return (s == 1).sum()
def inside_20_count(s):
    return (s == 1).sum()
grouped_touchback_averages = grouped_punting["touchback"].agg(["count", touchback_count])
grouped_touchback_averages["p_touchback"] = grouped_touchback_averages['touchback_count'] / grouped_touchback_averages['count']
grouped_punting_averages = grouped_punting["punt_inside_twenty"].agg(["count", inside_20_count])
grouped_punting_averages["p_inside_20"] = grouped_punting_averages['inside_20_count'] / grouped_punting_averages['count']
grouped_punting_averages["touchback_count"] = grouped_touchback_averages["touchback_count"]
grouped_punting_averages["p_touchback"] = grouped_touchback_averages["p_touchback"]
grouped_punting_averages["p_outside_20"] = (
    (
        grouped_punting_averages['count'] - (
            grouped_punting_averages['inside_20_count'] + \
            grouped_punting_averages['touchback_count']
        )
    ) / grouped_punting_averages['count']
)
print(grouped_punting_averages)
print()

punting_group_midpoints = [[(name.left + name.right)/2] for name in grouped_punting_averages.index]
p_inside_twenty_model = LinearRegression()
p_inside_twenty_model.fit(
    punting_group_midpoints,
    grouped_punting_averages["p_inside_20"]
)
print("Punt inside 20 probability model (skill-based)")
print(f"Coef: {p_inside_twenty_model.coef_}")
print(f"Intr: {p_inside_twenty_model.intercept_}")
print()
def inside_twenty_skill(skill):
    return (p_inside_twenty_model.coef_[0] * skill) + p_inside_twenty_model.intercept_

p_inside_twenty_pred = p_inside_twenty_model.predict(zero_to_one)
plt.scatter(punting_group_midpoints, grouped_punting_averages["p_inside_20"], color='g')
plt.plot(zero_to_one, p_inside_twenty_pred, color='b')
plt.title('Punt inside 20 probability by punting skill')
plt.xlabel('Punting skill')
plt.ylabel('Punt inside 20 probability')
plt.savefig('./figures/punt_inside_20_skill_probability.png')
plt.clf()

df['yardline_group'] = pd.cut(df['yardline_100'], bins=10)
grouped_yardline = df.groupby("yardline_group")
grouped_yardline_averages = grouped_yardline["punt_inside_twenty"].agg(["count", inside_20_count])
grouped_yardline_averages["p_inside_20"] = grouped_yardline_averages["inside_20_count"] / grouped_yardline_averages["count"]
print(grouped_yardline_averages)
print()

yardline_group_midpoints = [[(name.left + name.right) / 2] for name in grouped_yardline_averages.index]
def logistic(x, L, k, x0, b):
    return L / (1 + np.exp(-k*(x - x0))) + b
yardline_group_midpoints_np = np.array(yardline_group_midpoints).ravel()
p_inside_twenty_np = grouped_yardline_averages['p_inside_20'].values.ravel()
params, covariance = curve_fit(
    logistic,
    yardline_group_midpoints_np,
    p_inside_twenty_np,
    p0=[
        max(p_inside_twenty_np), 1,
        np.median(yardline_group_midpoints_np),
        min(p_inside_twenty_np)
    ],
    bounds=(
        [0, -np.inf, -np.inf, -np.inf],  # lower bounds
        [np.inf,  np.inf,  np.inf,  np.inf]  # upper bounds
    )
)
L, k, x0, b = params
print("Fitted parameters:", params)
print()

zero_to_hundred = pd.DataFrame(np.linspace(0, 100))
p_inside_twenty_yardline_pred = logistic(zero_to_hundred, *params)
plt.scatter(yardline_group_midpoints, grouped_yardline_averages["p_inside_20"], color='g')
plt.plot(zero_to_hundred, p_inside_twenty_yardline_pred, color='b')
plt.title('Punt inside 20 probability by yard line')
plt.xlabel('Yard line')
plt.ylabel('Punt inside 20 probability')
plt.savefig('./figures/punt_inside_20_yardline_probability.png')
plt.clf()

# Stack the two models
x_1d = np.linspace(0, 100)
y_1d = np.linspace(0, 1)
X, Y = np.meshgrid(x_1d, y_1d)
Z = ((inside_twenty_skill(Y) * 0.4) + (logistic(X, *params) * 0.6)) * 1.18
fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z, 100)
plt.colorbar(cf, ax=ax, label='Probability of a punt inside 20')
plt.title('Probability of a punt landing inside the 20 yard line')
plt.xlabel("Yard line")
plt.ylabel("Punting skill")
plt.savefig('./figures/punt_inside_20_probability.png')
plt.clf()

# NOTE: Constant 0.06 probability of a touchback regardless of punting ability

# 4. Punt distance given punt chunk
# TODO: Punt distance inside 20
df["punt_landing"] = df["yardline_100"] - df["kick_distance"]
df["relative_punt_landing"] = df["punt_landing"] / df["yardline_100"]
df = df.query("relative_punt_landing > 0")
punts_inside_20 = df[df["punt_inside_twenty"] == 1]
grouped_punts_inside_20 = punts_inside_20.groupby("yardline_group")
punts_inside_20_distance_averages = grouped_punts_inside_20["relative_punt_landing"].agg(["mean", "std", "skew"])
punts_inside_20_distance_averages["std"] = punts_inside_20_distance_averages["std"].fillna(0)
punts_inside_20_distance_averages["skew"] = punts_inside_20_distance_averages["skew"].fillna(0)
print(punts_inside_20_distance_averages)
print()

punt_inside_20_mean_model = LinearRegression()
punt_inside_20_mean_model.fit(
    yardline_group_midpoints,
    punts_inside_20_distance_averages["mean"]
)
print("Punt inside 20 mean relative distance model")
print(f"Coef: {punt_inside_20_mean_model.coef_}")
print(f"Intr: {punt_inside_20_mean_model.intercept_}")
print()

punt_inside_20_std_model = LinearRegression()
punt_inside_20_std_model.fit(
    yardline_group_midpoints,
    punts_inside_20_distance_averages["std"]
)
print("Punt inside 20 std relative distance model")
print(f"Coef: {punt_inside_20_std_model.coef_}")
print(f"Intr: {punt_inside_20_std_model.intercept_}")
print()

punt_inside_20_skew_model = LinearRegression()
pf = PolynomialFeatures(degree=2)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    punts_inside_20_distance_averages["skew"]
)
punt_inside_20_skew_model.fit(
    transformed_yardline_midpoints,
    punts_inside_20_distance_averages["skew"]
)
print("Punt inside 20 skew relative distance model")
print(f"Coef: {punt_inside_20_skew_model.coef_}")
print(f"Intr: {punt_inside_20_skew_model.intercept_}")
print()

punts_inside_20_mean_pred = punt_inside_20_mean_model.predict(zero_to_hundred)
punts_inside_20_std_pred = punt_inside_20_std_model.predict(zero_to_hundred)
punts_inside_20_skew_pred = punt_inside_20_skew_model.predict(pf.fit_transform(zero_to_hundred))
plt.scatter(yardline_group_midpoints, punts_inside_20_distance_averages["mean"], color='g')
plt.scatter(yardline_group_midpoints, punts_inside_20_distance_averages["std"], color='r')
plt.scatter(yardline_group_midpoints, punts_inside_20_distance_averages["skew"], color='y')
plt.plot(zero_to_hundred, punts_inside_20_mean_pred, color='b')
plt.plot(zero_to_hundred, punts_inside_20_std_pred, color='b')
plt.plot(zero_to_hundred, punts_inside_20_skew_pred, color='b')
plt.title("Relative punt landing for punts inside 20")
plt.xlabel("Current line")
plt.ylabel("Relative punt landing")
plt.ylim(-1, 1)
plt.savefig('./figures/punt_distance_inside_20.png')
plt.clf()

# Plot heatmap of PDF of normal distributions as yard line changes
x_1d = np.linspace(-0.1, 0.5)
y_1d = np.linspace(0, 100)
X, Y = np.meshgrid(x_1d, y_1d)
Z = skewnorm.pdf(
    X,
    a=(
        punt_inside_20_skew_model.intercept_ + \
        (punt_inside_20_skew_model.coef_[1] * Y) + \
        (punt_inside_20_skew_model.coef_[2] * pow(Y, 2))
    ),
    loc=(
        punt_inside_20_mean_model.intercept_ + \
        punt_inside_20_mean_model.coef_[0] * Y
    ),
    scale=(
        punt_inside_20_std_model.intercept_ + \
        punt_inside_20_std_model.coef_[0] * Y
    )
)
fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z, 100)
plt.colorbar(cf, ax=ax, label='Probability density')
plt.title('Probability density of punt inside the 20 by yard line')
plt.xlabel("Relative landing")
plt.ylabel("Yard line")
plt.savefig('./figures/punt_inside_20_distance_heatmap.png')
plt.clf()

# TODO: Punt distance outside 20
punts_outside_20 = df[(df["punt_inside_twenty"] == 0) & (df["touchback"] == 0)]
grouped_punts_outside_20 = punts_outside_20.groupby("yardline_group")
punts_outside_20_distance_averages = grouped_punts_outside_20["relative_punt_landing"].agg(["mean", "std", "skew"])
punts_outside_20_distance_averages["skew"] = punts_outside_20_distance_averages["skew"].fillna(-0.7)
print(punts_outside_20_distance_averages)
print()

punt_outside_20_mean_model = LinearRegression()
pf = PolynomialFeatures(degree=3)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    punts_outside_20_distance_averages["mean"]
)
punt_outside_20_mean_model.fit(
    transformed_yardline_midpoints,
    punts_outside_20_distance_averages["mean"]
)
print("Punt outside 20 mean relative distance model")
print(f"Coef: {punt_outside_20_mean_model.coef_}")
print(f"Intr: {punt_outside_20_mean_model.intercept_}")
print()
punts_outside_20_mean_pred = punt_outside_20_mean_model.predict(pf.fit_transform(zero_to_hundred))

punt_outside_20_std_model = LinearRegression()
punt_outside_20_std_model.fit(
    yardline_group_midpoints,
    punts_outside_20_distance_averages["std"]
)
print("Punt outside 20 std relative distance model")
print(f"Coef: {punt_outside_20_std_model.coef_}")
print(f"Intr: {punt_outside_20_std_model.intercept_}")
print()

punt_outside_20_skew_model = LinearRegression()
pf = PolynomialFeatures(degree=2)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    punts_outside_20_distance_averages["skew"]
)
punt_outside_20_skew_model.fit(
    transformed_yardline_midpoints,
    punts_outside_20_distance_averages["skew"]
)
print("Punt outside 20 skew relative distance model")
print(f"Coef: {punt_outside_20_skew_model.coef_}")
print(f"Intr: {punt_outside_20_skew_model.intercept_}")
print()

punts_outside_20_std_pred = punt_outside_20_std_model.predict(zero_to_hundred)
punts_outside_20_skew_pred = punt_outside_20_skew_model.predict(pf.fit_transform(zero_to_hundred))
plt.scatter(yardline_group_midpoints, punts_outside_20_distance_averages["mean"], color='g')
plt.scatter(yardline_group_midpoints, punts_outside_20_distance_averages["std"], color='r')
plt.scatter(yardline_group_midpoints, punts_outside_20_distance_averages["skew"], color='y')
plt.plot(zero_to_hundred, punts_outside_20_mean_pred, color='b')
plt.plot(zero_to_hundred, punts_outside_20_std_pred, color='b')
plt.plot(zero_to_hundred, punts_outside_20_skew_pred, color='b')
plt.title("Relative punt landing for punts outside 20")
plt.xlabel("Yard line")
plt.ylabel("Relative punt landing")
plt.ylim(-2, 2)
plt.savefig('./figures/punt_distance_outside_20.png')
plt.clf()

# Plot heatmap of PDF of normal distributions as yard line changes
x_1d = np.linspace(-0.1, 1)
y_1d = np.linspace(0, 100)
X, Y = np.meshgrid(x_1d, y_1d)
Z = skewnorm.pdf(
    X,
    a=(
        punt_outside_20_skew_model.intercept_ + \
        (punt_outside_20_skew_model.coef_[1] * Y) + \
        (punt_outside_20_skew_model.coef_[2] * pow(Y, 2))
    ),
    loc=(
        punt_outside_20_mean_model.intercept_ + \
        (punt_outside_20_mean_model.coef_[1] * Y) + \
        (punt_outside_20_mean_model.coef_[2] * pow(Y, 2)) + \
        (punt_outside_20_mean_model.coef_[3] * pow(Y, 3))
    ),
    scale=(
        punt_outside_20_std_model.intercept_ + \
        punt_outside_20_std_model.coef_[0] * Y
    )
)
fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z, 100)
plt.colorbar(cf, ax=ax, label='Probability density')
plt.title('Probability density of punt outside the 20 by yard line')
plt.xlabel("Relative landing")
plt.ylabel("Yard line")
plt.savefig('./figures/punt_outside_20_distance_heatmap.png')
plt.clf()

# 5. Out of bounds?
def out_of_bounds_count(s):
    return (s == 1).sum()
grouped_yardline_oob_averages = grouped_yardline["punt_out_of_bounds"].agg(["count", out_of_bounds_count])
grouped_yardline_oob_averages["p_out_of_bounds"] = grouped_yardline_oob_averages["out_of_bounds_count"] / grouped_yardline_oob_averages["count"]
print(grouped_yardline_oob_averages)
print()

punt_out_of_bounds_model = LinearRegression()
pf = PolynomialFeatures(degree=2)
transformed_yardline_midpoints = pf.fit_transform(yardline_group_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    grouped_yardline_oob_averages["p_out_of_bounds"]
)
punt_out_of_bounds_model.fit(
    transformed_yardline_midpoints,
    grouped_yardline_oob_averages["p_out_of_bounds"]
)
print("Punt out of bounds probability model")
print(f"Coef: {punt_out_of_bounds_model.coef_}")
print(f"Intr: {punt_out_of_bounds_model.intercept_}")
print()

p_punt_oob_pred = punt_out_of_bounds_model.predict(pf.fit_transform(zero_to_hundred))
plt.scatter(yardline_group_midpoints, grouped_yardline_oob_averages["p_out_of_bounds"], color='g')
plt.plot(zero_to_hundred, p_punt_oob_pred, color='b')
plt.title("Probability of punt out of bounds by yard line")
plt.xlabel("Yard line")
plt.ylabel("Probability of punt out of bounds")
plt.ylim(0, 0.13)
plt.savefig('./figures/p_punt_out_of_bounds.png')
plt.clf()

# 6. Fair catch? Muffed?

# 7. Return yards

# 8. Fumble?

# 9. Duration
