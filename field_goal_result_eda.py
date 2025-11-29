import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load data
df = pd.read_csv("./data/fgs.csv")

###
# Field goal model flow
# 1. Blocked?
# 2. If blocked, return yards
# 3. Made?
# 4. Duration
###

# 1. Blocked?
df["field_goal_blocked"] = 0
df.loc[df["field_goal_result"] == "blocked", "field_goal_blocked"] = 1
df['norm_diff_blocked_percent_group'] = pd.cut(df['norm_diff_blocked_percent'], bins=4)
grouped_blocked_percent = df.groupby('norm_diff_blocked_percent_group')
def blocked_count(s):
    return (s == 1).sum()
grouped_blocked_percent_averages = grouped_blocked_percent["field_goal_blocked"].agg(["count", blocked_count])
grouped_blocked_percent_averages["p_block"] = grouped_blocked_percent_averages["blocked_count"] / grouped_blocked_percent_averages["count"]
print(grouped_blocked_percent_averages)
print()

grouped_blocked_percent_midpoints = [[(name.left + name.right)/2] for name in grouped_blocked_percent_averages.index]
field_goal_blocked_model = LinearRegression()
field_goal_blocked_model.fit(
    grouped_blocked_percent_midpoints,
    grouped_blocked_percent_averages["p_block"]
)
zero_to_one = pd.DataFrame(np.linspace(0, 1))
field_goal_blocked_pred = field_goal_blocked_model.predict(zero_to_one)
print("Field goal blocked model (skill-based)")
print(f"Coef: {field_goal_blocked_model.coef_}")
print(f"Intr: {field_goal_blocked_model.intercept_}")
print()

plt.scatter(grouped_blocked_percent_midpoints, grouped_blocked_percent_averages["p_block"], color='g')
plt.plot(zero_to_one, field_goal_blocked_pred, color='b')
plt.title("Field goal blocked probability by normalized blocking skill differential")
plt.xlabel("Normalized blocking skill differential")
plt.ylabel("Field goal blocked probability")
plt.savefig("./figures/field_goal_blocked_skill.png")
plt.clf()

df['yard_line_group'] = pd.cut(df['yardline_100'], bins=5)
grouped_blocked_yardline_percent = df.groupby('yard_line_group')
grouped_blocked_yardline_percent_averages = grouped_blocked_yardline_percent["field_goal_blocked"].agg(["count", blocked_count])
grouped_blocked_yardline_percent_averages["p_block"] = grouped_blocked_yardline_percent_averages["blocked_count"] / grouped_blocked_yardline_percent_averages["count"]
print(grouped_blocked_yardline_percent_averages)
print()

grouped_blocked_percent_yardline_midpoints = [[(name.left + name.right)/2] for name in grouped_blocked_yardline_percent_averages.index]
transformed_p_block = np.log(grouped_blocked_yardline_percent_averages["p_block"])
p_block_yardline_model = LinearRegression()
p_block_yardline_model.fit(
    grouped_blocked_percent_yardline_midpoints,
    transformed_p_block
)
zero_to_sixty = pd.DataFrame(np.linspace(0, 60))
fg_blocked_yardline_pred = np.exp(p_block_yardline_model.predict(zero_to_sixty))
print("Field goal blocked model (yard line-based)")
print(f"coef: {p_block_yardline_model.coef_}")
print(f"intr: {p_block_yardline_model.intercept_}")
print()

plt.scatter(grouped_blocked_percent_yardline_midpoints, grouped_blocked_yardline_percent_averages["p_block"], color='g')
plt.plot(zero_to_sixty, fg_blocked_yardline_pred, color='b')
plt.title("Field goal blocked probability by yard line")
plt.xlabel("Yard line")
plt.ylabel("Field goal blocked probability")
plt.savefig("./figures/field_goal_blocked_yardline.png")
plt.clf()

def field_goal_blocked_skill(skill):
    return (field_goal_blocked_model.coef_[0] * skill) + field_goal_blocked_model.intercept_
def field_goal_blocked_yardline(yardline):
    return np.exp((p_block_yardline_model.coef_[0] * yardline) + p_block_yardline_model.intercept_)
x_1d = np.linspace(0, 100)
y_1d = np.linspace(0, 1)
X, Y = np.meshgrid(x_1d, y_1d)
Z = ((field_goal_blocked_skill(Y) * 0.7) + (field_goal_blocked_yardline(X) * 0.3)) * 0.7
fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z, 100)
plt.colorbar(cf, ax=ax, label='Field goal blocked probability')
plt.title('Field goal blocked probability')
plt.xlabel("Yard line")
plt.ylabel("Normalized blocking skill differential")
plt.savefig('./figures/field_goal_blocked_heatmap.png')
plt.clf()

# 2. If blocked, return yards
blocked_field_goals = df[df["field_goal_blocked"] == 1]
return_yard_averages = blocked_field_goals["return_yards"].agg(["mean", "std", "skew"])
print(return_yard_averages)
print()
# NOTE: Exponential distribution with Lambda = 0

# 3. Made?
df["field_goal_made"] = 0
df.loc[df["field_goal_result"] == "made", "field_goal_made"] = 1
df['norm_field_goal_percent_group'] = pd.cut(df['norm_field_goal_percent'], bins=4)
grouped_field_goal_percent = df.groupby('norm_field_goal_percent_group')
def made_count(s):
    return (s == 1).sum()
grouped_field_goal_percent_averages = grouped_field_goal_percent["field_goal_made"].agg(["count", made_count])
grouped_field_goal_percent_averages["p_made"] = grouped_field_goal_percent_averages["made_count"] / grouped_field_goal_percent_averages["count"]
print(grouped_field_goal_percent_averages)
print()

grouped_field_goal_percent_midpoints = [[(name.left + name.right)/2] for name in grouped_field_goal_percent_averages.index]
field_goal_made_skill_model = LinearRegression()
field_goal_made_skill_model.fit(
    grouped_field_goal_percent_midpoints,
    grouped_field_goal_percent_averages["p_made"]
)
zero_to_one = pd.DataFrame(np.linspace(0, 1))
field_goal_made_skill_pred = field_goal_made_skill_model.predict(zero_to_one)
print("Field goal made model (skill-based)")
print(f"Coef: {field_goal_made_skill_model.coef_}")
print(f"Intr: {field_goal_made_skill_model.intercept_}")
print()

plt.scatter(grouped_field_goal_percent_midpoints, grouped_field_goal_percent_averages["p_made"], color='g')
plt.plot(zero_to_one, field_goal_made_skill_pred, color='b')
plt.title("Field goal made probability by normalized kicking skill")
plt.xlabel("Normalized kicking skill")
plt.ylabel("Field goal made probability")
plt.savefig("./figures/field_goal_made_skill.png")
plt.clf()

grouped_field_goal_yardline_percent = df.groupby('yard_line_group')
grouped_fg_yardline_percent_averages = grouped_field_goal_yardline_percent['field_goal_made'].agg(["count", made_count])
grouped_fg_yardline_percent_averages["p_made"] = grouped_fg_yardline_percent_averages["made_count"] / grouped_fg_yardline_percent_averages["count"]
print(grouped_fg_yardline_percent_averages)
print()

grouped_fg_yardline_midpoints = [[(name.left + name.right)/2] for name in grouped_fg_yardline_percent_averages.index]
field_goal_made_yardline_model = LinearRegression()
pf = PolynomialFeatures(degree=2)
transformed_yardline_midpoints = pf.fit_transform(grouped_fg_yardline_midpoints)
pf.fit(
    transformed_yardline_midpoints,
    grouped_fg_yardline_percent_averages["p_made"]
)
field_goal_made_yardline_model.fit(
    transformed_yardline_midpoints,
    grouped_fg_yardline_percent_averages["p_made"]
)
field_goal_made_yardline_pred = field_goal_made_yardline_model.predict(pf.fit_transform(zero_to_sixty))
print("Field goal made model (yard line-based)")
print(f"Coef: {field_goal_made_yardline_model.coef_}")
print(f"Intr: {field_goal_made_yardline_model.intercept_}")
print()

plt.scatter(grouped_fg_yardline_midpoints, grouped_fg_yardline_percent_averages["p_made"], color='g')
plt.plot(zero_to_sixty, field_goal_made_yardline_pred, color='b')
plt.title("Field goal made probability by yard line")
plt.xlabel("Yard line")
plt.ylabel("Field goal made probability")
plt.savefig("./figures/field_goal_made_yardline.png")
plt.clf()

def field_goal_made_skill(skill):
    return (field_goal_made_skill_model.coef_[0] * skill) + field_goal_made_skill_model.intercept_
def field_goal_made_yardline(yardline):
    return (field_goal_made_yardline_model.coef_[1] * yardline) + \
            (field_goal_made_yardline_model.coef_[2] * pow(yardline, 2)) + \
            field_goal_made_yardline_model.intercept_
x_1d = np.linspace(0, 100)
y_1d = np.linspace(0, 1)
X, Y = np.meshgrid(x_1d, y_1d)
Z = ((field_goal_made_skill(Y) * 0.4) + (field_goal_made_yardline(X) * 0.6)) * 1.18
fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z, 100)
plt.colorbar(cf, ax=ax, label='Field goal made probability')
plt.title('Field goal made probability')
plt.xlabel("Yard line")
plt.ylabel("Kicking skill")
plt.savefig('./figures/field_goal_made_heatmap.png')
plt.clf()

# 4. Duration
non_blocked_fg_durations = df[(df["field_goal_blocked"] == 0) & (df["play_duration"] < 10)]
non_blocked_duration_averages = non_blocked_fg_durations["play_duration"].agg(["mean", "std", "skew"])
print(non_blocked_duration_averages)
print()

blocked_fg_durations = df[df["field_goal_blocked"] == 1]
blocked_fg_duration_averages = blocked_fg_durations["play_duration"].agg(["mean", "std", "skew"])
print(blocked_fg_duration_averages)
print()
