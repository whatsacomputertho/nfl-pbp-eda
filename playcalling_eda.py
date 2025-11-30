import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.pbp import load_clean_nfl_pbp_playcall_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('./data/playcall.csv')
df["norm_run_percent_group"] = pd.cut(df["norm_run_percent"], bins=4)
df["norm_go_for_it_percent_group"] = pd.cut(df["norm_go_for_it_percent"], bins=4)
df["is_run_play"] = 0
df.loc[df["play_type"] == "run", "is_run_play"] = 1

###
# 1st - 3rd down playcall model flow
# 1. Is clock management scenario?
# 2. If clock management scenario, weigh outcomes (TD, FG)
#    - < 4 -> FG
#    - 4 - 8 -> TD
#    - 9 - 17 -> TD or FG 20:80
# 3. Is last play?
# 4. If last play, final playcall based on weighed outcomes
# 5. If clock management scenario, conservative play call
# 6. If winning & can run out clock, kneel
# 7. Non-clock management play call
###

# Clock management play call
clock_management_scenarios = df[(df["qtr"] == 4) & (df["half_seconds_remaining"] <= 180) & (df["score_diff"].abs() < 17) & (df["down"] < 4)]
clock_management_conserve_scenarios = clock_management_scenarios[clock_management_scenarios["score_diff"] <= 0]
print(clock_management_conserve_scenarios["play_type"].value_counts())
print()

# First down clock management play call
clock_management_first = clock_management_conserve_scenarios[clock_management_conserve_scenarios["down"] == 1]
def run_count(s):
    return (s == 1).sum()
cm_first_down_averages = clock_management_first["is_run_play"].agg(["count", run_count])
cm_first_down_averages["p_run"] = cm_first_down_averages["run_count"] / cm_first_down_averages["count"]
print(cm_first_down_averages)
print()

# Second down clock management play call
clock_management_second = clock_management_conserve_scenarios[clock_management_conserve_scenarios["down"] == 2]
cm_second_down_averages = clock_management_second["is_run_play"].agg(["count", run_count])
cm_second_down_averages["p_run"] = cm_second_down_averages["run_count"] / cm_second_down_averages["count"]
print(cm_second_down_averages)
print()

# Third down clock management play call
clock_management_third = clock_management_conserve_scenarios[clock_management_conserve_scenarios["down"] == 2]
cm_third_down_averages = clock_management_third["is_run_play"].agg(["count", run_count])
cm_third_down_averages["p_run"] = cm_third_down_averages["run_count"] / cm_third_down_averages["count"]
print(cm_third_down_averages)
print()
# NOTE: Constant 0.15 p_run on clock management scenarios

# No timeouts left play call
clock_management_no_timeouts = clock_management_scenarios[clock_management_scenarios["posteam_timeouts_remaining"] < 1]
cm_no_timeout_averages = clock_management_no_timeouts["is_run_play"].agg(["count", run_count])
cm_no_timeout_averages["p_run"] = cm_no_timeout_averages["run_count"] / cm_no_timeout_averages["count"]
print(cm_no_timeout_averages)
print()
# NOTE: Constant 0.09 p_run on clock management scenarios with no timeouts left

# Fist down playcall
first_downs = df[df["down"] == 1]
grouped_first_down = first_downs.groupby("norm_run_percent_group")
grouped_first_down_averages = grouped_first_down["is_run_play"].agg(["count", run_count])
grouped_first_down_averages["p_run"] = grouped_first_down_averages["run_count"] / grouped_first_down_averages["count"]
print(grouped_first_down_averages)
print()

playcall_midpoints = [[(name.left + name.right)/2] for name in grouped_first_down_averages.index]
first_down_model = LinearRegression()
first_down_model.fit(
    playcall_midpoints,
    grouped_first_down_averages["p_run"]
)
zero_to_one = pd.DataFrame(np.linspace(0, 1))
first_down_model_pred = first_down_model.predict(zero_to_one)
print("First down run probability model")
print(f"Coef: {first_down_model.coef_}")
print(f"Intr: {first_down_model.intercept_}")
print()

# Second down playcall
second_downs = df[df["down"] == 2]
grouped_second_down = second_downs.groupby("norm_run_percent_group")
grouped_second_down_averages = grouped_second_down["is_run_play"].agg(["count", run_count])
grouped_second_down_averages["p_run"] = grouped_second_down_averages["run_count"] / grouped_second_down_averages["count"]
print(grouped_second_down_averages)
print()

second_down_model = LinearRegression()
second_down_model.fit(
    playcall_midpoints,
    grouped_second_down_averages["p_run"]
)
second_down_model_pred = second_down_model.predict(zero_to_one)
print("Second down run probability model")
print(f"Coef: {second_down_model.coef_}")
print(f"Intr: {second_down_model.intercept_}")
print()

# Third down playcall
third_downs = df[df["down"] == 3]
grouped_third_down = third_downs.groupby("norm_run_percent_group")
grouped_third_down_averages = grouped_third_down["is_run_play"].agg(["count", run_count])
grouped_third_down_averages["p_run"] = grouped_third_down_averages["run_count"] / grouped_third_down_averages["count"]
print(grouped_third_down_averages)
print()

third_down_model = LinearRegression()
third_down_model.fit(
    playcall_midpoints,
    grouped_third_down_averages["p_run"]
)
third_down_model_pred = third_down_model.predict(zero_to_one)
print("Third down run probability model")
print(f"Coef: {third_down_model.coef_}")
print(f"Intr: {third_down_model.intercept_}")
print()

plt.scatter(playcall_midpoints, grouped_first_down_averages["p_run"], color='g')
plt.scatter(playcall_midpoints, grouped_second_down_averages["p_run"], color='r')
plt.scatter(playcall_midpoints, grouped_third_down_averages["p_run"], color='y')
plt.plot(zero_to_one, first_down_model_pred, color='b')
plt.plot(zero_to_one, second_down_model_pred, color='b')
plt.plot(zero_to_one, third_down_model_pred, color='b')
plt.title("Probability of a run play call by playcalling tendency")
plt.xlabel("Playcalling tendency")
plt.ylabel("Probability of a run playcall")
plt.savefig("./figures/run_playcall_probability.png")
plt.clf()

# 1st-3rd playcall by distance remaining
df["distance_group"] = pd.cut(df["ydstogo"], bins=10)
grouped_distance = df.groupby("distance_group")
grouped_distance_averages = grouped_distance["is_run_play"].agg(["count", run_count])
grouped_distance_averages["p_run"] = grouped_distance_averages["run_count"] / grouped_distance_averages["count"]
print(grouped_distance_averages)
print()

distance_midpoints = [[(name.left + name.right)/2] for name in grouped_distance_averages.index]
dist_remaining_model = LinearRegression()
dist_remaining_model.fit(
    distance_midpoints,
    grouped_distance_averages["p_run"]
)
zero_to_fifty = pd.DataFrame(np.linspace(0, 50))
dist_remaining_model_pred = dist_remaining_model.predict(zero_to_fifty)
print("Distance remaining run probability model")
print(f"Coef: {dist_remaining_model.coef_}")
print(f"Intr: {dist_remaining_model.intercept_}")
print()

plt.scatter(distance_midpoints, grouped_distance_averages["p_run"], color='g')
plt.plot(zero_to_fifty, dist_remaining_model_pred, color='b')
plt.title("Probability of a run play call by distance remaining")
plt.xlabel("Distance to go")
plt.ylabel("Probability of a run playcall")
plt.savefig("./figures/run_playcall_dist_probability.png")
plt.clf()

def p_first_down_run_skill(skill):
    return (first_down_model.coef_[0] * skill) + first_down_model.intercept_
def p_second_down_run_skill(skill):
    return (second_down_model.coef_[0] * skill) + second_down_model.intercept_
def p_third_down_run_skill(skill):
    return (third_down_model.coef_[0] * skill) + third_down_model.intercept_
def p_run_dist(yardline):
    return (dist_remaining_model.coef_[0] * yardline) + dist_remaining_model.intercept_
x_1d = np.linspace(0, 100)
y_1d = np.linspace(0, 1)
X, Y = np.meshgrid(x_1d, y_1d)
Z_FIRST = ((p_first_down_run_skill(Y) * 0.7) + (p_run_dist(X) * 0.3))
Z_SECOND = ((p_second_down_run_skill(Y) * 0.7) + (p_run_dist(X) * 0.3))
Z_THIRD = ((p_third_down_run_skill(Y) * 0.7) + (p_run_dist(X) * 0.3))
fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z_FIRST, 100)
plt.colorbar(cf, ax=ax, label='First down run probability')
plt.title('First down run probability')
plt.xlabel("Yards to first")
plt.ylabel("Play calling tendency")
plt.savefig('./figures/first_down_run_heatmap.png')
plt.clf()
fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z_SECOND, 100)
plt.colorbar(cf, ax=ax, label='Second down run probability')
plt.title('Second down run probability')
plt.xlabel("Yards to first")
plt.ylabel("Play calling tendency")
plt.savefig('./figures/second_down_run_heatmap.png')
plt.clf()
fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z_THIRD, 100)
plt.colorbar(cf, ax=ax, label='Third down run probability')
plt.title('Third down run probability')
plt.xlabel("Yards to first")
plt.ylabel("Play calling tendency")
plt.savefig('./figures/third_down_run_heatmap.png')
plt.clf()

###
# 4th down playcall model flow
# 1. Is must score scenario?
# 2. If must score scenario, weigh options (TD, FG)
# - Scenarios
#    - < 4 -> FG
#    - 4 - 8 -> TD
#    - 9 - 17 -> TD or FG 20:80
# 3. If must score TD scenario -> Go for it (pass), else FG
# 4. Standard 4th down playcall
# - Go for it
#   - Between OWN 40 and OPP 40, < 4 yards remaining
# - Field goal
#   - In OPP territory
# - Punt
#   - Non-go-for-it, 4th and long outside of OPP 40
###

# Field goal scenarios
df["yard_line_group"] = pd.cut(df["yardline_100"], bins=10)
fourth_downs = df[df["down"] == 4]
field_goal_scenarios = fourth_downs[(fourth_downs["yardline_100"] <= 40)]
field_goal_scenarios["is_field_goal"] = 0
field_goal_scenarios.loc[field_goal_scenarios["play_type"] == "field_goal", "is_field_goal"] = 1
def field_goal_count(s):
    return (s == 1).sum()
grouped_field_goal = field_goal_scenarios.groupby("norm_go_for_it_percent_group")
field_goal_averages = grouped_field_goal["is_field_goal"].agg(["count", field_goal_count])
field_goal_averages["p_field_goal"] = field_goal_averages["field_goal_count"] / field_goal_averages["count"]
print(field_goal_averages)
print()

go_for_it_midpoints = [[(name.left + name.right)/2] for name in field_goal_averages.index]
field_goal_risk_model = LinearRegression()
field_goal_risk_model.fit(
    go_for_it_midpoints,
    field_goal_averages["p_field_goal"]
)
field_goal_risk_pred = field_goal_risk_model.predict(zero_to_one)
print("Field goal probability (risk-based)")
print(f"Coef: {field_goal_risk_model.coef_}")
print(f"Intr: {field_goal_risk_model.intercept_}")
print()

plt.scatter(go_for_it_midpoints, field_goal_averages["p_field_goal"], color='g')
plt.plot(zero_to_one, field_goal_risk_pred, color='b')
plt.title("Field goal probability by risk taking attribute")
plt.xlabel("Risk taking")
plt.ylabel("Field goal probability")
plt.savefig("./figures/field_goal_probability_risk.png")
plt.clf()

fourth_downs["is_field_goal"] = 0
fourth_downs.loc[fourth_downs["play_type"] == "field_goal", "is_field_goal"] = 1
fourth_downs_in_opp_territory = fourth_downs[fourth_downs["yardline_100"] <= 50]
fourth_downs_in_opp_territory["yard_line_group"] = pd.cut(fourth_downs_in_opp_territory["yardline_100"], bins=5)
grouped_fourth_down_yardline = fourth_downs_in_opp_territory.groupby("yard_line_group")
field_goal_yardline_averages = grouped_fourth_down_yardline["is_field_goal"].agg(["count", field_goal_count])
field_goal_yardline_averages["p_field_goal"] = field_goal_yardline_averages["field_goal_count"] / field_goal_yardline_averages["count"]
field_goal_yardline_averages["p_field_goal"] = field_goal_yardline_averages["p_field_goal"].fillna(0)
print(field_goal_yardline_averages)
print()

yard_line_midpoints = [[name.right] for name in field_goal_yardline_averages.index]
field_goal_yard_line_model = LinearRegression()
pf = PolynomialFeatures()
transformed_yardline = pf.fit_transform(yard_line_midpoints)
pf.fit(
    transformed_yardline,
    field_goal_yardline_averages["p_field_goal"]
)
field_goal_yard_line_model.fit(
    transformed_yardline,
    field_goal_yardline_averages["p_field_goal"]
)
field_goal_yard_line_pred = field_goal_yard_line_model.predict(pf.fit_transform(zero_to_fifty))
print("Field goal probability (yard line-based)")
print(f"Coef: {field_goal_yard_line_model.coef_}")
print(f"Intr: {field_goal_yard_line_model.intercept_}")
print()

plt.scatter(yard_line_midpoints, field_goal_yardline_averages["p_field_goal"], color='g')
plt.plot(zero_to_fifty, field_goal_yard_line_pred, color='b')
plt.title("Field goal probability by yard line")
plt.xlabel("Yard line")
plt.ylabel("Field goal probability")
plt.savefig("./figures/field_goal_probability_yardline.png")
plt.clf()

def p_field_goal_risk(risk):
    return (field_goal_risk_model.coef_[0] * risk) + field_goal_risk_model.intercept_
def p_field_goal_yard_line(yardline):
    return (field_goal_yard_line_model.coef_[1] * yardline) + \
        (field_goal_yard_line_model.coef_[2] * pow(yardline, 2)) + \
        field_goal_yard_line_model.intercept_
x_1d = np.linspace(0, 100)
y_1d = np.linspace(0, 1)
X, Y = np.meshgrid(x_1d, y_1d)
Z = ((p_field_goal_risk(Y) * 0.4) + (p_field_goal_yard_line(X) * 0.6))
fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z, 100)
plt.colorbar(cf, ax=ax, label='Field goal probability')
plt.title('Field goal probability')
plt.xlabel("Yard line")
plt.ylabel("Risk taking attribute")
plt.savefig('./figures/field_goal_probability_heatmap.png')
plt.clf()

# Go for it scenarios
go_for_it_scenarios = fourth_downs[(fourth_downs["ydstogo"] <= 4) & (fourth_downs["yardline_100"] > 40) & (fourth_downs["yardline_100"] < 60)]
go_for_it_scenarios["go_for_it"] = 0
go_for_it_scenarios.loc[(go_for_it_scenarios["play_type"] == "pass") | (go_for_it_scenarios["play_type"] == "run"), "go_for_it"] = 1
def go_for_it_count(s):
    return (s == 1).sum()
grouped_go_for_it = go_for_it_scenarios.groupby("norm_go_for_it_percent_group")
go_for_it_averages = grouped_go_for_it["go_for_it"].agg(["count", go_for_it_count])
go_for_it_averages["p_go_for_it"] = go_for_it_averages["go_for_it_count"] / go_for_it_averages["count"]
print(go_for_it_averages)
print()

go_for_it_midpoints = [[(name.left + name.right)/2] for name in go_for_it_averages.index]
go_for_it_model = LinearRegression()
go_for_it_model.fit(
    go_for_it_midpoints,
    go_for_it_averages["p_go_for_it"]
)
go_for_it_pred = go_for_it_model.predict(zero_to_one)
print("Go for it model")
print(f"Coef: {go_for_it_model.coef_}")
print(f"Intr: {go_for_it_model.intercept_}")
print()

plt.scatter(go_for_it_midpoints, go_for_it_averages["p_go_for_it"], color='g')
plt.plot(zero_to_one, go_for_it_pred, color='b')
plt.title("Go for it probability by risk taking attribute")
plt.xlabel("Risk taking")
plt.ylabel("Go for it probability")
plt.savefig('./figures/go_for_it_risk.png')
plt.clf()

# Fourth down go for it (non-desperation) playcall
grouped_fourth_down = fourth_downs.groupby("norm_run_percent_group")
grouped_fourth_down_averages = grouped_fourth_down["is_run_play"].agg(["count", run_count])
grouped_fourth_down_averages["p_run"] = grouped_fourth_down_averages["run_count"] / grouped_fourth_down_averages["count"]
print(grouped_fourth_down_averages)
print()

fourth_down_run_model = LinearRegression()
fourth_down_run_model.fit(
    playcall_midpoints,
    grouped_fourth_down_averages["p_run"]
)
fourth_down_run_pred = fourth_down_run_model.predict(zero_to_one)
print("Fourth down run model")
print(f"Coef: {fourth_down_run_model.coef_}")
print(f"Intr: {fourth_down_run_model.intercept_}")
print()

plt.scatter(playcall_midpoints, grouped_fourth_down_averages["p_run"], color='g')
plt.plot(zero_to_one, fourth_down_run_pred, color='b')
plt.title("Fourth down run probability by run-pass tendency")
plt.xlabel("Run-pass tendency")
plt.ylabel("Run probability")
plt.savefig('./figures/fourth_down_run_probability.png')
plt.clf()

def p_fourth_down_run_skill(skill):
    return (fourth_down_run_model.coef_[0] * skill) + fourth_down_run_model.intercept_
Z_FOURTH = ((p_fourth_down_run_skill(Y) * 0.7) + (p_run_dist(X) * 0.3))
fig, ax = plt.subplots()
cf = plt.contourf(X, Y, Z_FOURTH, 100)
plt.colorbar(cf, ax=ax, label='Fourth down run probability')
plt.title('Fourth down run probability')
plt.xlabel("Yards to first")
plt.ylabel("Play calling tendency")
plt.savefig('./figures/fourth_down_run_heatmap.png')
plt.clf()
