import pandas as pd
from context.context import PlayContext
from playresult.fieldgoal.model import FieldGoalResultModel
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill

def generate_fg_result_data():
    df = pd.DataFrame(
        {
            "norm_diff_fg_skill": [],
            "yard_line": [],
            "fg_made": []
        }
    )
    model = FieldGoalResultModel()
    for i in range(10):
        fg_skill = 1-(i * (1/10))
        offense = OffensiveSkill(
            field_goals=fg_skill
        )
        for j in range(10):
            fg_def_skill = 1-(j * (1/10))
            defense = DefensiveSkill(
                field_goal_defense=fg_def_skill
            )
            for k in range(10):
                yard_line = k * 10
                context = PlayContext(
                    yard_line=yard_line
                )
                for l in range(10):
                    result = model.sim(
                        offense=offense,
                        defense=defense,
                        context=context
                    )
                    result_df = pd.DataFrame(
                        {
                            "norm_diff_fg_skill": [0.5 + ((fg_skill - fg_def_skill) / 2)],
                            "yard_line": [yard_line],
                            "fg_made": [int(result.field_goal_made)],
                            "fg_blocked": [int(result.field_goal_blocked)]
                        }
                    )
                    df = pd.concat([df, result_df])
    return df

df = generate_fg_result_data()
grouped_yardline = df.groupby("yard_line")
grouped_skill = df.groupby("norm_diff_fg_skill")
print(grouped_yardline)
print(grouped_skill)
def made_count(s):
    return (s == 1).sum()
grouped_yardline_averages = grouped_yardline["fg_made"].agg(["count", made_count])
grouped_yardline_averages["p_made"] = grouped_yardline_averages["made_count"] / grouped_yardline_averages["count"]
grouped_skill_averages = grouped_skill["fg_made"].agg(["count", made_count])
grouped_skill_averages["p_made"] = grouped_skill_averages["made_count"] / grouped_skill_averages["count"]
print(grouped_yardline_averages)
print(grouped_skill_averages)
