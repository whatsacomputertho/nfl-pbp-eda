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
            "fg_result": []
        }
    )
    model = FieldGoalResultModel(from_file=True)
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
                    result = model.play(
                        offense=offense,
                        defense=defense,
                        context=context
                    )
                    result_df = pd.DataFrame(
                        {
                            "norm_diff_fg_skill": [(fg_skill - fg_def_skill) / 2],
                            "yard_line": [yard_line],
                            "fg_result": [result]
                        }
                    )
                    df = pd.concat([df, result_df])
    return df
