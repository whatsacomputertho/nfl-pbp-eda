from context.context import GameContext
from playcalling.model import PlayCallingModel
from playcalling.playcall import PlayCall
from playresult.fieldgoal.model import FieldGoalResultModel
from playresult.fieldgoal.result import FieldGoalResult
from playresult.passing.model import PassResultModel
from playresult.passing.result import PassResult
from playresult.rushing.model import RushResultModel
from playresult.rushing.result import RushResult
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill

rushing_model = RushResultModel()
passing_model = PassResultModel()
field_goal_model = FieldGoalResultModel(from_file=True)
playcall_model = PlayCallingModel(from_file=True)
context = GameContext(
    home_team="CAR",
    away_team="NYM"
)
print(str(context.into_play_context()))

result = None

while not context.game_over:
    play_context = context.into_play_context()
    playcall = playcall_model.play(play_context)
    print(f"Play type: {str(playcall)}")
    if playcall in [PlayCall.RUN_LEFT, PlayCall.RUN_MIDDLE, PlayCall.RUN_RIGHT]:
        result = rushing_model.sim(
            context=play_context,
            offense=OffensiveSkill(),
            defense=DefensiveSkill()
        )
    elif playcall in [PlayCall.SHORT_PASS, PlayCall.DEEP_PASS]:
        result = passing_model.sim(
            context=play_context,
            offense=OffensiveSkill(),
            defense=DefensiveSkill()
        )
    elif playcall in [PlayCall.FIELD_GOAL, PlayCall.EXTRA_POINT]:
        result = field_goal_model.play(
            offense=OffensiveSkill(),
            defense=DefensiveSkill(),
            context=play_context
        )
    print(result)
    context = result.next_context(context)
    print(str(context.into_play_context()))
