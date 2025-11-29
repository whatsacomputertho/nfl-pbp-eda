from context.context import PlayContext, GameContext
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill
from playresult.fieldgoal.model import FieldGoalResultModel

model = FieldGoalResultModel()
context = GameContext(
    home_team="NYM",
    away_team="CAR",
    quarter=2,
    half_seconds=900,
    down=4,
    distance=10,
    yard_line=75,
    home_score=0,
    away_score=0,
    home_positive_direction=True,
    home_opening_kickoff=True,
    home_possession=True,
    home_timeouts=3,
    away_timeouts=3,
    game_over=False
)
for _ in range(100):
    result = model.sim(
        context=context,
        offense=OffensiveSkill(field_goals=0, blocking=1),
        defense=DefensiveSkill(blitzing=0)
    )
    print(result)
