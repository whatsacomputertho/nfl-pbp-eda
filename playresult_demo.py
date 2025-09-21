from context.context import GameContext
from playcalling.playcall import PlayCall
from playresult.model import PlayResultModel
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill

model = PlayResultModel(from_file=True)
result = model.play(
    offense=OffensiveSkill(),
    defense=DefensiveSkill(),
    context=GameContext(
        2,      # Second quarter
        102,    # 1:02 clock
        2,      # Second down
        6,      # 6 yards to first
        39,     # Own 39 yard line
        -3,     # Down by 3
        2,      # Defense has 2 timeouts
        2,      # Offense has 2 timeouts
    ),
    playcall=PlayCall.SHORT_PASS
)
print(str(result))
