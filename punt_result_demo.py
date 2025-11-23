from context.context import PlayContext
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill
from playresult.punt.model import PuntResultModel

model = PuntResultModel()
context = PlayContext(
    2,      # Second quarter
    900,    # 15:00 clock
    4,      # Fourth down
    10,     # 10 yards to first
    35,     # Own 35 yard line
    0,      # Tied
    3,      # Defense has 3 timeouts
    3,      # Offense has 3 timeouts
)
for _ in range(100):
    result = model.sim(
        context=context,
        offense=OffensiveSkill(),
        defense=DefensiveSkill()
    )
    print(result)
