from context.context import PlayContext
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill
from playresult.passing.model import PassResultModel
from playresult.passing.result import PassResult

model = PassResultModel()
context = PlayContext(
    2,      # Second quarter
    900,    # 15:00 clock
    1,      # First down
    10,     # 10 yards to first
    35,     # Own 35 yard line
    0,      # Tied
    3,      # Defense has 3 timeouts
    3,      # Offense has 3 timeouts
)

yards_gained = 0
pass_att = 0
completions = 0
zero_yac = 0
total_yac = 0
for _ in range(100):
    result = model.sim(
        context=context,
        offense=OffensiveSkill(),
        defense=DefensiveSkill()
    )
    print(result)
    if isinstance(result, PassResult):
        pass_att += 1
        if result.complete:
            completions += 1
            if result.yac == 0:
                zero_yac += 1
            total_yac += result.yac
        yards_gained += result.yards_gained()
    else:
        yards_gained += result.yards_gained

print()
print("Summary")
print(f"Completion percentage: {completions / pass_att}")
print(f"Average yards gained: {yards_gained / 100}")
print(f"Zero YAC percentage: {zero_yac / pass_att}")
print(f"Average YAC: {total_yac / completions}")
