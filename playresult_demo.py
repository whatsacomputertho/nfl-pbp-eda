from context.context import GameContext
from playcalling.model import PlayCallingModel
from playcalling.playcall import PlayCall
from playresult.model import PlayResultModel
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill

playresult_model = PlayResultModel(from_file=True)
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
    if result == None:
        playcall = PlayCall.KICKOFF
    elif context.down == 0:
        playcall = PlayCall.EXTRA_POINT
    print(f"Play type: {str(playcall)}")
    result = playresult_model.play(
        offense=OffensiveSkill(),
        defense=DefensiveSkill(),
        context=play_context,
        playcall=playcall
    )
    context = result.next_context(context)
    print(str(context.into_play_context()))
