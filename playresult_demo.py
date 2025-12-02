from context.context import GameContext
from playcalling.model import PlayCallingModel
from playcalling.playcall import PlayCall
from playresult.betweenplay.model import BetweenPlayModel
from playresult.fieldgoal.model import FieldGoalResultModel
from playresult.fieldgoal.result import FieldGoalResult
from playresult.kickoff.model import KickoffResultModel
from playresult.kickoff.result import KickoffResult
from playresult.passing.model import PassResultModel
from playresult.passing.result import PassResult
from playresult.rushing.model import RushResultModel
from playresult.rushing.result import RushResult
from playresult.punt.model import PuntResultModel
from playresult.punt.result import PuntResult
from team.coach import CoachSkill
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill

field_goal_model = FieldGoalResultModel()
kickoff_model = KickoffResultModel()
rushing_model = RushResultModel()
passing_model = PassResultModel()
punt_model = PuntResultModel()
playcall_model = PlayCallingModel()
between_play_model = BetweenPlayModel()
context = GameContext(
    home_team="CAR",
    away_team="NYM"
)

result = None
playcall = PlayCall.KICKOFF

while not context.game_over:
    play_context = context.into_play_context()
    if playcall == PlayCall.RUN:
        result = rushing_model.sim(
            context=play_context,
            offense=OffensiveSkill(),
            defense=DefensiveSkill()
        )
    elif playcall == PlayCall.PASS:
        result = passing_model.sim(
            context=play_context,
            offense=OffensiveSkill(),
            defense=DefensiveSkill()
        )
    elif playcall == PlayCall.FIELD_GOAL:
        result = field_goal_model.sim(
            offense=OffensiveSkill(),
            defense=DefensiveSkill(),
            context=play_context
        )
    elif playcall == PlayCall.EXTRA_POINT:
        result = field_goal_model.sim(
            offense=OffensiveSkill(),
            defense=DefensiveSkill(),
            context=play_context,
            is_extra_point=True
        )
    elif playcall == PlayCall.PUNT:
        result = punt_model.sim(
            context=play_context,
            offense=OffensiveSkill(),
            defense=DefensiveSkill()
        )
    elif playcall == PlayCall.KICKOFF:
        result = kickoff_model.sim(
            context=play_context,
            offense=OffensiveSkill(),
            defense=DefensiveSkill()
        )
    print(f"{context.result_prefix()} {str(result)}")
    context = result.next_context(context)
    is_clock_running = True
    coach_skill = CoachSkill()
    between_play_duration, is_timeout, is_def_timeout = between_play_model.sim(
        context.into_play_context(),
        coach_skill.risk_taking,
        coach_skill.up_tempo,
        is_clock_running
    )
    if is_timeout:
        if context.home_possession ^ is_def_timeout:
            context.home_timeouts -= 1
        else:
            context.away_timeouts -= 1
    context.update_clock(between_play_duration)
    if context.next_play_kickoff:
        playcall = PlayCall.KICKOFF
    elif context.next_play_extra_point:
        playcall = PlayCall.EXTRA_POINT
    else:
        playcall = playcall_model.sim(
            context.into_play_context(),
            coach_skill
        )
