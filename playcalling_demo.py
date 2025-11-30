import os
from context.context import GameContext
from playcalling.model import PlayCallingModel
from team.coach import CoachSkill

WORKDIR = os.path.dirname(os.path.abspath(__file__))
model = PlayCallingModel()
coach = CoachSkill()

# A first playcall after a kickoff
after_kickoff = GameContext(
    home_team="NYM",
    away_team="CAR",
    quarter=2,          # Second quarter
    half_seconds=900,   # 15:00 clock
    down=1,             # First down
    distance=10,        # 10 yards to first
    yard_line=35,       # Own 35 yard line
    home_score=7,       # Tied
    away_score=7,
    away_timeouts=3,    # Defense has 3 timeouts
    home_timeouts=3,    # Offense has 3 timeouts
)
play_context = after_kickoff.into_play_context()
playcall = model.sim(play_context, coach)
print("After kickoff scenario:")
print(str(play_context))
print()
print(f"Playcall: {str(playcall)}")
print()

# A two-minute drill scenario
two_minute_drill = GameContext(
    home_team="NYM",
    away_team="CAR",
    quarter=2,          # Second quarter
    half_seconds=102,   # 1:02 clock
    down=2,             # Second down
    distance=6,         # 6 yards to first
    yard_line=39,       # Own 39 yard line
    home_score=7,       # Down by 3
    away_score=10,
    away_timeouts=2,    # Defense has 2 timeouts
    home_timeouts=2,    # Offense has 2 timeouts
)
play_context = two_minute_drill.into_play_context()
playcall = model.sim(play_context, coach)
print("Two-minute drill scenario:")
print(str(play_context))
print()
print(f"Playcall: {str(playcall)}")
print()

# A late-game goal line stand scenario
goal_line_stand = GameContext(
    home_team="NYM",
    away_team="CAR",
    quarter=4,          # Fourth quarter
    half_seconds=18,    # 0:18 clock
    down=4 ,            # Fourth down
    distance=2,         # 2 yards to first (goal)
    yard_line=98,       # Opp 2 yard line
    home_score=14,      # Down by 6
    away_score=20,
    away_timeouts=0,    # Defense has 0 timeouts
    home_timeouts=1,    # Offense has 1 timeout
)
play_context = goal_line_stand.into_play_context()
playcall = model.sim(play_context, coach)
print("Late game goal line stand scenario:")
print(str(play_context))
print()
print(f"Playcall: {str(playcall)}")
print()

# A post-turnover scenario
after_turnover = GameContext(
    home_team="NYM",
    away_team="CAR",
    quarter=3,          # Third quarter
    half_seconds=561,   # 9:21 clock
    down=1,             # First down
    distance=10,        # 10 yards to first
    yard_line=76,       # Opp 24 yard line
    home_score=14,      # Tied
    away_score=14,
    away_timeouts=3,    # Defense has 3 timeouts
    home_timeouts=3,    # Offense has 3 timeouts
)
play_context = after_turnover.into_play_context()
playcall = model.sim(play_context, coach)
print("Post-turnover scenario:")
print(str(play_context))
print()
print(f"Playcall: {str(playcall)}")
print()

# A late-game "ice the game" scenario
ice_the_game = GameContext(
    home_team="NYM",
    away_team="CAR",
    quarter=4,          # Fourth quarter
    half_seconds=69,    # 1:09 clock
    down=3,             # Third down
    distance=2,         # 2 yards to first
    yard_line=62,       # Opp 38 yard line
    home_score=17,      # Winning by 4
    away_score=13,
    away_timeouts=2,    # Defense has 2 timeouts
    home_timeouts=1,    # Offense has 1 timeout
)
play_context = ice_the_game.into_play_context()
playcall = model.sim(play_context, coach)
print("Ice the game scenario:")
print(str(play_context))
print()
print(f"Playcall: {str(playcall)}")
print()

# An end-of-half hail mary scenario
hail_mary_setup = GameContext(
    home_team="NYM",
    away_team="CAR",
    quarter=2,          # Second quarter
    half_seconds=3,     # 0:07 clock
    down=1,             # First down
    distance=10,        # 10 yards to first
    yard_line=54,       # Opp 46 yard line
    home_score=7,       # Tied
    away_score=7,
    away_timeouts=1,    # Defense has 1 timeout
    home_timeouts=0,    # Offense has 0 timeouts
)
play_context = hail_mary_setup.into_play_context()
playcall = model.sim(play_context, coach)
print("End-of-half hail mary scenario:")
print(str(play_context))
print()
print(f"Playcall: {str(playcall)}")
print()

# A potential "go for it on fourth" scenario
fourth_down_decision = GameContext(
    home_team="NYM",
    away_team="CAR",
    quarter=3,          # Third quarter
    half_seconds=355,   # 5:55 clock
    down=4,             # Fourth down
    distance=1,         # 1 yard to first
    yard_line=55,       # Opp 45 yard line
    home_score=20,      # Winning by 10
    away_score=10,
    away_timeouts=3,    # Defense has 3 timeouts
    home_timeouts=3,    # Offense has 3 timeouts
)
play_context = fourth_down_decision.into_play_context()
playcall = model.sim(play_context, coach)
print("Go for it on fourth scenario:")
print(str(play_context))
print()
print(f"Playcall: {str(playcall)}")
print()
