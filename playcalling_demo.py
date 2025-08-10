import os
from context.context import GameContext
from playcalling.model import PlayCallingModel

WORKDIR = os.path.dirname(os.path.abspath(__file__))

# Load the playcalling model
print("Loading the playcalling model")
model = PlayCallingModel(from_file=True)

# Test the model under a few playcalling scenarios
print("Demoing the playcalling model")

# A first playcall after a kickoff
after_kickoff = GameContext(
    2,      # Second quarter
    900,    # 15:00 clock
    1,      # First down
    10,     # 10 yards to first
    35,     # Own 35 yard line
    0,      # Tied
    3,      # Defense has 3 timeouts
    3,      # Offense has 3 timeouts
)
playcall = model.play(after_kickoff)
print("Scenario:")
print(str(after_kickoff))
print()
print(f"Playcall: {str(playcall)}")
print()

# A two-minute drill scenario
two_minute_drill = GameContext(
    2,      # Second quarter
    102,    # 1:02 clock
    2,      # Second down
    6,      # 6 yards to first
    39,     # Own 39 yard line
    -3,     # Down by 3
    2,      # Defense has 2 timeouts
    2,      # Offense has 2 timeouts
)
playcall = model.play(two_minute_drill)
print("Scenario:")
print(str(two_minute_drill))
print()
print(f"Playcall: {str(playcall)}")
print()

# A late-game goal line stand scenario
goal_line_stand = GameContext(
    4,      # Fourth quarter
    18,     # 0:18 clock
    4,      # Fourth down
    2,      # 2 yards to first (goal)
    98,     # Opp 2 yard line
    -6,     # Down by 6
    0,      # Defense has 0 timeouts
    1,      # Offense has 1 timeout
)
playcall = model.play(goal_line_stand)
print("Scenario:")
print(str(goal_line_stand))
print()
print(f"Playcall: {str(playcall)}")
print()

# A post-turnover scenario
after_turnover = GameContext(
    3,      # Third quarter
    561,    # 9:21 clock
    1,      # First down
    10,     # 10 yards to first
    76,     # Opp 24 yard line
    0,      # Tied
    3,      # Defense has 3 timeouts
    3,      # Offense has 3 timeouts
)
playcall = model.play(after_turnover)
print("Scenario:")
print(str(after_turnover))
print()
print(f"Playcall: {str(playcall)}")
print()

# A late-game "ice the game" scenario
ice_the_game = GameContext(
    4,      # Fourth quarter
    69,     # 1:09 clock
    3,      # Third down
    2,      # 2 yards to first
    62,     # Opp 38 yard line
    4,      # Winning by 4
    2,      # Defense has 2 timeouts
    1,      # Offense has 1 timeout
)
playcall = model.play(ice_the_game)
print("Scenario:")
print(str(ice_the_game))
print()
print(f"Playcall: {str(playcall)}")
print()

# An end-of-half hail mary scenario
hail_mary_setup = GameContext(
    2,      # Second quarter
    7,      # 0:07 clock
    1,      # First down
    10,     # 10 yards to first
    54,     # Opp 46 yard line
    0,      # Tied
    1,      # Defense has 1 timeout
    0,      # Offense has 0 timeouts
)
playcall = model.play(hail_mary_setup)
print("Scenario:")
print(str(hail_mary_setup))
print()
print(f"Playcall: {str(playcall)}")
print()

# A potential "go for it on fourth" scenario
fourth_down_decision = GameContext(
    3,      # Third quarter
    355,    # 5:55 clock
    4,      # Fourth down
    1,      # 1 yard to first
    55,     # Opp 45 yard line
    10,     # Winning by 10
    3,      # Defense has 3 timeouts
    3,      # Offense has 3 timeouts
)
playcall = model.play(fourth_down_decision)
print("Scenario:")
print(str(fourth_down_decision))
print()
print(f"Playcall: {str(playcall)}")
print()
