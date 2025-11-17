import copy
from context.context import GameContext

class RushResult:
    def __init__(
            self,
            yards_gained: int,
            play_duration: int,
            fumble: bool=False,
            return_yards: int=0,
            touchdown: bool=False,
            scramble: bool=False
        ):
        """
        Constructor for the RushResult class
        """
        self.yards_gained = yards_gained
        self.play_duration = play_duration
        self.fumble = fumble
        self.return_yards = return_yards
        self.touchdown = touchdown
        self.scramble = scramble

    def next_context(self, context: GameContext) -> GameContext:
        """
        Converts the current game context into the next game context given
        this play result
        """
        new_context = copy.deepcopy(context)
        new_context.update_clock(self.play_duration)
        new_context.update_yard_line(self.yards_gained)
        if self.fumble:
            new_context.home_possession = not new_context.home_possession
            new_context.yard_line = 100 - new_context.yard_line
        return new_context

    def __str__(self) -> str:
        """
        Formats a RushResult as a string

        Returns:
            str: The result as a human-readable string
        """
        res = f"({self.play_duration}s)"
        if self.scramble:
            res += " QB under pressure, scrambles."
        res = f" Rush for {self.yards_gained} yards."
        if self.fumble:
            res += " FUMBLE recovered by the defense."
            if self.return_yards != 0:
                res += f" Fumble was returned {self.return_yards} yards."
        if self.touchdown:
            res += " TOUCHDOWN!"
        return res
