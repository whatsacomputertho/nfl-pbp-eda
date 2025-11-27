import copy
from context.context import GameContext

class KickoffResult:
    def __init__(
            self,
            kickoff_yards: int,
            kick_return_yards: int,
            play_duration: int,
            fumble_return_yards: int=0,
            touchback: bool=False,
            out_of_bounds: bool=False,
            fair_catch: bool=False,
            fumble: bool=False,
            touchdown: bool=False
        ):
        """
        Constructor for the KickoffResult class
        """
        self.kickoff_yards = kickoff_yards
        self.kick_return_yards = kick_return_yards
        self.play_duration = play_duration
        self.fumble_return_yards = fumble_return_yards
        self.touchback = touchback
        self.out_of_bounds = out_of_bounds
        self.fair_catch = fair_catch
        self.fumble = fumble
        self.touchdown = touchdown

    def __str__(self):
        """
        Formats a KickoffResult as a string

        Returns:
            str: The result as a human-readable string
        """
        res = f"({self.play_duration}s)"
        res += f" Kickoff {self.kickoff_yards} yards"
        if self.touchback:
            res += f" for a touchback."
        if self.out_of_bounds:
            res += f" out of bounds."
        if self.fair_catch:
            res += f", fair catch."
        elif not (self.touchback or self.out_of_bounds or self.fair_catch):
            res += " fielded."
            res += f" Kick returned {self.kick_return_yards} yards."
        if self.fumble:
            res += f" FUMBLED, returned {self.fumble_return_yards} yards."
        if self.touchdown:
            res += f" TOUCHDOWN!"
        return res
