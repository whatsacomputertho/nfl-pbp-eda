import copy
from context.context import GameContext

class PuntResult:
    def __init__(
            self,
            punt_yards: int,
            punt_return_yards: int,
            play_duration: int,
            blocked: bool=False,
            fumble_return_yards: int=0,
            touchback: bool=False,
            out_of_bounds: bool=False,
            fair_catch: bool=False,
            muffed: bool=False,
            fumble: bool=False,
            touchdown: bool=False
        ):
        """
        Constructor for the PuntResult class
        """
        self.blocked = blocked
        self.fumble_return_yards = fumble_return_yards
        self.touchback = touchback
        self.punt_yards = punt_yards
        self.out_of_bounds = out_of_bounds
        self.fair_catch = fair_catch
        self.muffed = muffed
        self.punt_return_yards = punt_return_yards
        self.fumble = fumble
        self.play_duration = play_duration
        self.touchdown = touchdown

    def next_context(self, context: GameContext) -> GameContext:
        """
        Converts the current game context into the next game context given
        this play result
        """
        new_context = copy.deepcopy(context)
        new_context.update_clock(self.play_duration)
        new_context.down = 1

        # Update the yard line on touchbacks
        if self.touchback:
            if context.home_possession ^ context.home_positive_direction:
                new_context.yard_line = 25
            else:
                new_context.yard_line = 75
        
        # Calculate the deltas for each yardage & possession
        change_of_possession = True
        delta_punt_yards = self.punt_yards
        delta_return_yards = self.punt_return_yards * -1
        delta_fumble_yards = self.fumble_return_yards
        if context.home_possession ^ context.home_positive_direction:
            delta_punt_yards = delta_punt_yards * -1
            delta_return_yards = self.punt_return_yards
            delta_fumble_yards = delta_fumble_yards * -1

        # Update the yard line on OOB, fair catches
        new_context.yard_line = context.yard_line + delta_punt_yards

        # Update the yard line on muffed punts
        if self.muffed:
            new_context.yard_line = new_context.yard_line + delta_fumble_yards
            change_of_possession = False
        
        # Update the yard line on handled punts
        else:
            new_context.yard_line = new_context.yard_line + delta_return_yards

            # Update the yard line on handled but fumbled punts
            if self.fumble:
                new_context.yard_line = new_context.yard_line + delta_fumble_yards
                change_of_possession = False
        
        # Update the possession
        if change_of_possession:
            new_context.home_possession = not new_context.home_possession
        
        # Check for TDs / touchbacks
        if (context.home_possession and (not context.home_positive_direction and change_of_possession)) or \
            (not context.home_possession and (context.home_positive_direction or change_of_possession)):
            # Yard line greater than 100 is a TD
            if new_context.yard_line > 100:
                if new_context.home_possession:
                    new_context.home_score += 6
                else:
                    new_context.away_score += 6
                new_context.down = 0
                new_context.yard_line = 98
                new_context.next_play_extra_point = True
            elif new_context.yard_line < 0:
                new_context.yard_line = 25
        else:
            # Yard line less than 0 is a TD
            if new_context.yard_line < 0:
                if new_context.home_possession:
                    new_context.home_score += 6
                else:
                    new_context.away_score += 6
                new_context.down = 0
                new_context.yard_line = 2
                new_context.next_play_extra_point = True
            elif new_context.yard_line > 100:
                new_context.yard_line = 75

        # Check for end of quarter
        if new_context.half_seconds <= 0:
            if new_context.quarter == 2:
                if new_context.home_opening_kickoff:
                    new_context.home_possession = True
                    if new_context.home_positive_direction:
                        new_context.yard_line = 35
                    else:
                        new_context.yard_line = 65
                    new_context.next_play_kickoff = True
                    new_context.quarter = 3
                    new_context.half_seconds = 1800
            else:
                if new_context.home_score != new_context.away_score:
                    new_context.game_over = True
        return new_context

    def __str__(self) -> str:
        """
        Formats a PuntResult as a string

        Returns:
            str: The result as a human-readable string
        """
        res = f"({self.play_duration}s)"
        if self.blocked:
            res += f" Punt BLOCKED, returned {self.fumble_return_yards} yards."
            if self.touchdown:
                res += f" TOUCHDOWN!"
            return res
        res += f" Punt {self.punt_yards} yards"
        if self.touchback:
            res += f" for a touchback."
        if self.out_of_bounds:
            res += f" out of bounds."
        if self.fair_catch:
            res += f", fair catch"
        if self.muffed:
            res += f" MUFFED, recovered by the defense."
        elif not (self.touchback or self.out_of_bounds or self.fair_catch):
            res += " fielded."
            res += f" Punt returned {self.punt_return_yards} yards."
        if self.fumble:
            res += f" FUMBLED, returned {self.fumble_return_yards} yards."
        if self.touchdown:
            res += f" TOUCHDOWN!"
        return res
