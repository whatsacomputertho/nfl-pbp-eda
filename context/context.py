class GameContext:
    """
    A play calling scenario. Includes the quarter, clock, down & distance,
    yard line, point differential, and timeouts remaining for each team.
    """
    def validate_static(
            quarter: int=1,
            half_seconds: int=1800,
            down: int=0,
            distance: int=10,
            yard_line: int=25,
            score_diff: int=0,
            off_timeouts: int=3,
            def_timeouts: int=3
        ) -> tuple[bool, str]:
        """
        Validate game context parameters

        Args:
            quarter (int): The current quarter
            half_seconds (int): The seconds remaining in the half
            down (int): The current down
            distance (int): The yards remaining until first down
            yard_line (int): The current yard line (0-100) where > 50 is opp
            score_diff (int): The score differential with respect to offense
            off_timeouts (int): number of timeouts remaining for the offense
            def_timeouts (int): Number of timeouts remaining for the defense
        
        Returns:
            bool: Whether the game context is valid
            str: Error message if invalid
        """
        # Ensure the quarter is between 1-4
        if quarter > 4 or quarter < 1:
            return False, f"Quarter must be between 1-4, got: {quarter}"

        # Ensure the half-seconds is between 0-1800
        if half_seconds > 1800 or half_seconds < 0:
            return (
                False,
                f"Half seconds must be between 0-1800, got: {half_seconds}"
            )

        # Ensure the down is between 0-4
        if down > 4 or down < 0:
            return False, f"Down must be between 0-4, got: {down}"
        
        # Ensure the distance is between 0-100
        if distance > 100 or distance < 0:
            return (
                False,
                f"Distance must be between 0-100, got: {distance}"
            )

        # Ensure the yard line is between 0-100
        if yard_line > 100 or yard_line < 0:
            return (
                False,
                f"Yard line must be between 0-100, got: {yard_line}"
            )

        # Ensure the offense timeouts is between 0-3
        if off_timeouts > 3 or off_timeouts < 0:
            return (
                False,
                f"Offense timeouts must be between 0-3, got: {off_timeouts}"
            )

        # Ensure the defense timeouts is between 0-3
        if def_timeouts > 3 or def_timeouts < 0:
            return (
                False,
                f"Defense timeouts must be between 0-3, got: {def_timeouts}"
            )
        return True, ""

    def __init__(
            self,
            quarter: int=1,
            half_seconds: int=1800,
            down: int=0,
            distance: int=10,
            yard_line: int=25,
            score_diff: int=0,
            off_timeouts: int=3,
            def_timeouts: int=3
        ) -> "GameContext":
        """
        Constructor for the GameContext class

        Args:
            quarter (int): The current quarter
            half_seconds (int): The seconds remaining in the half
            down (int): The current down
            distance (int): The yards remaining until first down
            yard_line (int): The current yard line (0-100) where > 50 is opp
            score_diff (int): The score differential with respect to offense
            off_timeouts (int): number of timeouts remaining for the offense
            def_timeouts (int): Number of timeouts remaining for the defense
        
        Returns:
            GameContext: The initialized GameContext
        """
        # Validate the game context params
        valid, err = GameContext.validate_static(
            quarter,
            half_seconds,
            down,
            distance,
            yard_line,
            score_diff,
            off_timeouts,
            def_timeouts
        )
        if not valid:
            raise ValueError(f"Invalid game context: {err}")

        # If valid, save the game context params
        self.quarter = quarter
        self.half_seconds = half_seconds
        self.down = down
        self.distance = distance
        self.yard_line = yard_line
        self.score_diff = score_diff
        self.off_timeouts = off_timeouts
        self.def_timeouts = def_timeouts

    def __str__(self) -> str:
        """
        Format a GameContext as a human-readable string

        Returns:
            str: The GameContext as a human-readable string
        """
        # Format the clock
        clock = self.half_seconds if self.half_seconds <= 900 \
                else self.half_seconds - 900
        clock_mins = clock // 60
        clock_secs = clock - (clock_mins * 60)
        clock_secs_str = f"0{clock_secs}" if clock_secs < 10 \
                else str(clock_secs)
        clock_str = f"{clock_mins}:{clock_secs_str}"
        
        # Format the quarter
        quarter_suf = "th"
        if self.quarter == 1:
            quarter_suf = "st"
        elif self.quarter == 2:
            quarter_suf = "nd"
        elif self.quarter == 3:
            quarter_suf = "rd"
        quarter_str = f"{self.quarter}{quarter_suf} quarter"

        # Format the down & distance
        down_suf = "th"
        if self.down == 1:
            down_suf = "st"
        elif self.down == 2:
            down_suf = "nd"
        elif self.down == 3:
            down_suf = "rd"
        down_dist_str = f"{self.down}{down_suf} & {self.distance}"

        # Format the yard line
        yard = self.yard_line if self.yard_line < 50 \
            else 100 - self.yard_line
        side_of_field = "own" if self.yard_line < 50 else "opp"
        yard_str = f"from {side_of_field} {yard} yard line"

        # Format the point differential
        score_str = "tie game"
        if self.score_diff > 0:
            score_str = f"up by {abs(self.score_diff)}"
        elif self.score_diff < 0:
            score_str = f"down by {abs(self.score_diff)}"
        
        # Format and return the game context string
        return f"""{quarter_str} {clock_str}
{down_dist_str} {yard_str}
{score_str}
offense has {self.off_timeouts} timeouts
defense has {self.def_timeouts} timeouts
"""
