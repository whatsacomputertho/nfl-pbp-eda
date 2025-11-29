import random

class PlayContext:
    """
    A play calling scenario. Includes the quarter, clock, down & distance,
    yard line, point differential, and timeouts remaining for each team.
    """
    def validate_static(
            quarter: int=1,
            half_seconds: int=1800,
            down: int=0,
            distance: int=10,
            yard_line: int=35,
            goal_to_go: bool=False,
            score_diff: int=0,
            off_timeouts: int=3,
            def_timeouts: int=3
        ) -> tuple[bool, str]:
        """
        Validate play context parameters

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
            bool: Whether the play context is valid
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
            goal_to_go: bool=False,
            score_diff: int=0,
            off_timeouts: int=3,
            def_timeouts: int=3
        ) -> "PlayContext":
        """
        Constructor for the PlayContext class

        Args:
            quarter (int): The current quarter
            half_seconds (int): The seconds remaining in the half
            down (int): The current down
            distance (int): The yards remaining until first down
            yard_line (int): The current yard line (0-100) where > 50 is opp
            goal_to_go (bool): Whether a first down is unachievable
            score_diff (int): The score differential with respect to offense
            off_timeouts (int): number of timeouts remaining for the offense
            def_timeouts (int): Number of timeouts remaining for the defense
        
        Returns:
            PlayContext: The initialized PlayContext
        """
        # Validate the play context params
        valid, err = PlayContext.validate_static(
            quarter,
            half_seconds,
            down,
            distance,
            yard_line,
            goal_to_go,
            score_diff,
            off_timeouts,
            def_timeouts
        )
        if not valid:
            raise ValueError(f"Invalid play context: {err}")

        # If valid, save the play context params
        self.quarter = quarter
        self.half_seconds = half_seconds
        self.down = down
        self.distance = distance
        self.yard_line = yard_line
        self.goal_to_go = goal_to_go
        self.score_diff = score_diff
        self.off_timeouts = off_timeouts
        self.def_timeouts = def_timeouts

    def result_prefix(self) -> str:
        """
        Format a PlayContext as a concise prefix for a play result string

        Returns:
            str: The PlayContext as a concise string prefix
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
        quarter_str = f"{self.quarter}Q"

        # Format the down & distance
        down_suf = "th"
        if self.down == 1:
            down_suf = "st"
        elif self.down == 2:
            down_suf = "nd"
        elif self.down == 3:
            down_suf = "rd"
        down_dist_str = f"{self.down}{down_suf} & {self.distance}"
        if self.goal_to_go:
            down_dist_str = f"{self.down}{down_suf} & goal"

        # Format the yard line
        yard = self.yard_line if self.yard_line < 50 \
            else 100 - self.yard_line
        side_of_field = "OWN" if self.yard_line < 50 else "OPP"
        yard_str = f"{side_of_field} {yard}"

        return f"[{clock_str} {quarter_str}] {down_dist_str} {yard_str}"

    def __str__(self) -> str:
        """
        Format a PlayContext as a human-readable string

        Returns:
            str: The PlayContext as a human-readable string
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
        if self.goal_to_go:
            down_dist_str = f"{self.down}{down_suf} & goal"

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
        
        # Format and return the play context string
        return f"""{quarter_str} {clock_str}
{down_dist_str} {yard_str}
{score_str}
offense has {self.off_timeouts} timeouts
defense has {self.def_timeouts} timeouts
"""

class GameContext:
    """
    The overarching game context. Includes individual team scores, possession
    information, etcetera.
    """
    def __init__(
            self,
            home_team: str,
            away_team: str,
            quarter: int=1,
            half_seconds: int=1800,
            down: int=0,
            distance: int=10,
            yard_line: int=25,
            home_score: int=0,
            away_score: int=0,
            home_positive_direction: bool=True,
            home_opening_kickoff: bool=True,
            home_possession: bool=True,
            home_timeouts: int=3,
            away_timeouts: int=3,
            next_play_extra_point: bool=False,
            next_play_kickoff: bool=False,
            game_over: bool=False
        ) -> "GameContext":
        """
        Constructor for the GameContext class

        Args:
            home_team (str): The home team acronym
            away_team (str): The away team acronym
            quarter (int): The current quarter
            half_seconds (int): The seconds remaining in the half
            down (int): The current down
            distance (int): The yards remaining until first down
            yard_line (int): The current yard line
            home_score (int): The home team score
            away_score (int): The away team score
            home_positive_direction (bool): Whether the home team drives 0-100
            home_opening_kickoff (bool): Whether the home team received the opening kick
            home_possession (bool): Whether the home team has possession
            home_timeouts (int): number of timeouts remaining for the home team
            away_timeouts (int): Number of timeouts remaining for the away team
            next_play_extra_point (bool): Whether the next play will be an extra point
            next_play_kickoff (bool): Whether the next play will be a kickoff
            game_over (bool): Whether the game is complete
        
        Returns:
            GameContext: The game context
        """
        self.home_team = home_team
        self.away_team = away_team
        self.quarter = quarter
        self.half_seconds = half_seconds
        self.down = down
        self.distance = distance
        self.yard_line = yard_line
        self.home_score = home_score
        self.away_score = away_score
        self.home_positive_direction = home_positive_direction
        self.home_opening_kickoff = home_opening_kickoff
        self.home_possession = home_possession
        self.home_timeouts = home_timeouts
        self.away_timeouts = away_timeouts
        self.next_play_extra_point = next_play_extra_point
        self.next_play_kickoff = next_play_kickoff
        self.game_over = game_over

    def into_play_context(self) -> PlayContext:
        """
        Converts a GameContext into a PlayContext

        Returns:
            PlayContext: The play context derived from teh game context
        """
        # Derive the yard line property
        if self.home_positive_direction:
            if self.home_possession:
                yard_line = self.yard_line
            else:
                yard_line = 100 - self.yard_line
        else:
            if self.home_possession:
                yard_line = 100 - self.yard_line
            else:
                yard_line = self.yard_line
        
        # Derive the goal to go property
        goal_to_go = False
        if (yard_line + self.distance) >= 100:
            goal_to_go = True
        
        # Derive the score diff property
        score_diff = self.home_score - self.away_score
        if not self.home_possession:
            score_diff = self.away_score - self.home_score
        
        # Derive the offense and defense timeouts property
        if self.home_possession:
            off_timeouts = self.home_timeouts
            def_timeouts = self.away_timeouts
        else:
            off_timeouts = self.away_timeouts
            def_timeouts = self.home_timeouts
        
        return PlayContext(
            quarter=self.quarter,
            half_seconds=self.half_seconds,
            down=self.down,
            distance=self.distance,
            yard_line=yard_line,
            goal_to_go=goal_to_go,
            score_diff=score_diff,
            off_timeouts=off_timeouts,
            def_timeouts=def_timeouts
        )

    def result_prefix(self) -> str:
        """
        Format a GameCnotext as a concise prefix for a play result string

        Returns:
            str: The GameCnotext as a concise string prefix
        """
        play_context = self.into_play_context()
        if self.home_possession:
            home_team_str = f"*{self.home_team}"
            away_team_str = self.away_team
        else:
            away_team_str = f"*{self.away_team}"
            home_team_str = self.home_team
        return f"{play_context.result_prefix()} ({home_team_str} {self.home_score} - {away_team_str} {self.away_score})"

    def update(self, play_duration: int, yards_gained: int):
        """
        Updates the clock and yard line

        Args:
            play_duration (int): How long the play took in seconds
            yards_gained (int): Yards gained on the play
        """
        self.update_clock(play_duration)
        self.update_yard_line(yards_gained)
        if self.half_seconds <= 0:
            if self.quarter == 2:
                if self.home_opening_kickoff:
                    self.home_possession = True
                    if self.home_positive_direction:
                        self.yard_line = 35
                    else:
                        self.yard_line = 65
                    self.next_play_kickoff = True
                    self.quarter = 3
                    self.half_seconds = 1800
            else:
                if self.home_score != self.away_score:
                    self.game_over = True

    def update_clock(self, play_duration: int):
        """
        Updates the clock given the duration of the play. Also updates the
        quarter if applicable.

        Args:
            play_duration (int): How long the play took in seconds
        """
        half_seconds = self.half_seconds - play_duration
        quarter = self.quarter

        # End of first or third quarter
        if self.half_seconds > 900 and half_seconds < 900:
            quarter += 1
            half_seconds = 900
        
        # End of half
        if half_seconds < 0:
            half_seconds = 0
            if quarter >= 4:
                if self.home_score != self.away_score:
                    # Game is over
                    self.game_over = True
                else:
                    # Overtime
                    half_seconds = 900
                    quarter = 5
                    
                    # Randomize possession
                    self.home_possession = random.randint(0, 1) == 1
        
        # Regular play
        self.half_seconds = half_seconds
        self.quarter = quarter

    def update_yard_line(
            self,
            yards_gained: int
        ):
        """
        Updates the yard line given the number of yards gained on the play
        as well as the down and distance

        Args:
            yards_gained (int): The number of yards gained on the play
        """
        # Update the yard line, check for a first down, touchdown, or safety
        yard_line = self.yard_line
        distance = self.distance
        first_down = False
        if self.home_positive_direction:
            distance -= yards_gained
            if self.home_possession:
                yard_line += yards_gained
                if not (self.yard_line + self.distance) >= 100:
                    if yard_line >= (self.yard_line + self.distance):
                        # First down
                        self.down = 1
                        self.distance = 10
                        self.yard_line = yard_line
                        first_down = True
                if yard_line > 100:
                    # Touchdown
                    self.down = 0
                    self.home_score += 6
                    self.yard_line = 98
                    self.next_play_extra_point = True
                    return
                elif yard_line < 0:
                    # Safety
                    self.down = 0
                    self.away_score += 2
                    self.yard_line = 35
                    self.next_play_kickoff = True
                    return
            else:
                yard_line -= yards_gained
                if not (self.yard_line - self.distance) <= 0:
                    if yard_line <= (self.yard_line - self.distance):
                        # First down
                        self.down = 1
                        self.distance = 10
                        self.yard_line = yard_line
                        first_down = True
                if yard_line < 0:
                    # Touchdown
                    self.down = 0
                    self.away_score += 6
                    self.yard_line = 2
                    self.next_play_extra_point = True
                    return
                elif yard_line > 100:
                    # Safety
                    self.down = 0
                    self.home_score += 2
                    self.yard_line = 65
                    self.next_play_kickoff = True
                    return
        else:
            distance -= yards_gained
            if self.home_possession:
                yard_line -= yards_gained
                if not (self.yard_line - self.distance) <= 0:
                    if yard_line <= (self.yard_line - self.distance):
                        # First down
                        self.down = 1
                        self.distance = 10
                        self.yard_line = yard_line
                        first_down = True
                if yard_line < 0:
                    # Touchdown
                    self.down = 0
                    self.home_score += 6
                    self.yard_line = 2
                    self.next_play_extra_point = True
                    return
                elif yard_line > 100:
                    # Safety
                    self.down = 0
                    self.away_score += 2
                    self.yard_line = 65
                    self.next_play_kickoff = True
                    return
            else:
                yard_line += yards_gained
                if not (self.yard_line + self.distance) >= 100:
                    if yard_line >= (self.yard_line + self.distance):
                        # First down
                        self.down = 1
                        self.distance = 10
                        self.yard_line = yard_line
                        first_down = True
                if yard_line > 100:
                    # Touchdown
                    self.down = 0
                    self.away_score += 6
                    self.yard_line = 98
                    self.next_play_extra_point = True
                    return
                elif yard_line < 0:
                    # Safety
                    self.down = 0
                    self.home_score += 2
                    self.yard_line = 35
                    self.next_play_kickoff = True
                    return

        # Update the yard line, down and distance
        if not first_down:
            self.yard_line = yard_line
            self.distance = distance
            self.down += 1
            if self.down > 4:
                # Turnover on downs
                self.down = 1
                self.home_possession = not self.home_possession
