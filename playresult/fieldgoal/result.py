import copy
import numpy as np
from enum import Enum
from context.context import GameContext

class FieldGoalResult:
    def __init__(
            self,
            field_goal_made: bool=True,
            field_goal_blocked: bool=False,
            field_goal_block_return_yards: int=0,
            field_goal_distance: int=25,
            play_duration: int=4
        ):
        self.field_goal_made = field_goal_made
        self.field_goal_blocked = field_goal_blocked
        self.field_goal_block_return_yards = field_goal_block_return_yards
        self.field_goal_distance = field_goal_distance
        self.play_duration = play_duration

    def next_context(self, context: GameContext) -> GameContext:
        """
        Converts the current game context into the next game context given
        this play result
        """
        new_context = copy.deepcopy(context)
        new_context.update_clock(self.play_duration)

        # Update the yard line and score on blocked field goals
        if self.field_goal_blocked or not self.field_goal_made:
            if context.home_possession ^ context.home_positive_direction:
                new_context.yard_line = context.yard_line + self.field_goal_block_return_yards
                new_context.distance = min(10, 100 - context.yard_line)
            else:
                new_context.yard_line = context.yard_line - self.field_goal_block_return_yards
                new_context.distance = min(10, context.yard_line)
            new_context.down = 1
            new_context.home_possession = not context.home_possession

        # Update the yard line and score on made field goals
        if self.field_goal_made:
            if context.home_possession ^ context.home_positive_direction:
                new_context.yard_line = 65
            else:
                new_context.yard_line = 35
            if context.home_possession:
                if context.next_play_extra_point:
                    new_context.home_score += 1
                else:
                    new_context.home_score += 3
            else:
                if context.next_play_extra_point:
                    new_context.away_score += 1
                else:
                    new_context.away_score += 3
            new_context.down = 0
            new_context.distance = 10
            new_context.next_play_kickoff = True
            return new_context

        # Check for TDs / touchbacks
        if (context.home_possession and not context.home_positive_direction) or \
            (not context.home_possession and context.home_positive_direction):
            # Yard line greater than 100 is a TD
            if new_context.yard_line > 100:
                if new_context.home_possession:
                    if context.next_play_extra_point:
                        new_context.home_score += 2
                    else:
                        new_context.home_score += 6
                else:
                    if context.next_play_extra_point:
                        new_context.away_score += 2
                    else:
                        new_context.away_score += 6
                new_context.down = 0
                new_context.yard_line = 98
                if not context.next_play_extra_point:
                    new_context.next_play_extra_point = True
            elif new_context.yard_line < 0:
                new_context.yard_line = 25
        else:
            # Yard line less than 0 is a TD
            if new_context.yard_line < 0:
                if new_context.home_possession:
                    if context.next_play_extra_point:
                        new_context.home_score += 2
                    else:
                        new_context.home_score += 6
                else:
                    if context.next_play_extra_point:
                        new_context.away_score += 2
                    else:
                        new_context.away_score += 6
                new_context.down = 0
                new_context.yard_line = 2
                if not context.next_play_extra_point:
                    new_context.next_play_extra_point = True
            elif new_context.yard_line < 0:
                new_context.yard_line = 75
        if context.next_play_extra_point:
            new_context.next_play_extra_point = False
            new_context.next_play_kickoff = True
            if context.home_possession ^ context.home_positive_direction:
                new_context.yard_line = 65
            else:
                new_context.yard_line = 35
        return new_context

    def __str__(self):
        res = f"({self.play_duration}s) {self.field_goal_distance} yard field goal"
        if self.field_goal_blocked:
            res += f" BLOCKED. Returned {self.field_goal_block_return_yards} yards"
            return res
        if self.field_goal_made:
            res += " made."
        else:
            res += " MISSED."
        return res
