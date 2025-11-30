import random
import math
import numpy as np
import pandas as pd
from context.context import PlayContext
from playcalling.playcall import PlayCall
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from team.coach import CoachSkill

class PlayCallingModel:
    """
    Generates a playcall given a play context and coach playcalling tendency
    """
    def __init__(self):
        """
        Constructor for the PlayCallingModel class
        
        Returns:
            PlayCallingModel: The constructed PlayCallingModel class
        """
        # Run probabilities on 1st-3rd down clock management scenarios
        self.p_run_clock_management = 0.15
        self.p_run_clock_management_no_timeouts = 0.01

        # Run probability regression on 1st down
        self.p_run_first_down_intr = 0.41649529080915104
        self.p_run_first_down_coef = 0.2035597

        # Run probability regression on 2nd down
        self.p_run_second_down_intr = 0.3250691394699521
        self.p_run_second_down_coef = 0.19162143

        # Run probability regression on 3rd down
        self.p_run_third_down_intr = 0.1340492470213823
        self.p_run_third_down_coef = 0.22902729

        # Run probability regression by distance to first / goal
        self.p_run_dist_intr = 0.30634251685198927
        self.p_run_dist_coef = -0.00318081

        # Field goal probability regression on 4th down by risk taking
        self.p_field_goal_risk_intr = 0.7886141537295228
        self.p_field_goal_risk_coef = -0.26532936

        # Field goal probability regression on 4th down by yard line
        self.p_field_goal_yard_line_intr = 0.24354785898372522
        self.p_field_goal_yard_line_coef_1 = 0.05165115
        self.p_field_goal_yard_line_coef_2 = -0.00112775

        # Go for it probability by risk taking
        # (when < 4 to go and between the 40 yard lines)
        self.p_go_for_it_intr = 0.19565011246401598
        self.p_go_for_it_coef = 0.51602604

        # Run probability regression on 4th down
        self.p_run_fourth_down_intr = 0.040592196833718536
        self.p_run_fourth_down_coef = 0.05793641

    def sim(self, context: PlayContext, coach: CoachSkill) -> PlayCall:
        """
        Generates a play call

        Args:
            context (PlayContext): The current play context
            coach (CoachSkill): The coach skill levels
        
        Returns:
            PlayCall: The generated play call
        """
        if context.down == 4:
            if self.is_must_score_scenario(context):
                return self.last_play_playcall(context)
            return self.fourth_down_playcall(context, coach.risk_taking, coach.run_pass)
        if self.is_clock_management_situation(context):
            if self.is_last_play(context):
                return self.last_play_playcall(context)
            return self.clock_management_playcall(context)
        return self.normal_play_call(context, coach.run_pass)

    def is_clock_management_situation(self, context: PlayContext) -> bool:
        """
        Determines whether the current context is a clock management situation

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether the current context is a clock management situation
        """
        return (context.quarter >= 4) and (context.half_seconds <= 180) \
            and (context.score_diff < 0) and (context.score_diff >= -17)

    def last_play_need_td(self, context: PlayContext) -> bool:
        """
        Determines whether the offense needs a touchdown in a clock management
        situation

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether the offense is aiming for a touchdown
        """
        if context.score_diff > -4:
            return False
        if context.score_diff > -8:
            return True
        return random.random() < 0.2

    def is_last_play(self, context: PlayContext) -> bool:
        """
        Determines whether this is the last play of the game

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether this is the last play of the game
        """
        return context.half_seconds < 5

    def p_field_goal_yardline(self, yard_line: int) -> float:
        """
        Generates the probability a field goal is called by yard line

        Args:
            yard_line (int): The current yard line
        
        Returns:
            float: Probability a field goal is called
        """
        return max(
            self.p_field_goal_yard_line_intr + \
                (self.p_field_goal_yard_line_coef_1 * yard_line) + \
                (self.p_field_goal_yard_line_coef_2 * pow(yard_line, 2)),
            0
        )

    def last_play_playcall(self, context: PlayContext) -> PlayCall:
        """
        Generates the playcall for the last play of the game

        Args:
            need_td (bool): Whether the offense is aiming for a touchdown
        
        Returns:
            PlayCall: The playcall for the final play
        """
        if self.last_play_need_td(context):
            return PlayCall.PASS
        if random.random() < self.p_field_goal_yardline(context.yard_line):
            return PlayCall.FIELD_GOAL
        return PlayCall.PASS

    def clock_management_playcall(self, context: PlayContext) -> PlayCall:
        """
        Generates the playcall for a clock management scenario on 1st-3rd down

        Returns:
            PlayCall: The playcall for the clock management scenario
        """
        if context.off_timeouts > 0:
            p_run = self.p_run_clock_management
        else:
            p_run = self.p_run_clock_management_no_timeouts
        if random.random() < p_run:
            return PlayCall.RUN
        return PlayCall.PASS

    def can_kneel(self, context: PlayContext) -> bool:
        """
        Determines if the offense can kneel to end the game

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether the offense can kneel to end the game
        """
        downs_remaining = 4 - context.down
        runoff_seconds = 42 * max(downs_remaining - context.def_timeouts, 0)
        return runoff_seconds >= context.half_seconds

    def normal_play_call(self, context: PlayContext, run_pass: float) -> PlayCall:
        """
        Generates the play call for a non-clock management scenario on 1st-3rd
        or for a go-for-it scenario on 4th

        Args:
            context (PlayContext): The current play context
            run_pass (float): The coach's run/pass playcall tendency
        
        Returns:
            PlayCall: The play call
        """
        if context.down == 1:
            p_run_call = self.p_run_first_down_intr + (self.p_run_first_down_coef * run_pass)
        elif context.down == 2:
            p_run_call = self.p_run_second_down_intr + (self.p_run_second_down_coef * run_pass)
        elif context.down == 3:
            p_run_call = self.p_run_third_down_intr + (self.p_run_third_down_coef * run_pass)
        else:
            p_run_call = self.p_run_fourth_down_intr + (self.p_run_fourth_down_coef * run_pass)
        p_run_dist = self.p_run_dist_intr + (self.p_run_dist_coef * context.distance)
        p_run = (p_run_dist * 0.3) + (p_run_call * 0.7)
        if random.random() < p_run:
            return PlayCall.RUN
        return PlayCall.PASS

    def is_must_score_scenario(self, context: PlayContext) -> bool:
        """
        Generates whether this is a must-score scenario on 4th down

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether this is a must score scenario
        """
        if context.score_diff >= 0:
            return False
        scores_needed = int(abs(round(context.score_diff / 8)))
        timeout_drive_time = (42 * (3 - context.off_timeouts)) + 8
        non_timeout_drive_time = (42 * 3) + 8
        if context.half_seconds <= timeout_drive_time:
            return True
        timeout_drives_remaining = 1
        non_timeout_drives_remaining = math.ceil(
            (context.half_seconds - timeout_drive_time) / non_timeout_drive_time
        )
        if (timeout_drives_remaining + non_timeout_drives_remaining) <= scores_needed:
            return True
        return False

    def is_go_for_it_scenario(self, context: PlayContext) -> bool:
        """
        Determines whether this is a go-for-it on 4th scenario

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether this is a go-for-it scenario
        """
        return (context.yard_line >= 40 and context.yard_line <= 60 and context.distance <=4) or \
            (context.yard_line >= 80 and context.distance <= 4)

    def fourth_down_go_for_it_playcall(self, run_pass: float, yard_line: int) -> PlayCall:
        """
        Generates the play call on fourth down if going for it

        Args:
            run_pass (float): The coach's run-pass playcalling tendency
            yard_line (int): The current yard line
        
        Returns:
            PlayCall: The play call on 4th
        """
        p_run_call = self.p_run_fourth_down_intr + (self.p_run_fourth_down_coef * run_pass)
        p_run_dist = self.p_run_dist_intr + (self.p_run_dist_coef * yard_line)
        p_run = (p_run_dist * 0.3) + (p_run_call * 0.7)
        if random.random() < p_run:
            return PlayCall.RUN
        return PlayCall.PASS

    def in_field_goal_range(self, context: PlayContext) -> bool:
        """
        Determines whether a field goal can be kicked

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether a field goal can be kicked
        """
        return context.yard_line >= 50

    def fourth_down_playcall(
            self,
            context: PlayContext,
            risk_taking: float,
            run_pass: float
        ) -> PlayCall:
        """
        Generates the fourth down play call based on risk-taking and run-pass
        tendency

        Args:
            context (PlayContext): The current play context
            risk_taking (float): The coach's risk-taking tendency
            run_pass (float): The coach's run-pass playcall tendency
        
        Returns:
            PlayCall: The fourth down playcall
        """
        in_field_goal_range = self.in_field_goal_range(context)
        is_go_for_it_scenario = self.is_go_for_it_scenario(context)
        if not (in_field_goal_range or is_go_for_it_scenario):
            return PlayCall.PUNT
        if not in_field_goal_range and is_go_for_it_scenario:
            p_go_for_it = self.p_go_for_it_intr + (self.p_go_for_it_coef * risk_taking)
            if random.random() < p_go_for_it:
                return self.fourth_down_go_for_it_playcall(run_pass, context.yard_line)
            return PlayCall.PUNT
        if in_field_goal_range and is_go_for_it_scenario:
            p_field_goal_risk = self.p_field_goal_risk_intr + (self.p_field_goal_risk_coef * risk_taking)
            p_field_goal_dist = self.p_field_goal_yard_line_intr + \
                (self.p_field_goal_yard_line_coef_1 * context.yard_line) + \
                (self.p_field_goal_yard_line_coef_2 * pow(context.yard_line, 2))
            p_field_goal = (0.4 * p_field_goal_risk) + (0.6 * p_field_goal_dist)
            if random.random() < p_field_goal:
                return PlayCall.FIELD_GOAL
            if context.yard_line >= 80:
                return self.fourth_down_go_for_it_playcall(run_pass, context.yard_line)
            p_go_for_it = self.p_go_for_it_intr + (self.p_go_for_it_coef * risk_taking)
            if random.random() < p_go_for_it:
                return self.fourth_down_go_for_it_playcall(run_pass, context.yard_line)
        return PlayCall.PUNT
