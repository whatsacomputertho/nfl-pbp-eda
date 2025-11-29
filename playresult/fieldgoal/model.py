import numpy as np
import os
import random
from context.context import PlayContext
from playresult.fieldgoal.result import FieldGoalResult
from scipy.stats import skewnorm
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill

WORKDIR = os.path.dirname(os.path.abspath(__file__))

class FieldGoalResultModel:
    def __init__(self):
        """
        Constructor for the FieldGoalResultModel class

        Args:
            from_file (bool): Whether to load from a pre-trained file
        
        Returns:
            FieldGoalResultModel: The constructed FieldGoalResultModel class
        """
        # Field goal blocked skill regression
        self.p_blocked_skill_coef = 0.01919733
        self.p_blocked_skill_intr = 0.013200206956159479

        # Field goal blocked yard-line regression
        self.p_blocked_yard_line_coef = 0.05875677
        self.p_blocked_yard_line_intr = -5.320426815163247

        # Field goal made skill regression
        self.field_goal_made_skill_coef = 0.57103524
        self.field_goal_made_skill_intr = 0.44298810053776055

        # Field goal made yard-line regression
        self.field_goal_made_yard_line_coef_1 = 0.00399668
        self.field_goal_made_yard_line_coef_2 = -0.00035704
        self.field_goal_made_yard_line_intr = 0.9580405463949037

        # Field goal blocked duration regression
        self.field_goal_blocked_duration_mean = 6.843750
        self.field_goal_blocked_duration_std = 3.385612
        self.field_goal_blocked_duration_skew = 1.541247

        # Field goal not blocked duration regression
        self.field_goal_not_blocked_duration_mean = 4.054470
        self.field_goal_not_blocked_duration_std = 1.001211
        self.field_goal_not_blocked_duration_skew = -0.440028

    def sim(
            self,
            context: PlayContext,
            offense: OffensiveSkill,
            defense: DefensiveSkill,
            is_extra_point: bool=False
        ) -> FieldGoalResult:
        """
        Simulates a field goal play

        Args:
            context (PlayContext): The current play context
            offense (OffensiveSkill): The offense's skill levels
            defense (DefensiveSkill): The defense's skill levels
        
        Returns:
            FieldGoalResult: The result of a field goal play
        """
        yard_line = 100 - context.yard_line
        norm_diff_blocking_blitzing = 0.5 + ((defense.blitzing - offense.blocking) / 2)
        field_goal_distance = yard_line + 10
        if self.is_field_goal_blocked(norm_diff_blocking_blitzing, yard_line):
            return_yards = self.field_goal_block_return_yards()
            return FieldGoalResult(
                field_goal_made=False,
                field_goal_blocked=True,
                field_goal_block_return_yards=return_yards,
                field_goal_distance=field_goal_distance,
                play_duration=0 if is_extra_point else self.field_goal_duration(is_blocked=True)
            )
        play_duration = 0 if is_extra_point else self.field_goal_duration(is_blocked=False)
        if self.is_field_goal_made(offense.field_goals):
            return FieldGoalResult(
                field_goal_made=True,
                field_goal_blocked=False,
                field_goal_distance=field_goal_distance,
                play_duration=play_duration
            )
        return FieldGoalResult(
            field_goal_made=False,
            field_goal_blocked=False,
            field_goal_distance=field_goal_distance,
            play_duration=play_duration
        )

    def is_field_goal_blocked(self, norm_diff_blocking_blitzing: float, yard_line: int) -> bool:
        """
        Generates whether a field goal is blocked

        Args:
            norm_diff_blocking_blitzing (float): Blocking & blitzing skill differential
            yard_line (int): The current yard line
        
        Returns:
            bool: Whether the field goal was blocked
        """
        p_blocked_skill = self.p_blocked_skill_intr + (self.p_blocked_skill_coef * norm_diff_blocking_blitzing)
        p_blocked_yardline = np.exp(
            self.p_blocked_yard_line_intr + (self.p_blocked_yard_line_coef * yard_line)
        )
        p_blocked = ((p_blocked_skill * 0.7) + (p_blocked_yardline * 0.3)) * 0.7
        return random.random() < p_blocked

    def field_goal_block_return_yards(self) -> int:
        """
        Generates the return yards for a blocked field goal

        Returns:
            int: The blocked field goal return yards
        """
        return int(np.random.exponential(scale=1))

    def is_field_goal_made(self, norm_kicking: float) -> bool:
        """
        Generates whether a field goal is made

        Args:
            norm_kicking (float): Kicking skill level
        
        Returns:
            bool: Whether the field goal was made
        """
        p_made_skill = self.field_goal_made_skill_intr + (self.field_goal_made_skill_coef * norm_kicking)
        p_made_yardline = self.field_goal_made_yard_line_intr + \
            (self.field_goal_made_yard_line_coef_1 * norm_kicking) + \
            (self.field_goal_made_yard_line_coef_2 * pow(norm_kicking, 2))
        p_made = ((p_made_skill * 0.4) + (p_made_yardline * 0.6)) * 1.18
        return random.random() < p_made

    def field_goal_duration(self, is_blocked: bool) -> int:
        """
        Generates the duration of a field goal play

        Args:
            is_blocked (bool): Whether a field goal was blocked
        
        Returns:
            int: The duration of the play
        """
        if is_blocked:
            return int(round(
                skewnorm.rvs(
                    a=self.field_goal_blocked_duration_skew,
                    loc=self.field_goal_blocked_duration_mean,
                    scale=self.field_goal_blocked_duration_std
                )
            ))
        else:
            return int(round(
                skewnorm.rvs(
                    a=self.field_goal_not_blocked_duration_skew,
                    loc=self.field_goal_not_blocked_duration_mean,
                    scale=self.field_goal_not_blocked_duration_std
                )
            ))
