import random
import numpy as np
from context.context import PlayContext
from playresult.kickoff.result import KickoffResult
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill
from scipy.stats import skewnorm

class KickoffResultModel:
    def __init__(self):
        """
        Constructor for the KickoffResultModel class
        """
        # Touchback probability regression
        self.p_touchback_intr = 0.2528877428268531
        self.p_touchback_coef = 0.62457076

        # Out of bounds probability regression
        self.p_out_of_bounds_intr = 0.013879833381776598
        self.p_out_of_bounds_coef = -0.01063523

        # Kickoff inside 20 probability
        self.p_kickoff_inside_20 = 0.2

        # Kickoff inside 20 mean distance
        self.mean_kickoff_inside_20_dist = 64.3

        # Kickoff inside 20 std distance regression
        self.std_kickoff_inside_20_dist_intr = 4.516109138481186
        self.std_kickoff_inside_20_dist_coef = 1.97369663

        # Kickoff inside 20 skew distance
        self.skew_kickoff_inside_20_dist = -1.7

        # Kickoff outside 20 mean distance regression
        self.mean_kickoff_outside_20_dist_intr = 59.31943845056676
        self.mean_kickoff_outside_20_dist_coef = -3.42944893

        # Kickoff outside 20 std distance regression
        self.std_kickoff_outside_20_dist_intr = 11.602550109235546
        self.std_kickoff_outside_20_dist_coef = 6.81862647

        # Kickoff outside 20 skew distance
        self.skew_kickoff_outside_20_dist = -2

        # Fair catch probability regression
        self.p_fair_catch_intr = 0.02694588730554516
        self.p_fair_catch_coef = -0.03716183

        # Mean kickoff return yards regression
        self.mean_kickoff_return_yards_intr = -0.6236115656913945
        self.mean_kickoff_return_yards_coef = 20.05077203

        # Std kickoff return yards regression
        self.std_kickoff_return_yards_intr = 6.421970424325094
        self.std_kickoff_return_yards_coef = 12.34550665

        # Skew kickoff return yards regression
        self.skew_kickoff_return_yards_intr = 3.62041405111988
        self.skew_kickoff_return_yards_coef = -2.65709746

        # Kickoff return fumble probability
        self.p_kickoff_return_fumble = 0.007

        # Kickoff return play duration regression
        self.kickoff_return_play_duration_intr = 0.11217103
        self.kickoff_return_play_duration_coef = 1.20326252

    def sim(
            self,
            context: PlayContext,
            offense: OffensiveSkill,
            defense: DefensiveSkill
        ) -> KickoffResult:
        """
        Simulate a kickoff play

        Args:
            context (PlayContext): The current play context
            offense (OffensiveSkill): The offense's skill levels
            defense (DefensiveSkill): The defense's skill levels
        
        Returns:
            KickoffResult: The result of the kickoff play
        """
        norm_kicking = offense.kickoffs
        norm_diff_returning = 0.5 + ((defense.kick_returning - offense.kick_return_defense) / 2)
        if self.is_touchback(norm_kicking):
            return KickoffResult(
                kickoff_yards=65,
                kick_return_yards=0,
                play_duration=0,
                touchback=True
            )
        out_of_bounds = self.is_out_of_bounds(norm_kicking)
        inside_20 = self.is_kickoff_inside_20(norm_kicking)
        kickoff_yards = self.kickoff_distance(norm_kicking, inside_20)
        if out_of_bounds:
            return KickoffResult(
                kickoff_yards=kickoff_yards,
                kick_return_yards=0,
                play_duration=0,
                out_of_bounds=True
            )
        if self.is_fair_catch(norm_diff_returning):
            return KickoffResult(
                kickoff_yards=kickoff_yards,
                kick_return_yards=0,
                play_duration=0,
                fair_catch=True
            )
        return_yards = self.kick_return_yards(norm_diff_returning)
        if self.is_kick_return_fumble():
            fumble_return_yards = self.fumble_recovery_return_yards()
            return KickoffResult(
                kickoff_yards=kickoff_yards,
                kick_return_yards=return_yards,
                play_duration=self.kick_return_duration(return_yards + fumble_return_yards),
                fumble=True,
                touchdown=(35 + kickoff_yards - return_yards + fumble_return_yards) >= 100
            )
        return KickoffResult(
            kickoff_yards=kickoff_yards,
            kick_return_yards=return_yards,
            play_duration=self.kick_return_duration(return_yards),
            touchdown=(35 + kickoff_yards - return_yards) <= 0
        )

    def is_touchback(self, norm_kicking: float) -> bool:
        """
        Generates whether a touchback occurred

        Args:
            norm_kicking (float): Kicking skill level
        
        Returns:
            bool: Whether a touchback occurred
        """
        p_touchback = self.p_touchback_intr + (self.p_touchback_coef * norm_kicking)
        return random.random() < p_touchback

    def is_out_of_bounds(self, norm_kicking: float) -> bool:
        """
        Generates whether the kickoff went out of bounds

        Args:
            norm_kicking (float): Kicking skill level
        
        Returns:
            bool: Whether the kickoff went out of bounds
        """
        p_oob = self.p_out_of_bounds_intr + (self.p_out_of_bounds_coef * norm_kicking)
        return random.random() < p_oob

    def is_kickoff_inside_20(self, norm_kicking: float) -> bool:
        """
        Generates whether the kickoff landed inside the 20

        Args:
            norm_kicking (float): Kicking skill level
        
        Returns:
            bool: Whether the kickoff landed inside the 20
        """
        return random.random() < self.p_kickoff_inside_20

    def kickoff_distance(self, norm_kicking: float, inside_20: bool) -> bool:
        """
        Generates the kickoff distance

        Args:
            norm_kicking (float): Kicking skill level
            inside_20 (bool): Whether the kickoff landed inside the 20
        
        Returns:
            bool: The kickoff distance
        """
        if inside_20:
            return int(round(skewnorm.rvs(
                a=self.skew_kickoff_inside_20_dist,
                loc=self.mean_kickoff_inside_20_dist,
                scale=self.std_kickoff_inside_20_dist_intr + (
                    self.std_kickoff_inside_20_dist_coef * norm_kicking
                )
            )))
        else:
            return int(round(skewnorm.rvs(
                a=self.skew_kickoff_outside_20_dist,
                loc=self.mean_kickoff_outside_20_dist_intr + (
                    self.mean_kickoff_outside_20_dist_coef * norm_kicking
                ),
                scale=self.std_kickoff_outside_20_dist_intr + (
                    self.std_kickoff_outside_20_dist_coef * norm_kicking
                )
            )))

    def is_fair_catch(self, norm_diff_returning: float) -> bool:
        """
        Generates whether the kickoff resulted in a fair catch
        """
        p_fair_catch = self.p_fair_catch_intr + (self.p_fair_catch_coef * norm_diff_returning)
        return random.random() < p_fair_catch

    def kick_return_yards(self, norm_diff_returning: float) -> bool:
        """
        Generates the yards gained or lost on the kick return
        """
        return int(round(skewnorm.rvs(
            a=self.skew_kickoff_return_yards_intr + (
                self.skew_kickoff_return_yards_coef * norm_diff_returning
            ),
            loc=self.mean_kickoff_return_yards_intr + (
                self.mean_kickoff_return_yards_coef * norm_diff_returning
            ),
            scale=self.std_kickoff_return_yards_intr + (
                self.std_kickoff_return_yards_coef * norm_diff_returning
            )
        )))

    def is_kick_return_fumble(self) -> bool:
        """
        Generates whether a fumble occurred on the kick return
        
        Returns:
            bool: Whether a fumble occurred
        """
        return random.random() < self.p_kickoff_return_fumble

    def fumble_recovery_return_yards(self) -> int:
        """
        Generates the fumble recovery return yards for a rushing fumble

        Returns:
            int: The fumble recovery return yards
        """
        return int(np.random.exponential(scale=1))

    def kick_return_duration(self, yards_gained: int) -> int:
        """
        Generates the duration of the kickoff return

        Args:
            yards_gained (int): Yards gained on the return

        Returns:
            int: Duration in clock seconds of the return
        """
        return int(
            np.sqrt(
                np.abs(
                    np.random.normal(
                        loc=self.kickoff_return_play_duration_intr + \
                        (self.kickoff_return_play_duration_coef * yards_gained),
                        scale=2
                    )
                )
            )
        )
