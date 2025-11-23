import random
import numpy as np
from context.context import PlayContext
from playresult.punt.result import PuntResult
from scipy.stats import skewnorm
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill

class PuntResultModel:
    def __init__(self):
        """
        Constructor for the PuntResultModel class
        """
        # Punt block probability regression
        self.p_block_intr = -0.0010160286505995551
        self.p_block_coef = 0.00703673

        # Punt inside 20 skill-based regression
        self.p_punt_inside_20_skill_intr = 0.21398823243670145
        self.p_punt_inside_20_skill_coef = 0.32878206

        # Punt inside 20 yardage-based nonlinear logistic function fit
        self.p_punt_inside_20_yardage_param_1 = 0.783829627
        self.p_punt_inside_20_yardage_param_2 = -0.200560110
        self.p_punt_inside_20_yardage_param_3 = 0.651500015
        self.p_punt_inside_20_yardage_param_4 = -0.00178251834

        # Punt inside 20 mean relative distance regression
        self.punt_inside_20_mean_rel_dist_intr = 0.20907739629135946
        self.punt_inside_20_mean_rel_dist_coef = -0.0001755

        # Punt inside 20 std relative distance regression
        self.punt_inside_20_std_rel_dist_intr = 0.17519244654293623
        self.punt_inside_20_std_rel_dist_coef = -0.0016178

        # Punt inside 20 skew distance regression
        self.punt_inside_20_skew_rel_dist_intr = 3.691739354624472
        self.punt_inside_20_skew_rel_dist_coef_1 = -0.11961015
        self.punt_inside_20_skew_rel_dist_coef_2 = 0.00081621

        # Punt outside 20 mean distance regression
        self.punt_outside_20_mean_rel_dist_intr = -0.24995460069957565
        self.punt_outside_20_mean_rel_dist_coef_1 = 0.0400507456
        self.punt_outside_20_mean_rel_dist_coef_2 = -0.000758718087
        self.punt_outside_20_mean_rel_dist_coef_3 = 0.00000442573043

        # Punt outside 20 std distance regression
        self.punt_outside_20_std_rel_dist_intr = 0.2748076520973469
        self.punt_outside_20_std_rel_dist_coef = -0.00196699

        # Punt outside 20 skew distance regression
        self.punt_outside_20_skew_rel_dist_intr = -5.631745519232158
        self.punt_outside_20_skew_rel_dist_coef_1 = 0.19789058
        self.punt_outside_20_skew_rel_dist_coef_2 = -0.00134607

        # Punt out of bounds probability regression
        self.p_punt_oob_intr = -0.0846243447082426
        self.p_punt_oob_coef_1 = 0.00575805979
        self.p_punt_oob_coef_2 = -0.0000428367831

        # Fair catch probability regression
        #self.p_fair_catch_intr = 0.37613371173695526 # Adjusted + 0.1
        self.p_fair_catch_intr = 0.37613371173695526 + 0.1
        #self.p_fair_catch_coef = -0.00441214 # Adjusted + 0.003
        self.p_fair_catch_coef = -0.00441214 + 0.003

        # Muffed punt probability regression
        self.p_muffed_punt_intr = 0.036855240326056096
        self.p_muffed_punt_coef = -0.02771741

        # Return yards mean regression
        #self.mean_rel_return_yards_intr = 0.042967812945618286 # Adjusted -0.1
        self.mean_rel_return_yards_intr = 0.042967812945618286 - 0.1
        self.mean_rel_return_yards_coef_1 = -0.02282631
        self.mean_rel_return_yards_coef_2 = 0.28982747

        # Return yards std regression
        self.std_rel_return_yards_intr = 0.06751127059206394
        self.std_rel_return_yards_coef_1 = 0.01035858
        self.std_rel_return_yards_coef_2 = 0.26338509

        # Return yards skew regression
        #self.skew_rel_return_yards_intr = 0.9832527719192217 # Adjusted -1
        self.skew_rel_return_yards_intr = 0.9832527719192217 - 1
        self.skew_rel_return_yards_coef_1 = 7.06931813
        self.skew_rel_return_yards_coef_2 = -6.94528823

        # Fumble probability regression
        self.p_fumble_intr = 0.0460047101408259
        self.p_fumble_coef = -0.04389777

        # Punt play duration regression
        self.punt_play_duration_intr = 5.2792296
        self.punt_play_duration_coef = 0.09291598

    def sim(
            self,
            context: PlayContext,
            offense: OffensiveSkill,
            defense: DefensiveSkill
        ) -> PuntResult:
        """
        Simulate a punt play

        Args:
            context (PlayContext): The current play context
            offense (OffensiveSkill): The offense's skill levels
            defense (DefensiveSkill): The defense's skill levels
        
        Returns:
            PuntResult: The result of the punt play
        """
        norm_diff_blocking_blitzing = 0.5 + ((defense.blitzing - offense.blocking) / 2)
        norm_diff_returning = 0.5 + ((defense.kick_returning - offense.kick_return_defense) / 2)

        # Is the punt blocked?
        if self.is_blocked(norm_diff_blocking_blitzing):
            return_yards = self.fumble_recovery_return_yards()
            return PuntResult(
                punt_yards=0,
                punt_return_yards=0,
                play_duration=self.duration(return_yards),
                blocked=True,
                fumble_return_yards=return_yards,
                touchback=False,
                fumble=True,
                touchdown=(context.yard_line - return_yards) <= 0
            )

        # Generate relative punt distance & calculate the actual distance
        current_yard_line = (100 - context.yard_line)
        relative_punt_distance = self.relative_punt_distance(
            is_inside_20=self.is_punt_inside_20(
                current_yard_line,
                offense.punting
            ),
            yard_line=current_yard_line
        )
        new_yard_line = int(round(current_yard_line * relative_punt_distance))
        punt_distance = current_yard_line - new_yard_line

        # Is the punt out of bounds?
        if self.is_punt_out_of_bounds(current_yard_line):
            return PuntResult(
                punt_yards=punt_distance,
                punt_return_yards=0,
                play_duration=self.duration(punt_distance),
                out_of_bounds=True
            )

        # Is a fair catch called?
        fair_catch = self.is_fair_catch(new_yard_line)
        punt_muffed = self.is_muffed_punt(norm_diff_returning)
        if fair_catch and not punt_muffed:
            return PuntResult(
                punt_yards=punt_distance,
                punt_return_yards=0,
                play_duration=self.duration(punt_distance),
                fair_catch=True
            )

        # Is the punt muffed?
        if punt_muffed:
            return_yards = self.fumble_recovery_return_yards()
            return PuntResult(
                punt_yards=punt_distance,
                punt_return_yards=0,
                play_duration=self.duration(punt_distance + return_yards),
                fumble_return_yards=return_yards,
                fair_catch=fair_catch,
                muffed=True,
                touchdown=(context.yard_line + punt_distance + return_yards) >= 100
            )

        # Relative return distance
        relative_return_distance = self.relative_return_distance(norm_diff_returning)
        punt_return_yards = int(round((100 - new_yard_line) * relative_return_distance))

        # Is there a fumble on the return?
        if self.is_fumble(norm_diff_returning):
            return_yards = self.fumble_recovery_return_yards()
            return PuntResult(
                punt_yards=punt_distance,
                punt_return_yards=punt_return_yards,
                play_duration=self.duration(punt_distance + return_yards + punt_return_yards),
                fumble_return_yards=return_yards,
                fumble=True,
                touchdown=(context.yard_line + punt_distance - return_yards + return_yards) >= 100
            )
        return PuntResult(
            punt_yards=punt_distance,
            punt_return_yards=punt_return_yards,
            play_duration=self.duration(punt_distance + punt_return_yards),
            touchdown=(context.yard_line + punt_distance - punt_return_yards) <= 0
        )

    def is_blocked(self, norm_diff_blocking_blitzing: float) -> bool:
        """
        Generates whether the punt is blocked

        Args:
            norm_diff_blocking_blitzing (float): Blitzing skill differential
        
        Returns:
            bool: Whether the punt was blocked
        """
        p_blocked = self.p_block_intr + (self.p_block_coef * norm_diff_blocking_blitzing)
        return random.random() < p_blocked

    def is_punt_inside_20(self, yard_line: int, norm_punting: float) -> bool:
        """
        Generates whether the punt lands inside the 20

        Args:
            yard_line (int): The current yard line
            norm_punting (float): How good the punter is at punting
        
        Returns:
            bool: Whether the punt landed inside the 20
        """
        p_inside_20_skill = self.p_punt_inside_20_skill_intr + (self.p_punt_inside_20_skill_coef * norm_punting)
        p_inside_20_yardline = self.p_punt_inside_20_yardage_param_1 / ((
                1 + np.exp(
                    -self.p_punt_inside_20_yardage_param_2*(
                        yard_line - self.p_punt_inside_20_yardage_param_3
                    )
                )
            ) + self.p_punt_inside_20_yardage_param_4)
        p_inside_20 = ((p_inside_20_skill * 0.4) + (p_inside_20_yardline * 0.6)) * 1.18
        return random.random() < p_inside_20

    def relative_punt_distance(self, is_inside_20: bool, yard_line: int) -> float:
        """
        Generates the punt distance relative to the current yard line

        Args:
            is_inside_20 (bool): Whether the punt will land inside the 20
            yard_line (int): The current yard line
        
        Returns:
            float: The relative distance
        """
        if is_inside_20:
            return float(skewnorm.rvs(
                a=self.punt_inside_20_skew_rel_dist_intr + \
                    (self.punt_inside_20_skew_rel_dist_coef_1 * yard_line) + \
                    (self.punt_inside_20_skew_rel_dist_coef_2 * pow(yard_line, 2)),
                loc=self.punt_inside_20_mean_rel_dist_intr + \
                    (self.punt_inside_20_mean_rel_dist_coef * yard_line),
                scale=self.punt_inside_20_std_rel_dist_intr + \
                    (self.punt_inside_20_std_rel_dist_coef * yard_line)
            ))
        return float(skewnorm.rvs(
            a=self.punt_outside_20_skew_rel_dist_intr + \
                (self.punt_outside_20_skew_rel_dist_coef_1 * yard_line) + \
                (self.punt_outside_20_skew_rel_dist_coef_2 * pow(yard_line, 2)),
            loc=self.punt_outside_20_mean_rel_dist_intr + \
                (self.punt_outside_20_mean_rel_dist_coef_1 * yard_line) + \
                (self.punt_outside_20_mean_rel_dist_coef_2 * pow(yard_line, 2)) + \
                (self.punt_outside_20_mean_rel_dist_coef_3 * pow(yard_line, 3)),
            scale=self.punt_outside_20_std_rel_dist_intr + \
                (self.punt_outside_20_std_rel_dist_coef * yard_line)
        ))

    def is_punt_out_of_bounds(self, yard_line: int) -> bool:
        """
        Generates whether the punt went out of bounds

        Args:
            yard_line (int): The current yard line
        
        Returns:
            bool: Whether the punt went out of bounds
        """
        p_punt_oob = self.p_punt_oob_intr + (self.p_punt_oob_coef_1 * yard_line) + \
            (self.p_punt_oob_coef_2 * pow(yard_line, 2))
        return random.random() < p_punt_oob

    def is_fair_catch(self, punt_landing: int) -> bool:
        """
        Generates whether a fair catch was called on the punt

        Args:
            punt_landing (int): Where the punt landed
        
        Returns:
            bool: Whether a fair catch was called
        """
        p_fair_catch = self.p_fair_catch_intr + (self.p_fair_catch_coef * punt_landing)
        return random.random() < p_fair_catch

    def is_muffed_punt(self, norm_diff_returning: float) -> bool:
        """
        Generates whether the punt was muffed

        Args:
            norm_diff_returning (float): Punt returning skill differential
        
        Returns:
            bool: Whether the punt was muffed
        """
        p_muffed_punt = self.p_muffed_punt_intr + (self.p_muffed_punt_coef * norm_diff_returning)
        return random.random() < p_muffed_punt

    def relative_return_distance(self, norm_diff_returning: float) -> float:
        """
        Generates the return distance relative to the punt landing

        Args:
            norm_diff_returning (float): Punt returning skill differential
        
        Returns:
            float: Return distance relative to punt landing
        """
        return float(skewnorm.rvs(
            a=self.skew_rel_return_yards_intr + \
                (self.skew_rel_return_yards_coef_1 * norm_diff_returning) + \
                (self.skew_rel_return_yards_coef_2 * pow(norm_diff_returning, 2)),
            loc=self.mean_rel_return_yards_intr + \
                (self.mean_rel_return_yards_coef_1 * norm_diff_returning) + \
                (self.mean_rel_return_yards_coef_2 * pow(norm_diff_returning, 2)),
            scale=self.std_rel_return_yards_intr + \
                (self.std_rel_return_yards_coef_1 * norm_diff_returning) + \
                (self.std_rel_return_yards_coef_2 * pow(norm_diff_returning, 2))
        ))

    def is_fumble(self, norm_diff_returning: float) -> bool:
        """
        Generates whether there was a fumble on the punt return

        Args:
            norm_diff_returning (float): Punt returning skill differential
        
        Returns:
            bool: Whether there was a fumble on the punt return
        """
        p_fumble = self.p_fumble_intr + (self.p_fumble_coef * norm_diff_returning)
        return random.random() < p_fumble

    def fumble_recovery_return_yards(self) -> int:
        """
        Generates the fumble recovery return yards for a rushing fumble

        Returns:
            int: The fumble recovery return yards
        """
        return int(np.random.exponential(scale=1))

    def duration(self, yards: int) -> int:
        """
        Generates the duration of the punt play

        Args:
            yards (int): The total yards, including punt & return distances
        
        Returns:
            int: The duration of the play in seconds
        """
        return int(
            np.random.normal(
                loc=self.punt_play_duration_intr + \
                    (self.punt_play_duration_coef * yards),
                scale=2
            )
        )
