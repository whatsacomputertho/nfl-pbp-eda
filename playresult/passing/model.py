import random
import numpy as np
from context.context import PlayContext
from playresult.rushing.model import RushResultModel
from playresult.rushing.result import RushResult
from playresult.passing.result import PassResult
from scipy.stats import skewnorm
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill
from typing import Union

class PassResultModel:
    def __init__(self):
        """
        Constructor for the PassResultModel class
        """
        # Pressure probability regression
        self.p_pressure_intr = 0.271330308819705
        self.p_pressure_coef = -0.21949841

        # Sack probability regression
        self.p_sack_intr = 0.10898853099029118
        self.p_sack_coef = -0.08144463

        # Scramble probability regression
        self.p_scramble_intr = 0.004914770911025865
        self.p_scramble_coef = 0.13433329

        # Short pass probability regression
        self.p_short_pass_intr = 0.8410555875020549
        self.p_short_pass_coef_1 = -0.0054862949
        self.p_short_pass_coef_2 = 0.000050472999

        # Mean short pass distance regression
        self.mean_short_pass_dist_intr = 3.4999015440062564
        self.mean_short_pass_dist_coef_1 = 0.0604532760
        self.mean_short_pass_dist_coef_2 = -0.00118944537
        self.mean_short_pass_dist_coef_3 = 0.00000662934811

        # Std short pass distance regression
        self.std_short_pass_dist_intr = 3.365933454906047
        self.std_short_pass_dist_coef_1 = 0.130891269
        self.std_short_pass_dist_coef_2 = -0.00237804912
        self.std_short_pass_dist_coef_3 = 0.0000127875476

        # Mean deep pass distance regression
        self.mean_deep_pass_dist_intr = 2.405519456054698
        self.mean_deep_pass_dist_coef_1 = 1.23979494
        self.mean_deep_pass_dist_coef_2 = -0.0204279438
        self.mean_deep_pass_dist_coef_3 = 0.000106455687

        # Std deep pass distance regression
        self.std_deep_pass_dist_intr = -1.3385882641162565
        self.std_deep_pass_dist_coef_1 = 0.277596854
        self.std_deep_pass_dist_coef_2 = -0.00120030840
        self.std_deep_pass_dist_coef_3 = -0.00000553839342

        # Interception probability regression
        self.p_interception_intr = 0.05628420712097409
        self.p_interception_coef = -0.06021105

        # Mean interception return yards regression
        self.mean_int_return_yards_intr = 11.952396063360451
        self.mean_int_return_yards_coef_1 = 0.134680678
        self.mean_int_return_yards_coef_2 = -0.00176264090
        self.mean_int_return_yards_coef_3 = -0.00000170755614

        # Std interception return yards regression
        self.std_int_return_yards_intr = 27.359295307597726
        self.std_int_return_yards_coef_1 = -0.298495830
        self.std_int_return_yards_coef_2 = 0.00302760757
        self.std_int_return_yards_coef_3 = -0.0000206954185

        # Skew interception return yards regression
        self.skew_int_return_yards_intr = 2.4745876927563324
        self.skew_int_return_yards_coef_1 = -0.00592938387
        self.skew_int_return_yards_coef_2 = -0.000720407529
        self.skew_int_return_yards_coef_3 = 0.00000700818986

        # Completed pass probability regression
        self.p_complete_intr = 0.3039580511583472 # Adjusted -0.33
        self.p_complete_coef = 0.40872763 # Adusted + 0.31

        # Zero yards after catch regression
        self.p_zero_yac_intr = 0.16761265601223527
        self.p_zero_yac_coef = -0.06038915

        # Mean yards after catch regression
        self.mean_yac_intr = 3.744998660966435
        self.mean_yac_coef_1 = 2.21147177
        self.mean_yac_coef_2 = 2.36122192

        # Std yards after catch regression
        self.std_yac_intr = 5.404781207922575
        self.std_yac_coef_1 = 0.28690679
        self.std_yac_coef_2 = 5.88666152

        # Skew yards after catch regression
        self.skew_yac_intr = 3.0784534230008083
        self.skew_yac_coef = -0.10326043

        # Fumble probability
        self.p_fumble = 0.1

        # Mean regression for play duration, std fixed at 2
        self.mean_play_duration_intr = 5.32135821
        self.mean_play_duration_coef_1 = 0.11343699
        self.mean_play_duration_coef_2 = -0.00056798

    def sim(
            self,
            context: PlayContext,
            offense: OffensiveSkill,
            defense: DefensiveSkill
        ) -> Union[PassResult, RushResult]:
        """
        Simulates a passing play

        Args:
            context (PlayContext): The current play context
            offense (OffensiveSkill): The offense's skill levels
            defense (DefensiveSkill): The defense's skill levels
        
        Returns:
            PassResult | RushResult: The result of the play
        """
        norm_diff_blocking_blitzing = 0.5 + ((offense.blocking - defense.blitzing) / 2)
        norm_diff_turnovers = 0.5 + ((offense.turnovers - defense.turnovers) / 2)
        norm_diff_passing = 0.5 + ((offense.passing - defense.pass_defense) / 2)
        norm_diff_receiving = 0.5 + ((offense.receiving - defense.coverage) / 2)

        # 1. Is QB pressured?
        pressure = self.is_pressure(norm_diff_blocking_blitzing)
        if pressure:
            # 2. If pressured, is QB sacked?
            if self.is_sack(norm_diff_blocking_blitzing):
                # TODO: Model yards lost on sack plays
                return PassResult(
                    pressure=pressure,
                    sack=True,
                    sack_yards_lost=3,
                    play_duration=self.play_duration(3)
                )
            
            # 3. If not sacked, does QB scramble?
            if self.is_scramble(offense.scrambling):
                # 4. If scramble, rush result
                return self.scramble_result(
                    context=context,
                    scrambling=offense.scrambling,
                    ball_handling=offense.turnovers,
                    rush_defense=defense.rush_defense,
                    forced_fumbles=defense.turnovers
                )
        
        # 5. Pass distance
        if self.is_short_pass(yard_line=context.yard_line):
            pass_dist = self.short_pass_distance(yard_line=context.yard_line)
        else:
            pass_dist = self.deep_pass_distance(yard_line=context.yard_line)
        
        # 6. Interception?
        if self.is_interception(norm_diff_turnovers):
            # 7. If interception, return yards
            return_yards = self.interception_return_yards(context.yard_line)
            return PassResult(
                pressure=pressure,
                pass_dist=pass_dist,
                interception=True,
                return_yards=return_yards,
                play_duration=self.play_duration(pass_dist+return_yards)
            )
        
        # 8. Complete?
        complete = self.complete_pass(norm_diff_passing)
        if complete:
            # 9. Yards after catch
            if self.zero_yards_after_catch(norm_diff_receiving):
                yac = 0
            else:
                yac = self.yards_after_catch(norm_diff_receiving)

            # 10. Fumble?
            if self.is_fumble():
                # 11. If fumble, return yards
                return_yards = self.fumble_recovery_return_yards()
                return PassResult(
                    pressure=pressure,
                    pass_dist=pass_dist,
                    complete=complete,
                    fumble=True,
                    return_yards=return_yards,
                    yac=yac,
                    play_duration=self.play_duration(pass_dist+yac+return_yards)
                )
            return PassResult(
                pressure=pressure,
                pass_dist=pass_dist,
                complete=complete,
                yac=yac,
                play_duration=self.play_duration(pass_dist+yac)
            )
        
        # Incomplete pass
        return PassResult(
            pressure=pressure,
            pass_dist=pass_dist,
            complete=complete,
            play_duration=self.play_duration(pass_dist)
        )

    # 1. Is QB pressured?
    def is_pressure(self, norm_diff_blocking_blitzing: float) -> bool:
        """
        Based on the normalized skill differential between the offense's
        blocking and the defense's blitzing, generates whether the QB came
        under pressure on the play

        Args:
            norm_diff_blocking_blitzing (float): Blocking / blitzing skill diff
        
        Returns:
            bool: Whether the quarterback was pressured on the play
        """
        p_pressure = self.p_pressure_intr + (self.p_pressure_coef * norm_diff_blocking_blitzing)
        return random.random() < p_pressure

    # 2. If pressured, is QB sacked?
    def is_sack(self, norm_diff_blocking_blitzing: float) -> bool:
        """
        Based on the normalized skill differential between the offense's
        blocking and the defense's blitzing, generates whether the QB was
        sacked on the play, assuming the QB was pressured

        Args:
            norm_diff_blocking_blitzing (float): Blocking / blitzing skill diff
        
        Returns:
            bool: Whether the quarterback was sacked on the play
        """
        p_sack = self.p_sack_intr + (self.p_sack_coef * norm_diff_blocking_blitzing)
        return random.random() < p_sack
    
    # 3. If not sacked, does QB scramble?
    def is_scramble(self, norm_scrambling: float) -> bool:
        """
        Based on the quarterback's normalized scrambling skill, generates
        whether the QB scrambles while under pressure

        Args:
            norm_scrambling (float): QB scrambling skill
        
        Returns:
            bool: Whether the quarterback scrambled on the play
        """
        p_scramble = self.p_scramble_intr + (self.p_scramble_coef * norm_scrambling)
        return random.random() < p_scramble
    
    # 4. If scramble, rush result
    def scramble_result(
            self,
            context: PlayContext,
            scrambling: float,
            ball_handling: float,
            rush_defense: float,
            forced_fumbles: float
        ) -> RushResult:
        """
        Generates the result of a QB scramble

        Args:
            context (PlayContext): The play context
            scrambling (float): QB scrambling skill
            ball_handling (float): QB ball handling skill
            rush_defense (float): Defense's rush defense skill
            forced_fumbles (float): Defense's forced fumbles skill
        
        Returns:
            RushResult: The result of the QB scramble
        """
        model = RushResultModel()
        result = model.sim(
            context=context,
            offense=OffensiveSkill(
                rushing=scrambling,
                turnovers=ball_handling
            ),
            defense=DefensiveSkill(
                rush_defense=rush_defense,
                turnovers=forced_fumbles
            ),
            scramble=True
        )
        return result

    def is_short_pass(self, yard_line: int) -> bool:
        """
        Generates whether the pass is short or long

        Args:
            yard_line(int): The current yard line
        
        Returns:
            bool: Whether this was a short pass
        """
        p_short_pass = self.p_short_pass_intr + \
            (self.p_short_pass_coef_1 * yard_line) + \
            (self.p_short_pass_coef_2 * pow(yard_line, 2))
        return random.random() < p_short_pass

    def short_pass_distance(self, yard_line: int) -> int:
        """
        Generates the distance of the QB's pass for a short pass

        Args:
            yard_line (int): The current yard line
        
        Returns:
            int: The distance of the pass in yards
        """
        # Generate the mean pass distance
        mean_pass_dist = self.mean_short_pass_dist_intr + \
            (self.mean_short_pass_dist_coef_1 * yard_line) + \
            (self.mean_short_pass_dist_coef_2 * pow(yard_line, 2)) + \
            (self.mean_short_pass_dist_coef_3 * pow(yard_line, 3))

        # Generate the std pass distance
        std_pass_dist = self.std_short_pass_dist_intr + \
            (self.std_short_pass_dist_coef_1 * yard_line) + \
            (self.std_short_pass_dist_coef_2 * pow(yard_line, 2)) + \
            (self.std_short_pass_dist_coef_3 * pow(yard_line, 3))

        # Sample the normal dist to generate the past distance
        pass_dist = int(
            np.random.normal(
                loc=mean_pass_dist,
                scale=std_pass_dist
            )
        )
        if pass_dist < -2:
            pass_dist = -2
        return pass_dist

    def deep_pass_distance(self, yard_line: int) -> int:
        """
        Generates the distance of the QB's pass for a deep pass

        Args:
            yard_line (int): The current yard line
        
        Returns:
            int: The distance of the pass in yards
        """
        # Generate the mean pass distance
        mean_pass_dist = self.mean_deep_pass_dist_intr + \
            (self.mean_deep_pass_dist_coef_1 * yard_line) + \
            (self.mean_deep_pass_dist_coef_2 * pow(yard_line, 2)) + \
            (self.mean_deep_pass_dist_coef_3 * pow(yard_line, 3))

        # Generate the std pass distance
        std_pass_dist = self.std_deep_pass_dist_intr + \
            (self.std_deep_pass_dist_coef_1 * yard_line) + \
            (self.std_deep_pass_dist_coef_2 * pow(yard_line, 2)) + \
            (self.std_deep_pass_dist_coef_3 * pow(yard_line, 3))

        # Sample the normal dist to generate the past distance
        pass_dist = int(
            np.random.normal(
                loc=mean_pass_dist,
                scale=np.abs(std_pass_dist)
            )
        )
        return pass_dist

    # 6. Interception?
    def is_interception(self, norm_diff_turnovers: float) -> bool:
        """
        Generates whether a pass resulted in an interception

        Args:
            norm_diff_turnovers (float): Turnover skill differential
        
        Returns:
            bool: Whether an interception occurred
        """
        p_interception = self.p_interception_intr + (self.p_interception_coef * norm_diff_turnovers)
        return random.random() < p_interception

    # 7. If interception, return yards
    def interception_return_yards(self, yard_line: int) -> int:
        """
        Generates the return yards following an interception

        Args:
            yard_line (int): The current yard line
        
        Returns:
            int: The return yards following the interception
        """
        # Generate the mean interception return yards
        mean_int_return_yards = self.mean_int_return_yards_intr + \
            (self.mean_int_return_yards_coef_1 * yard_line) + \
            (self.mean_int_return_yards_coef_2 * pow(yard_line, 2)) + \
            (self.mean_int_return_yards_coef_3 * pow(yard_line, 3))

        # Generate the std interception return yards
        std_int_return_yards = self.std_int_return_yards_intr + \
            (self.std_int_return_yards_coef_1 * yard_line) + \
            (self.std_int_return_yards_coef_2 * pow(yard_line, 2)) + \
            (self.std_int_return_yards_coef_3 * pow(yard_line, 3))

        # Generate the skew interception return yards
        skew_int_return_yards = self.skew_int_return_yards_intr + \
            (self.skew_int_return_yards_coef_1 * yard_line) + \
            (self.skew_int_return_yards_coef_2 * pow(yard_line, 2)) + \
            (self.skew_int_return_yards_coef_3 * pow(yard_line, 3))

        # Sample the skewed normal distribution to generate INT return yards
        return_yards = int(skewnorm.rvs(
            a=skew_int_return_yards,
            loc=mean_int_return_yards,
            scale=std_int_return_yards
        ))
        return return_yards
    
    # 8. Complete?
    def complete_pass(self, norm_diff_passing: float) -> bool:
        """
        Generates whether a pass was complete

        Args:
            norm_diff_passing (float): Passing skill differential
        
        Returns:
            bool: Whether the pass was complete
        """
        p_complete = self.p_complete_intr + (self.p_complete_coef * norm_diff_passing)
        return random.random() < p_complete

    def zero_yards_after_catch(self, norm_diff_receiving: float) -> bool:
        """
        Generates whether there were no yards after the catch

        Args:
            norm_diff_receiving (float): Receiving skill differential
        
        Returns:
            int: Whether the receiver was held to 0 YAC
        """
        p_zero_yac = self.p_zero_yac_intr + (self.p_zero_yac_coef * norm_diff_receiving)
        return random.random() < p_zero_yac

    # 9. Yards after catch
    def yards_after_catch(self, norm_diff_receiving: float) -> int:
        """
        Generates yards after the catch for a completed pass

        Args:
            norm_diff_receiving (float): Receiving skill differential
        
        Returns:
            int: Yards after the catch
        """
        # Generate the mean YAC
        mean_yac = self.mean_yac_intr + \
            (self.mean_yac_coef_1 * norm_diff_receiving) + \
            (self.mean_yac_coef_2 * pow(norm_diff_receiving, 2))

        # Generate the std YAC
        std_yac = self.std_yac_intr + \
            (self.std_yac_coef_1 * norm_diff_receiving) + \
            (self.std_yac_coef_2 * pow(norm_diff_receiving, 2))

        # Generate the skew YAC
        skew_yac = self.skew_yac_intr + (self.skew_yac_coef * norm_diff_receiving)

        # GSample the skewed normal distribution to generate the YAC
        yac = int(skewnorm.rvs(
            a=skew_yac,
            loc=mean_yac,
            scale=std_yac
        ))
        return yac

    # 10. Fumble?
    def is_fumble(self) -> bool:
        """
        Generates whether a fumble occurred after the catch

        Returns:
            bool: Whether a fumble occurred after the catch
        """
        return random.random() < self.p_fumble

    def fumble_recovery_return_yards(self) -> int:
        """
        Generates the fumble recovery return yards for a rushing fumble

        Returns:
            int: The fumble recovery return yards
        """
        return int(np.random.exponential(scale=1))

    def play_duration(self, yards: int) -> float:
        """
        Generates the duration of the play in seconds.
        
        Args:
            yards (int): The yards gained, air yards, pr air + return yards
        
        Returns:
            float: The mean duration of the play in seconds
        """
        mean_play_duration = self.mean_play_duration_intr + \
            (self.mean_play_duration_coef_1 * yards) + \
            (self.mean_play_duration_coef_2 * pow(yards, 2))
        return abs(int(np.random.normal(loc=mean_play_duration, scale=2)))
