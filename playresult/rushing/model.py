import random
import numpy as np
from context.context import PlayContext
from playresult.rushing.result import RushResult
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill

class RushResultModel:
    def __init__(self):
        """
        Constructor for the RushResultModel class
        """
        # Mean & std regression for standard rushing play
        self.mean_yards_intr = 3.0503791522871384 # Adjusted -0.5
        self.mean_yards_coef = 0.32550597 # Adjusted +0.2
        self.std_yards_intr = 4.053915588534795
        self.std_yards_coef_1 = 0.2487578
        self.std_yards_coef_2 = 0.0593874

        # Mean & std regression for big, non-TD rushing play
        self.mean_big_play_yards_intr = 15.781025340879893 # Adjusted - 12
        self.mean_big_play_yards_coef = 6.32805521 # Adjusted +1
        self.std_big_play_yards_intr = 10.014877063200005
        self.std_big_play_yards_coef_1 = -3.82403981
        self.std_big_play_yards_coef_2 = 7.60215528

        # Mean regression for play duration, std fixed at 2
        self.mean_play_duration_intr = 5.32135821
        self.mean_play_duration_coef_1 = 0.11343699
        self.mean_play_duration_coef_2 = -0.00056798

        # TD probability regression for big rushing play
        self.p_big_play_td_coef = 0.39426769
        self.p_big_play_td_intr = -3.9968093269427603

        # Big play probability regression
        self.p_big_play_coef = 0.82863208 # Adjusted +0.5
        self.p_big_play_intr = -2.878726031553263 # Adjusted -1

        # Fumble probability regression
        self.p_fumble_coef = -0.05432772
        self.p_fumble_intr = 0.04932479844415921

    def sim(self, context: PlayContext, offense: OffensiveSkill, defense: DefensiveSkill) -> RushResult:
        """
        Simulates a rushing play

        Args:
            context (PlayContext): The play context
            offense (OffensiveSkill): The offense's skill levels
            defense (DevensiveSkill): The defense's skill levels
        
        Returns:
            RushResult: The result of the run play
        """
        # Derive the normalized skill differentials between each team
        norm_diff_rushing = 0.5 + ((offense.rushing - defense.rush_defense) / 2)
        norm_diff_ball_handling = 0.5 + ((offense.turnovers - defense.turnovers) / 2)

        # Determine if this is a big play, if so generate big play yards
        if self.is_big_play(norm_diff_rushing):
            # Determine if this is a big play touchdown
            # If so then yards = yards remaining
            if self.is_big_play_touchdown(norm_diff_rushing):
                yards = 100 - context.yard_line
                return RushResult(
                    yards_gained=yards,
                    play_duration=abs(
                        int(
                            np.random.normal(
                                loc=self.mean_play_duration(yards),
                                scale=2
                            )
                        )
                    ),
                    fumble=False,
                    touchdown=True
                )
            
            # Otherwise generate yards gained
            yards = int(
                np.random.normal(
                    loc=self.mean_big_play_rushing_yards(norm_diff_rushing),
                    scale=self.std_big_play_rushing_yards(norm_diff_rushing)
                )
            )
            return RushResult(
                yards_gained=yards,
                play_duration=abs(
                    int(
                        np.random.normal(
                            loc=self.mean_play_duration(yards),
                            scale=2
                        )
                    )
                ),
                fumble=False,
                touchdown=yards > (100 - context.yard_line)
            )
        
        # Generate normal play yards
        yards = int(
            np.random.normal(
                loc=self.mean_rushing_yards(norm_diff_rushing),
                scale=self.std_rushing_yards(norm_diff_rushing)
            )
        )

        # Determine if this was a fumble, if so generate return yards
        if self.is_fumble(norm_diff_ball_handling):
            ret_yards = self.fumble_recovery_return_yards()
            yards = yards - ret_yards
            dur_yards = yards + ret_yards
            return RushResult(
                yards_gained=yards,
                play_duration=abs(
                    int(
                        np.random.normal(
                            loc=self.mean_play_duration(dur_yards),
                            scale=2
                        )
                    )
                ),
                fumble=True,
                return_yards=ret_yards,
                touchdown=(context.yard_line + yards) < 0
            )
        
        # Return rush result
        return RushResult(
            yards_gained=yards,
            play_duration=abs(
                int(
                    np.random.normal(
                        loc=self.mean_play_duration(yards),
                        scale=2
                    )
                )
            ),
            fumble=False,
            touchdown=yards > (100 - context.yard_line)
        )

    def is_fumble(self, norm_diff_ball_handling: float) -> bool:
        """
        Based on the normalized skill differential between the offense's ball
        handling and the defense's forced fumbles, generates whether a given
        run play is a fumble

        Args:
            norm_diff_ball_handling (float): The ball handling skill diff
        
        Returns:
            bool: Whether the play resulted in a fumble
        """
        p_fumble = self.p_fumble_intr + (self.p_fumble_coef * norm_diff_ball_handling)
        return random.random() < p_fumble

    def is_big_play(self, norm_diff_rushing: float) -> bool:
        """
        Based on the normalized skill differential between the offense's
        rushing and the defense's rush defense, generates whether a given run
        play will go for a big play

        Args:
            norm_diff_rushing (float): The rushing skill diff
        
        Returns:
            bool: Whether the play is a big play
        """
        p_big_play = np.exp(self.p_big_play_intr + (self.p_big_play_coef * norm_diff_rushing))
        return random.random() < p_big_play

    def is_big_play_touchdown(self, norm_diff_rushing: float) -> bool:
        """
        Based on the normalized skill differential between the offense's
        rushing and the defense's rush defense, generates whether a given big
        run play will result in a touchdown

        Args:
            norm_diff_rushing (float): The rushing skill diff
        
        Returns:
            bool: Whether the play is a big play
        """
        p_big_play_td = np.exp(self.p_big_play_td_intr + (self.p_big_play_td_coef * norm_diff_rushing))
        return random.random() < p_big_play_td

    def mean_play_duration(self, yards_gained: int) -> float:
        """
        Based on the yards gained on the play, generates the mean duration of
        the play in seconds

        Args:
            yards_gained (int): The yards gained on the play
        
        Returns:
            float: The mean duration of the play in seconds
        """
        return self.mean_play_duration_intr + \
            (self.mean_play_duration_coef_1 * yards_gained) + \
            (self.mean_play_duration_coef_2 * pow(yards_gained, 2))

    def mean_rushing_yards(self, norm_diff_rushing: float) -> float:
        """
        Based on the normalized skill differential between the offense's
        rushing and the defense's rush defense, generates the mean rushing
        yards for a standard rushing play

        Args:
            norm_diff_rushing (float): The rushing skill diff
        
        Returns:
            float: The mean rushing yards
        """
        return self.mean_yards_intr + (self.mean_yards_coef * norm_diff_rushing)

    def std_rushing_yards(self, norm_diff_rushing: float) -> float:
        """
        Based on the normalized skill differential between the offense's
        rushing and the defense's rush defense, generates the standard
        deviation rushing yards for a standard rushing play

        Args:
            norm_diff_rushing (float): The rushing skill diff
        
        Returns:
            float: The standard deviation rushing yards
        """
        return self.std_yards_intr + \
            (self.std_yards_coef_1 * norm_diff_rushing) + \
            (self.std_yards_coef_2 * pow(norm_diff_rushing, 2))

    def mean_big_play_rushing_yards(self, norm_diff_rushing: float) -> float:
        """
        Based on the normalized skill differential between the offense's
        rushing and the defense's rush defense, generates the mean rushing
        yards for a big rushing play

        Args:
            norm_diff_rushing (float): The rushing skill diff
        
        Returns:
            float: The mean rushing yards
        """
        return self.mean_big_play_yards_intr + (self.mean_big_play_yards_coef * norm_diff_rushing)

    def std_big_play_rushing_yards(self, norm_diff_rushing: float) -> float:
        """
        Based on the normalized skill differential between the offense's
        rushing and the defense's rush defense, generates the standard
        deviation rushing yards for a big rushing play

        Args:
            norm_diff_rushing (float): The rushing skill diff
        
        Returns:
            float: The standard deviation rushing yards
        """
        return self.std_big_play_yards_intr + \
            (self.std_big_play_yards_coef_1 * norm_diff_rushing) + \
            (self.std_big_play_yards_coef_2 * pow(norm_diff_rushing, 2))

    def fumble_recovery_return_yards(self) -> int:
        """
        Generates the fumble recovery return yards for a rushing fumble

        Returns:
            int: The fumble recovery return yards
        """
        return int(np.random.exponential(scale=1))
