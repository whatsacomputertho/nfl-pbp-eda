import random
import numpy as np
from context.context import PlayContext

class BetweenPlayModel():
    """
    Generates how many seconds pass between plays
    """
    def __init__(self):
        """
        Constructor for the BetweenPlayModel class
        """
        # Up-tempo probability regression
        self.p_up_tempo_intr = -3.5395125211354683
        self.p_up_tempo_coef = 3.03267023

        # Normal between play duration
        self.mean_between_play_duration = 20
        self.std_between_play_duration = 5

        # Up-tempo between play duration
        self.mean_up_tempo_between_play_duration = 6
        self.std_up_tempo_between_play_duration = 2

        # Probability defense is not set
        self.p_defense_not_set = 0.08
        self.p_defense_not_set_up_tempo = 0.3

        # Probability a coach calls a timeout when the defense is not set
        # Also the probability a coach calls a timeout on a critical down to get set
        self.p_defense_not_set_timeout_intr = 0.2
        self.p_defense_not_set_timeout_coef = 0.4

    def sim(
            self,
            context: PlayContext,
            risk_taking: float,
            up_tempo_tendency: float,
            is_clock_running: bool
        ) -> tuple[int, bool, bool]:
        """
        Generates the number of seconds which pass between the play if the
        clock is running

        Args:
            context (PlayContext): The current play context
            up_tempo_tendency (bool): The coach's tendency to go up-tempo
        
        Return:
            int: The time elapsed after the play
            bool: Whether a timeout was called
            bool: Whether the defense called a timeout
        """
        if not is_clock_running:
            return 0, False, False
        is_up_tempo = self.is_up_tempo(context, up_tempo_tendency)
        if self.is_defense_not_set_timeout(context, risk_taking, is_up_tempo):
            return 0, True, True
        if self.is_defense_clock_management_timeout(context, is_clock_running):
            return 0, True, True
        if self.is_offense_clock_management_timeout(context, is_clock_running):
            return 0, True, False
        is_drain_the_clock = self.is_drain_the_clock(context)
        return self.between_play_duration(is_up_tempo, is_drain_the_clock), False, False

    def is_drain_the_clock(self, context: PlayContext) -> bool:
        """
        Determine whether the offense will drain the clock

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether the offense will drain the clock
        """
        scores_up_by = context.score_diff / 8
        if scores_up_by <= 0:
            return False
        drain_clock_threshold = int(scores_up_by * 4 * 60)
        if (context.quarter >= 4) and (context.half_seconds < drain_clock_threshold):
            return True
        return False

    def is_up_tempo(self, context: PlayContext, up_tempo_tendency: float) -> bool:
        """
        Determine whether the offense will go up-tempo

        Args:
            context (PlayContext): The current play context
            up_tempo_tendency (bool): The coach's tendency to go up-tempo
        
        Returns:
            bool: Whether the offense will go up-tempo
        """
        if (context.quarter >= 4) and (context.half_seconds <= 180) \
            and (context.score_diff < 0) and (context.score_diff >= -17):
            return True
        p_up_tempo = np.exp(
            self.p_up_tempo_intr + (self.p_up_tempo_coef * up_tempo_tendency)
        )
        return random.random() < p_up_tempo

    def is_defense_not_set(self, is_up_tempo: bool) -> bool:
        """
        Generates whether the defense is not set

        Args:
            is_up_tempo (bool): Whether the offense is going up-tempo
        
        Returns:
            bool: Whether the defense is not set
        """
        rng = random.random()
        if is_up_tempo:
            return rng < self.p_defense_not_set_up_tempo
        return rng < self.p_defense_not_set

    def is_defense_not_set_timeout(
            self,
            context: PlayContext,
            risk_taking: float,
            is_up_tempo: bool
        ) -> bool:
        """
        Generates whether the defense calls a timeout due to being not set

        Args:
            context (PlayContext): The current play context
            risk_taking (float): The defense's coach's risk taking tendency
            is_up_tempo (bool): Whether the offense is going up-tempo
        
        Returns:
            bool: Whether a timeout is called
        """
        if (context.def_timeouts <= 0) or (context.quarter > 2):
            return False
        p_timeout = self.p_defense_not_set_timeout_intr + (self.p_defense_not_set_timeout_coef * risk_taking)
        if self.is_defense_not_set(is_up_tempo):
            return random.random() < p_timeout
        return False

    def is_critical_down(self, context: PlayContext) -> bool:
        """
        Determines whether this is a critical down, which we define as a third
        down late in the half in a one-score game
        """
        return (context.down == 3) and (context.half_seconds < 180) \
            and (context.score_diff < 0) and (context.score_diff > 0)

    def is_critical_down_timeout(self, context: PlayContext, risk_taking: float) -> bool:
        """
        Generates whether a timeout is called on a critical down to get set

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether a timeout is called
        """
        if (context.def_timeouts <= 0) or (context.quarter > 2):
            return False
        p_timeout = self.p_defense_not_set_timeout_intr + (self.p_defense_not_set_timeout_coef * risk_taking)
        if self.is_critical_down(context):
            return random.random() < p_timeout
        return False

    def is_offense_clock_management_situation(self, context: PlayContext) -> bool:
        """
        Determines whether the current context is a clock management situation
        for the offense

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether the current context is a clock management situation
        """
        return (context.quarter >= 4) and (context.half_seconds <= 180) \
            and (context.score_diff < 0) and (context.score_diff >= -17)

    def is_defense_clock_management_situation(self, context: PlayContext) -> bool:
        """
        Determines whether the current context is a clock management situation
        for the defense

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether the current context is a clock management situation
        """
        return (context.quarter >= 4) and (context.half_seconds <= 180) \
            and (context.score_diff < 0) and (context.score_diff >= 17)

    def is_last_play(self, context: PlayContext) -> bool:
        """
        Determines whether this is the last play of the game

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether this is the last play of the game
        """
        return context.half_seconds < 5

    def is_offense_clock_management_timeout(self, context: PlayContext, is_clock_running: bool) -> bool:
        """
        Determines whether the offense calls a timeout to stop the clock

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether a timeout is called
        """
        if (not is_clock_running) or (context.off_timeouts <= 0):
            return False
        if self.is_offense_clock_management_situation(context):
            return True
        return False

    def is_defense_clock_management_timeout(self, context: PlayContext, is_clock_running: bool) -> bool:
        """
        Determines whether the offense calls a timeout to stop the clock

        Args:
            context (PlayContext): The current play context
        
        Returns:
            bool: Whether a timeout is called
        """
        if (not is_clock_running) or (context.def_timeouts <= 0):
            return False
        if self.is_defense_clock_management_situation(context):
            return True
        return False

    def between_play_duration(self, is_up_tempo: bool, is_drain_the_clock: bool) -> int:
        """
        Generates the number of seconds which pass between the play

        Args:
            is_up_tempo (bool): Whether the offense is going up-tempo
            is_drain_the_clock (bool): Whether the offense is draining the clock
        
        Return:
            int: The time elapsed after the play
        """
        if is_drain_the_clock:
            return 40 - int(np.random.exponential(scale=1))
        if is_up_tempo:
            return abs(int(np.random.normal(
                loc=self.mean_up_tempo_between_play_duration,
                scale=self.std_up_tempo_between_play_duration
            )))
        return abs(int(np.random.normal(
            loc=self.mean_between_play_duration,
            scale=self.std_between_play_duration
        )))
