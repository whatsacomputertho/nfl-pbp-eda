from enum import Enum
from typing import Any

class PlayCall(Enum):
    SHORT_PASS = 0
    DEEP_PASS = 1
    RUN_LEFT = 2
    RUN_MIDDLE = 3
    RUN_RIGHT = 4
    KICKOFF = 5
    PUNT = 6
    EXTRA_POINT = 7
    FIELD_GOAL = 8
    QB_KNEEL = 9
    QB_SPIKE = 10
    OFFENSE_TIMEOUT = 11
    DEFENSE_TIMEOUT = 12

    def from_str(playcall: str) -> "PlayCall":
        """
        Initialize a PlayCall from a human-readable string

        Args:
            playcall (str): The human-readable playcall string

        Returns:
            PlayCall: The initialized PlayCall enum instance
        """
        return PlayCall[playcall.upper()]

    def to_one_hot(self) -> dict[str, Any]:
        """
        Encode the playcall as one-hot encoded floating point properties in a
        dictionary

        Returns:
            dict: The one-hot encoded playcall dict
        """
        return {
            "play_type_short_pass": 0.0 if self.value == 0 else 1.0,
            "play_type_deep_pass": 0.0 if self.value == 1 else 1.0,
            "play_type_run_left": 0.0 if self.value == 2 else 1.0,
            "play_type_run_middle": 0.0 if self.value == 3 else 1.0,
            "play_type_run_right": 0.0 if self.value == 4 else 1.0,
            "play_type_kickoff": 0.0 if self.value == 5 else 1.0,
            "play_type_punt": 0.0 if self.value == 6 else 1.0,
            "play_type_extra_point": 0.0 if self.value == 7 else 1.0,
            "play_type_field_goal": 0.0 if self.value == 8 else 1.0,
            "play_type_qb_kneel": 0.0 if self.value == 9 else 1.0,
            "play_type_qb_spike": 0.0 if self.value == 10 else 1.0,
            "play_type_offense_timeout": 0.0 if self.value == 11 else 1.0,
            "play_type_defense_timeout": 0.0 if self.value == 12 else 1.0,
        }

    def __str__(self) -> str:
        """
        Converts a PlayCall instance to a human-readable string

        Returns:
            str: The PlayCall as a human-readable string
        """
        return self.name
