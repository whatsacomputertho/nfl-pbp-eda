from enum import Enum
from typing import Any

class PlayCall(Enum):
    RUN = 0
    PASS = 1
    FIELD_GOAL = 2
    PUNT = 3
    KICKOFF = 4
    EXTRA_POINT = 5
    QB_KNEEL = 6
    QB_SPIKE = 7
    OFFENSE_TIMEOUT = 8
    DEFENSE_TIMEOUT = 9

    def __str__(self) -> str:
        """
        Converts a PlayCall instance to a human-readable string

        Returns:
            str: The PlayCall as a human-readable string
        """
        return self.name
