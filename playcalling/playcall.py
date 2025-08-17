from enum import Enum

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

    def __str__(self) -> str:
        """
        Converts a PlayCall instance to a human-readable string

        Returns:
            str: The PlayCall as a human-readable string
        """
        return self.name
