from enum import Enum

class PlayCall(Enum):
    PASS = 0
    RUN = 1
    KICKOFF = 2
    NO_PLAY = 3
    PUNT = 4
    EXTRA_POINT = 5
    FIELD_GOAL = 6
    QB_KNEEL = 7
    QB_SPIKE = 8

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
