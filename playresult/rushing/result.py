class RushResult:
    def __init__(
            self,
            yards_gained: int,
            play_duration: int,
            fumble: bool=False,
            return_yards: int=0,
            touchdown: bool=False
        ):
        """
        Constructor for the RushResult class
        """
        self.yards_gained = yards_gained
        self.play_duration = play_duration
        self.fumble = fumble
        self.return_yards = return_yards
        self.touchdown = touchdown

    def __str__(self) -> str:
        """
        Formats a RushResult as a string

        Returns:
            str: The result as a human-readable string
        """
        res = f"({self.play_duration}s) Rush for {self.yards_gained} yards."
        if self.fumble:
            res += " FUMBLE recovered by the defense."
            if self.return_yards != 0:
                res += f" Fumble was returned {self.return_yards} yards."
        if self.touchdown:
            res += " TOUCHDOWN!"
        return res
