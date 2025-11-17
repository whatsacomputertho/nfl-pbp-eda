import copy
from context.context import GameContext

class PassResult:
    def __init__(
            self,
            pressure: bool=False,
            sack: bool=False,
            sack_yards_lost: int=0,
            pass_dist: int=10,
            interception: bool=False,
            return_yards: int=0,
            complete: bool=True,
            yac: int=0,
            fumble: bool=False,
            touchdown: bool=False,
            play_duration: int=5
        ):
        """
        Constructor for the PassResult class
        """
        self.pressure = pressure
        self.sack = sack
        self.sack_yards_lost = sack_yards_lost
        self.pass_dist = pass_dist
        self.interception = interception
        self.return_yards = return_yards
        self.complete = complete
        self.yac = yac
        self.fumble = fumble
        self.touchdown = touchdown
        self.play_duration = play_duration

    def next_context(self, context: GameContext) -> GameContext:
        """
        Converts the current game context into the next game context given
        this play result
        """
        new_context = copy.deepcopy(context)
        new_context.update_clock(self.play_duration)
        new_context.update_yard_line(self.yards_gained())
        if self.fumble or self.interception:
            new_context.home_possession = not new_context.home_possession
            new_context.yard_line = 100 - new_context.yard_line
        return new_context

    def yards_gained(self) -> int:
        """
        Returns the yards gained on the play
        """
        if self.sack:
            return -self.sack_yards_lost
        if self.complete:
            return self.pass_dist + self.yac
        if self.interception:
            return self.pass_dist - self.return_yards
        if self.fumble:
            return self.pass_dist + self.yac - self.return_yards
        return 0 # Incomplete pass

    def __str__(self) -> str:
        """
        Formats a PassResult as a string
        """
        res = ""
        if self.pressure:
            res += "Defense brings pressure."
            if self.sack:
                res += f" SACKED for a loss of {self.sack_yards_lost} yards."
                return res
        is_deep = self.pass_dist > 15
        if len(res) > 0:
            res += " "
        if is_deep:
            res += f"Deep pass {self.pass_dist} yards downfield"
        else:
            res += f"Pass {self.pass_dist} yards downfield"
        if self.interception:
            res += f" INTERCEPTED and returned for {self.return_yards} yards."
            if self.touchdown:
                res += " TOUCHDOWN!"
            return res
        if self.complete:
            res += " complete"
            if self.fumble:
                res += f". FUMBLE recovered by the defense."
                if self.return_yards > 0:
                    res += f" Fumble was returned {self.return_yards} yards."
                if self.touchdown:
                    res += " TOUCHDOWN!"
                return res
            res += f" for a gain of {self.pass_dist + self.yac} yards."
            if self.touchdown:
                res += " TOUCHDOWN!"
            return res
        res += f" incomplete."
        return res.lstrip()
