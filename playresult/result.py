import copy
import json
from context.context import GameContext, PlayContext
from typing import Any

class PlayResult:
    def __init__(
            self,
            play_duration: int,
            yards_gained: int,
            first_down: bool,
            touchdown: bool,
            complete_pass: bool,
            out_of_bounds: bool,
            qb_scramble: bool,
            qb_hit: bool,
            sack: bool,
            tackle_for_loss: bool,
            fumble: bool,
            interception: bool,
            field_goal_made: bool,
            field_goal_missed: bool,
            field_goal_blocked: bool,
            penalty: bool,
            posteam_penalty: bool,
            penalty_yards: int,
            timeout: bool,
            posteam_timeout: bool
        ) -> "PlayResult":
        """
        Constructor for the PlayResult class

        Args:
            play_duration (int): The duration of the play in game clock seconds
            yards_gained (int): The yards gained on the play
            first_down (bool): Whether the play resulted in a first down
            touchdown (bool): Whether the play resulted in a touchdown
            complete_pass (bool): Whether the play resulted in a completed pass
            out_of_bounds (bool): Whether the player went out of bounds
            qb_scramble (bool): Whether the quarterback scrambled on the play
            qb_hit (bool): Whether the quarterback was hit on the play
            sack (bool): Whether the play resulted in a sack
            tackle_for_loss (bool): Whether a tackle for loss occurred
            fumble (bool): Whether the play resulted in a fumble
            interception (bool): Whether the play resulted in an interception
            field_goal_made (bool): Whether a field goal was made
            field_goal_missed (bool): Whether a field goal was missed
            field_goal_blocked (bool): Whether a field goal was blocked
            penalty (bool): Whether a penalty occurred on the play
            posteam_penalty (bool): Whether the offense committed a penalty
            penalty_yards (int): The penalty yards enforced on the play
            timeout (bool): Whether a timeout was called after the play
            posteam_timeout (bool): Whether the offense called a timeout
        
        Returns:
            PlayResult: The instantiated PlayResult instance
        """
        # Validate that the integer arguments are in-bounds
        if play_duration > 1800 or play_duration < 0:
            raise ValueError(
                f"Play duration out of bounds (0 - 1800): {play_duration}"
            )
        if yards_gained > 100 or yards_gained < -100:
            raise ValueError(
                f"Yards gained out of bounds (-100 - 100): {yards_gained}"
            )
        if penalty_yards > 100 or penalty_yards < -100:
            raise ValueError(
                f"Penalty yards out of bounds (-100 - 100): {penalty_yards}"
            )
        
        # Validate mutually-exclusive boolean arguments
        if  (field_goal_made and field_goal_missed) or \
            (field_goal_missed and field_goal_blocked) or \
            (field_goal_blocked and field_goal_made):
            raise ValueError(
                "Field goal result flags are mutually exclusive, but got:" + \
                f" (made) {field_goal_made}, (missed) {field_goal_missed}," + \
                f" (blocked) {field_goal_blocked}"
            )

        # Save as object properties
        self.play_duration = play_duration
        self.yards_gained = yards_gained
        self.first_down = first_down
        self.touchdown = touchdown
        self.complete_pass = complete_pass
        self.out_of_bounds = out_of_bounds
        self.qb_scramble = qb_scramble
        self.qb_hit = qb_hit
        self.sack = sack
        self.tackle_for_loss = tackle_for_loss
        self.fumble = fumble
        self.interception = interception
        self.field_goal_made = field_goal_made
        self.field_goal_missed = field_goal_missed
        self.field_goal_blocked = field_goal_blocked
        self.penalty = penalty
        self.posteam_penalty = posteam_penalty
        self.penalty_yards = penalty_yards
        self.timeout = timeout
        self.posteam_timeout = posteam_timeout

    @staticmethod
    def from_prediction(prediction: list[float]) -> "PlayResult":
        """
        Given a prediction from the PlayResultModel.play method, this method
        instantiates a PlayResult instance by rounding the prediction's
        properties to the nearest integers

        Args:
            prediction (list): The prediction from PlayResultModel.play
        
        Returns:
            PlayResult: The instantiated PlayResult
        """
        # Validate the length of the list
        num_elements = len(prediction)
        if num_elements != 20:
            raise ValueError(
                f"Expected a list of 20 floats, got: {num_elements}"
            )
        
        # Round each list element to the nearest integer
        rounded = []
        for element in prediction:
            rounded.append(round(element))
        
        # Instantiate and return the PlayResult
        return PlayResult(
            play_duration=rounded[0],
            yards_gained=rounded[1],
            first_down=bool(rounded[2]),
            touchdown=bool(rounded[3]),
            complete_pass=bool(rounded[4]),
            out_of_bounds=bool(rounded[5]),
            qb_scramble=bool(rounded[6]),
            qb_hit=bool(rounded[7]),
            sack=bool(rounded[8]),
            tackle_for_loss=bool(rounded[9]),
            fumble=bool(rounded[10]),
            interception=bool(rounded[11]),
            field_goal_made=bool(rounded[12]),
            field_goal_missed=bool(rounded[13]),
            field_goal_blocked=bool(rounded[14]),
            penalty=bool(rounded[15]),
            posteam_penalty=bool(rounded[16]),
            penalty_yards=rounded[17],
            timeout=bool(rounded[18]),
            posteam_timeout=bool(rounded[19])
        )

    def next_context(self, context: GameContext) -> GameContext:
        """
        Derive the next game context based on the current context and the play
        result

        Args:
            context (GameContext): The current game context
        
        Returns:
            PlayContext: The next game context
        """
        new_context = copy.deepcopy(context)

        # TODO: Check if a change of possession occurred

        # If not, then update the clock and yard line
        new_context.update_clock(self.play_duration)
        new_context.update_yard_line(self.yards_gained)
        return new_context

    def __json__(self) -> dict[str, Any]:
        """
        Encodes the play result as JSON

        Returns:
            dict: The PlayResult as a JSON-serializable dictionary
        """
        return {
            "play_duration": self.play_duration,
            "yards_gained": self.yards_gained,
            "first_down": self.first_down,
            "touchdown": self.touchdown,
            "complete_pass": self.complete_pass,
            "out_of_bounds": self.out_of_bounds,
            "qb_scramble": self.qb_scramble,
            "qb_hit": self.qb_hit,
            "sack": self.sack,
            "tackle_for_loss": self.tackle_for_loss,
            "fumble": self.fumble,
            "interception": self.interception,
            "field_goal_made": self.field_goal_made,
            "field_goal_missed": self.field_goal_missed,
            "field_goal_blocked": self.field_goal_blocked,
            "penalty": self.penalty,
            "posteam_penalty": self.posteam_penalty,
            "penalty_yards": self.penalty_yards,
            "timeout": self.timeout,
            "posteam_timeout": self.posteam_timeout
        }

    def __str__(self) -> str:
        """
        Displays the play result as a human-readable string

        Returns:
            str: The PlayResult as a human-readable string
        """
        # TODO: Improve
        return json.dumps(self.__json__())
