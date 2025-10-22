import numpy as np
from enum import Enum

class FieldGoalResult(Enum):
    MADE = 0
    MISSED = 1
    BLOCKED = 2

    @staticmethod
    def from_prediction(result: list[float]) -> "FieldGoalResult":
        """
        Initializes a field goal result from the output probability vector
        of the field goal result model

        Args:
            result (list[float]): The field goal result probabilities
        
        Returns:
            FieldGoalResult: The field goal result
        """
        return np.random.choice(
            [
                FieldGoalResult.MADE,
                FieldGoalResult.MISSED,
                FieldGoalResult.BLOCKED
            ],
            size=1,
            p=result
        )[0]
