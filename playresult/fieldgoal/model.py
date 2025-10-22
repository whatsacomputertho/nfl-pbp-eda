import os
import pandas as pd
from context.context import PlayContext
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense
from keras.optimizers import Adam
from playresult.fieldgoal.result import FieldGoalResult
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill

WORKDIR = os.path.dirname(os.path.abspath(__file__))

class FieldGoalResultModel:
    def __init__(
            self,
            from_file: bool=False
        ) -> "FieldGoalResultModel":
        """
        Constructor for the FieldGoalResultModel class

        Args:
            from_file (bool): Whether to load from a pre-trained file
        
        Returns:
            FieldGoalResultModel: The constructed FieldGoalResultModel class
        """
        if from_file:
            self.load()
        else:
            model = Sequential()
            model.add(Dense(3, input_shape=(2,), activation='softmax'))
            model.compile(
                Adam(learning_rate=0.05),
                loss='categorical_crossentropy',
                metrics=[ "accuracy" ]
            )

    def play(
            self,
            offense: OffensiveSkill,
            defense: DefensiveSkill,
            context: PlayContext
        ) -> FieldGoalResult:
        """
        Generate a field goal result

        Args:
            offense (OffensiveSkill): The offensive skill levels
            defense (DefensiveSkill): The defensive skill levels
            context (PlayContext): The game context
        
        Returns:
            FieldGoalResult: The result of the field goal
        """
        prediction = self.model.predict(
            pd.DataFrame({
                "yardline_100": [context.yard_line],
                "norm_diff_field_goal_percent": [
                    (offense.field_goals - defense.field_goal_defense) / 2
                ]
            })
        )
        return FieldGoalResult.from_prediction(prediction[0])

    def save(
            self,
            path: str=f'{WORKDIR}/field_goal_result_v0.0.1-alpha.1.keras'
        ):
        """
        Save the underlying deep learning model to a keras file

        Args:
            path (str): The path to which to save the model
        """
        save_model(self.model, path)

    def load(
            self,
            path: str=f'{WORKDIR}/field_goal_result_v0.0.1-alpha.1.keras'
        ):
        """
        Load the underlying deep learning model from a keras file

        Args:
            path (str): The path from which to load the model
        """
        self.model = load_model(path)
