import os
import pandas as pd
from context.context import GameContext
from keras.layers import Input, Dense
from keras.models import Model, load_model, save_model
from playcalling.playcall import PlayCall
from playresult.result import PlayResult
from team.offense import OffensiveSkill
from team.defense import DefensiveSkill

WORKDIR = os.path.dirname(os.path.abspath(__file__))

class PlayResultModel:
    """
    Functionality for training and using a deep learning model which generates
    a result of a play given each team skill, the game context, and a play call
    """
    def __init__(
            self,
            from_file: bool=False
        ) -> "PlayResultModel":
        """
        Constructor for the PlayResultModel class

        Args:
            from_file (bool): Whether to load from a pre-trained file
        
        Returns:
            PlayResultModel: The constructed PlayResultModel class
        """
        if from_file:
            self.load()
        else:
            input_layer = Input(shape=(35,), name='input_features')
            shared_hidden = Dense(64, activation='relu')(input_layer)
            shared_hidden = Dense(32, activation='relu')(shared_hidden)
            output_layer = Dense(20, name='output_features')(shared_hidden)
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(
                optimizer='adam',
                loss={"output_features": "mean_squared_error"},
                metrics={"output_features": "mae"}
            )
            self.model = model

    def play(
            self,
            offense: OffensiveSkill,
            defense: DefensiveSkill,
            context: GameContext,
            playcall: PlayCall
        ) -> PlayResult:
        """
        Generate a play result

        Args:
            offense (OffensiveSkill): The offensive skill levels
            defense (DefensiveSkill): The defensive skill levels
            context (GameContext): The game context
            playcall (PlayCall): The play call
        
        Returns:
            PlayResult: The result of the play
        """
        one_hot_playcall = playcall.to_one_hot()
        prediction = self.model.predict(
            pd.DataFrame({
                # Game context
                "qtr": [context.quarter],
                "half_seconds_remaining": [context.half_seconds],
                "down": [context.down],
                "ydstogo": [context.distance],
                "yardline_100": [context.yard_line],
                "goal_to_go": [context.goal_to_go],
                "score_diff": [context.score_diff],
                "defteam_timeouts_remaining": [context.def_timeouts],
                "posteam_timeouts_remaining": [context.off_timeouts],

                # Play call (one-hot encoded)
                "play_type_short_pass": [one_hot_playcall["play_type_short_pass"]],
                "play_type_deep_pass": [one_hot_playcall["play_type_deep_pass"]],
                "play_type_run_left": [one_hot_playcall["play_type_run_left"]],
                "play_type_run_middle": [one_hot_playcall["play_type_run_middle"]],
                "play_type_run_right": [one_hot_playcall["play_type_run_right"]],
                "play_type_kickoff": [one_hot_playcall["play_type_kickoff"]],
                "play_type_punt": [one_hot_playcall["play_type_punt"]],
                "play_type_extra_point": [one_hot_playcall["play_type_extra_point"]],
                "play_type_field_goal": [one_hot_playcall["play_type_field_goal"]],
                "play_type_qb_kneel": [one_hot_playcall["play_type_qb_kneel"]],
                "play_type_qb_spike": [one_hot_playcall["play_type_qb_spike"]],
                "play_type_offense_timeout": [one_hot_playcall["play_type_offense_timeout"]],
                "play_type_defense_timeout": [one_hot_playcall["play_type_defense_timeout"]],

                # Offense skill
                "norm_blocking": [offense.blocking],
                "norm_rushing": [offense.rushing],
                "norm_passing": [offense.passing],
                "norm_receiving": [offense.receiving],
                "norm_scrambling": [offense.scrambling],
                "norm_offensive_turnovers": [offense.turnovers],
                "norm_offensive_penalties": [offense.penalties],

                # Defense skill
                "norm_blitzing": [defense.blitzing],
                "norm_rush_defense": [defense.rush_defense],
                "norm_pass_defense": [defense.pass_defense],
                "norm_coverage": [defense.coverage],
                "norm_defensive_turnovers": [defense.turnovers],
                "norm_defensive_penalties": [defense.penalties],
            })
        )
        return PlayResult.from_prediction(prediction[0])

    def save(
            self,
            path: str=f'{WORKDIR}/playresult_v0.0.1-alpha.1.keras'
        ):
        """
        Save the underlying logistic regression model to a pickle file

        Args:
            path (str): The path to which to save the model
        """
        save_model(self.model, path)

    def load(
            self,
            path: str=f'{WORKDIR}/playresult_v0.0.1-alpha.1.keras'
        ):
        """
        Load the underlying deep learning model from a keras file

        Args:
            path (str): The path from which to load the model
        """
        self.model = load_model(path)
