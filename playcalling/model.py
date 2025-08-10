import os
import pickle
import numpy as np
import pandas as pd
from context.context import GameContext
from playcalling.playcall import PlayCall
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

WORKDIR = os.path.dirname(os.path.abspath(__file__))

class PlayCallingModel:
    """
    Functionality for training and using a logistic regression model which
    calls plays given a game context / game scenarios
    """
    def __init__(
            self,
            from_file: bool=False
        ) -> "PlayCallingModel":
        """
        Constructor for the PlayCallingModel class

        Args:
            from_file (bool): Whether to load from a pre-trained file
        
        Returns:
            PlayCallingModel: The constructed PlayCallingModel class
        """
        if from_file:
            self.load()
        else:
            self.model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=5000
            )
    
    def play(self, context: GameContext) -> PlayCall:
        """
        Call a play using the model

        Args:
            context (GameContext): The play calling scenario
        
        Returns:
            PlayCall: The play call
        """
        res = self.model.predict(
            pd.DataFrame({
                "qtr": [context.quarter],
                "half_seconds_remaining": [context.half_seconds],
                "down": [context.down],
                "ydstogo": [context.distance],
                "yardline_100": [context.yard_line],
                "score_diff": [context.score_diff],
                "defteam_timeouts_remaining": [context.def_timeouts],
                "posteam_timeouts_remaining": [context.off_timeouts]
            })
        )
        return PlayCall.from_str(res[0])

    def train(
            self,
            train: pd.DataFrame
        ):
        """
        Train the underlying logistic regression model for calling football
        plays based on the game scenario

        Args:
            train (DataFrame): A training data set from load_clean_nfl_pbp_data
        """
        # Train the logistic regression model using the input dataframe
        self.model.fit(
            train[
                [
                    "qtr",
                    "half_seconds_remaining",
                    "down",
                    "ydstogo",
                    "yardline_100",
                    "score_diff",
                    "defteam_timeouts_remaining",
                    "posteam_timeouts_remaining"
                ]
            ],
            train[["play_type"]]
        )

    def test(
            self,
            test: pd.DataFrame
        ):
        """
        Test the underlying logistic regression model for calling football
        plays based on the game scenario, and print the results

        Args:
            test (DataFrame): A test date set from load_clean_nfl_pbp_data
        """
        pred = self.model.predict(
            test[
                [
                    "qtr",
                    "half_seconds_remaining",
                    "down",
                    "ydstogo",
                    "yardline_100",
                    "score_diff",
                    "defteam_timeouts_remaining",
                    "posteam_timeouts_remaining"
                ]
            ]
        )
        print(confusion_matrix(test[["play_type"]], pred))
        print(classification_report(test[["play_type"]], pred))

    def save(
            self,
            path: str=f'{WORKDIR}/model.pkl'
        ):
        """
        Save the underlying logistic regression model to a pickle file

        Args:
            path (str): The path to which to save the model
        """
        with open(path,'wb') as f:
            pickle.dump(self.model, f)

    def save_params(
            self,
            coef_path: str=f"{WORKDIR}/coefficients.csv",
            intercept_path: str=f"{WORKDIR}/intercept.csv"
        ):
        """
        Save the coefficients and intercepts of the underlying regression model
        to two CSV files

        Args:
            coef_path (str): The path to which to save the model coefficients
            intercept_path (str): The path to which to save the model intercepts
        """
        np.savetxt(
            coef_path,
            self.model.coef_,
            delimiter=","
        )
        np.savetxt(
            intercept_path,
            self.model.intercept_,
            delimiter=","
        )

    def load(
            self,
            path: str=f'{WORKDIR}/model.pkl'
        ):
        """
        Load the underlying logistic regression model from a pickle file

        Args:
            path (str): The path from which to load the model
        """
        with open(path,'rb') as f:
            self.model = pickle.load(f)
