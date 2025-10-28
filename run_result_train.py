import numpy as np
import pandas as pd
from data.pbp import load_clean_nfl_pbp_run_data
from keras.models import Sequential, save_model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report

# Load rushing data
df = pd.read_csv("./data/rushing.csv")

# Prepare the dataset features
x = df[
    [
        # Game context
        "qtr",
        "half_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",
        "defteam_timeouts_remaining",
        "posteam_timeouts_remaining",
        "score_diff",
        "goal_to_go",

        # Team skill levels and skill diffs
        "norm_diff_rushing",
        "norm_diff_blocking_blitzing",
        "norm_diff_ball_handling",
        "norm_rushing_penalties",
        "norm_rush_defense_penalties"
    ]
]
y = df[
    [
        # Rush result
        "play_duration",
        "yards_gained",
        "penalty",
        "posteam_penalty",
        "fumble"
    ]
]

# Prepare training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Prepare and train the model
model = Sequential()
model.add(Dense(16, input_shape=(14,), activation='softplus'))
model.add(Dense(2, activation='softplus'))
model.compile(
    Adam(learning_rate=0.01),
    loss=['mean_squared_error']
)
model.fit(
    x_train[
        [
            # Game context
            "qtr",
            "half_seconds_remaining",
            "down",
            "ydstogo",
            "yardline_100",
            "defteam_timeouts_remaining",
            "posteam_timeouts_remaining",
            "score_diff",
            "goal_to_go",

            # Team skill levels and skill diffs
            "norm_diff_rushing",
            "norm_diff_blocking_blitzing",
            "norm_diff_ball_handling",
            "norm_rushing_penalties",
            "norm_rush_defense_penalties"
        ]
    ],
    y_train[
        [
            # Rush result numeric outputs
            "play_duration",
            "yards_gained"
        ]
    ],
    epochs=5
)

# Test the model
y_pred_num = model.predict(x_test)
save_model(model, "./playresult/rushing/rush_result_v0.0.1-alpha.1.keras")

# Evaluate the mean squared error of the model
mse = mean_squared_error(
    y_test[
        [
            # Rush result numeric outputs
            "play_duration",
            "yards_gained"
        ]
    ].values,
    y_pred_num
)
print(f"Mean squared error: {mse}")
