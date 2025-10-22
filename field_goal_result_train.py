import numpy as np
import pandas as pd
import seaborn as sns
from data.pbp import load_clean_nfl_pbp_fieldgoal_data
from keras.models import Sequential, save_model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("./data/fgs.csv")

# Prepare the dataset features
x = df.drop(
    [
        "field_goal_result",
        "desc"
    ],
    axis=1
)
y = pd.get_dummies(df, columns=['field_goal_result'])
y = y.drop(
    [
        "yardline_100",
        "norm_diff_field_goal_percent",
        "field_goal_attempt",
        "desc"
    ],
    axis=1
)

# Prepare training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Prepare the model
model = Sequential()
model.add(Dense(3, input_shape=(2,), activation='softmax'))
model.compile(
    Adam(learning_rate=0.05),
    loss='categorical_crossentropy',
    metrics=[ "accuracy" ]
)
model.fit(
    x_train[
        [
            "yardline_100",
            "norm_diff_field_goal_percent"
        ]
    ],
    y_train[
        [
            "field_goal_result_made",
            "field_goal_result_missed",
            "field_goal_result_blocked"
        ]
    ],
    epochs=5
)

# Test the model
save_model(model, "./playresult/fieldgoal/field_goal_result_v0.0.1-alpha.1.keras")
y_pred = model.predict(
    x_test[
        [
            "yardline_100",
            "norm_diff_field_goal_percent"
        ]
    ]
)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test[
    [
        "field_goal_result_made",
        "field_goal_result_missed",
        "field_goal_result_blocked"
    ]
].values, axis=1)
print(
    classification_report(
        y_test_class,
        y_pred_class
    )
)
