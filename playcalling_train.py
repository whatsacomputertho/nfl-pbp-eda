import os
from data.pbp import load_clean_nfl_pbp_data
from playcalling.model import PlayCallingModel
from sklearn.model_selection import train_test_split

WORKDIR = os.path.dirname(os.path.abspath(__file__))

# Load the NFL data and split into training and test data
print("Loading NFL play-by-play data")
df = load_clean_nfl_pbp_data()
train, test = train_test_split(df)

# Train the logistic regression model
print("Training the playcalling model")
model = PlayCallingModel()
model.train(train)

# Save the model and the model parameters
print("Saving the playcalling model")
model.save()
model.save_params()
