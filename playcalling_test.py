from data.pbp import load_clean_nfl_pbp_playcall_data
from playcalling.model import PlayCallingModel
from sklearn.model_selection import train_test_split

# Load the NFL data and split into training and test data
print("Loading NFL play-by-play data")
df = load_clean_nfl_pbp_playcall_data()
train, test = train_test_split(df, random_state=337)

# Load the playcalling model
print("Loading the playcalling model")
model = PlayCallingModel(from_file=True)

# Test the playcalling model
print("Testing the playcalling model")
model.test(test)
