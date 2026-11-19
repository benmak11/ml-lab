import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = 'data/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

# Drop houses with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

# In-sample score: train and evaluate on the same houses (the mistake)
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

predicted_home_prices = melbourne_model.predict(X)
print("In-sample MAE:", mean_absolute_error(y, predicted_home_prices))

# Split data into training and validation data, for both features and target.
# The split is random; random_state makes it the same every run.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define and fit the model on the training data only
melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(train_X, train_y)

# Get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print("Validation MAE:", mean_absolute_error(val_y, val_predictions))

# For scale: the average price of the houses being predicted
print("Average validation price:", val_y.mean())
