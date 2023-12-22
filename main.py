import json

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load merged dataset from merged_dataset.json
with open('synthetic_data.json', 'r') as merged_file:
    merged_dataset = json.load(merged_file)

# Flatten the nested structure to create a flat DataFrame
flat_dataset = []
for entry in merged_dataset['dataset']:
    order = entry['order']
    for driver in entry['drivers']:
        flat_entry = {**order, **driver}
        flat_dataset.append(flat_entry)

# Convert the flat dataset to a Pandas DataFrame
df = pd.DataFrame(flat_dataset)

# Define features and target variable
features = ['OrderType', 'OrderPriority', 'DistanceToDropOff']
features += ['AccDistanceFromHereToPickup', 'ETA', 'OnlineStatus', 'CurrentOrders']
target = 'ETA'

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(train_df[features], train_df[target])

# Make predictions on the test set
predictions = model.predict(test_df[features])

# Evaluate the model performance
mse = mean_squared_error(test_df[target], predictions)
print(f'Mean Squared Error on Test Set: {mse}')

# Now, you can use the trained model to predict the delivery time for a new order
new_order = pd.DataFrame({
    'OrderType': [0],  # product
    'OrderPriority': [0],  # standard
    'DistanceToDropOff': [2.3],
    'AccDistanceFromHereToPickup': [10.8],
    'ETA': [12],
    'OnlineStatus': [1],  # moving
    'CurrentOrders': [1]
})

new_order_prediction = model.predict(new_order[features])
print(f'Predicted Timestamp for the New Order: {new_order_prediction}')
