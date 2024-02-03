import json

import numpy as np
from keras.layers import LSTM, Dense, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load merged dataset from merged_dataset.json
with open('synthetic_data.json', 'r') as merged_file:
    merged_dataset = json.load(merged_file)

X = []
y = []

# Flatten the nested structure to create a flat DataFrame
flat_dataset = []
for entry in merged_dataset['dataset']:
    order = entry['order']
    for driver in entry['drivers']:
        features = [
            driver["AccDistanceFromHereToPickup"],
            driver["OnlineStatus"],
            driver["CurrentOrders"],
            order["OrderType"],
            order["OrderPriority"],
            order["DistanceToDropOff"],
        ]
        label = driver["ETA"]
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scalar mean and scale
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the RNN model
model = Sequential()
model.add(LSTM(32, input_shape=(X_train.shape[1], 1), activation='relu', return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Reshape the input for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Set: {loss}')

# Save the entire model (architecture and weights)
model.save('trained_model.keras')
# model.save('tm')

# Make predictions for a new set of features
new_features = np.array([[10.8, 1, 1, 0, 0, 2.3]])
new_features = scaler.transform(new_features)
new_features = new_features.reshape((1, new_features.shape[1], 1))
predicted_eta = model.predict(new_features)

print(f'Predicted ETA for the new features: {predicted_eta[0][0]}')
