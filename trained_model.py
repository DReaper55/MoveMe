import json

import numpy as np

from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the saved model
loaded_model = load_model('trained_model.keras')

# Load the mean and scale values from numpy files
scaler_mean = np.load('scaler_mean.npy')
scaler_scale = np.load('scaler_scale.npy')


# X = []
#
# # Load merged dataset from merged_dataset.json
# with open('synthetic_data.json', 'r') as merged_file:
#     merged_dataset = json.load(merged_file)
#
# # Flatten the nested structure to create a flat DataFrame
# flat_dataset = []
# for entry in merged_dataset['dataset']:
#     order = entry['order']
#     for driver in entry['drivers']:
#         features = [
#             driver["AccDistanceFromHereToPickup"],
#             driver["OnlineStatus"],
#             driver["CurrentOrders"],
#             order["OrderType"],
#             order["OrderPriority"],
#             order["DistanceToDropOff"],
#         ]
#         label = driver["ETA"]
#         X.append(features)
#
# X = np.array(X)

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale
# X = scaler.fit_transform(X)

# Use the loaded_model to make predictions
new_features = np.array([[10.8, 1, 1, 0, 0, 2.3]])

# new_features = scaler.fit_transform(new_features)

new_features = scaler.transform(new_features)
new_features = new_features.reshape((1, new_features.shape[1], 1))
predicted_eta = loaded_model.predict(new_features)
print(f'Predicted ETA for the new features: {predicted_eta[0][0]}')
