import json
import os

import numpy as np
import sys

import tensorflow as tf
from keras.api.models import load_model
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.realpath(__file__))


def predict_eta(features):
    # Get files using absolute path
    # model_path = "D:/Development/Web Projects/moveme/moveme/src/main/resources/python/trained_model.keras"
    model_path = os.path.join(current_dir, 'trained_model.keras')

    scalar_mean_path = os.path.join(current_dir, 'scaler_mean.npy')
    # scalar_mean_path = "D:/Development/Web Projects/moveme/moveme/src/main/resources/python/scaler_mean.npy"

    scalar_scale_path = os.path.join(current_dir, 'scaler_scale.npy')
    # scalar_scale_path = "D:/Development/Web Projects/moveme/moveme/src/main/resources/python/scaler_scale.npy"

    # Load the saved model
    loaded_model = load_model(model_path)

    # Load the mean and scale values from numpy files
    scaler_mean = np.load(scalar_mean_path)
    scaler_scale = np.load(scalar_scale_path)

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale

    # Use the loaded_model to make predictions
    new_features = np.array([features])

    # Transform the new features using the loaded scaler
    new_features = scaler.transform(new_features)
    new_features = new_features.reshape((1, new_features.shape[1], 1))

    # Make predictions with the loaded model
    predicted_eta = loaded_model.predict(new_features)

    return predicted_eta[0][0]


if __name__ == "__main__":
    predicted_eta_result = predict_eta([23, 1, 7.58])
    print({"eta": predicted_eta_result})

    # Read arguments from the command line
    # try:
    #     input_data = sys.argv[1].replace("'", "\"")
    #     input_data = json.loads(input_data)
    #
    #     new_features_input = input_data.get("features", [])
    #     predicted_eta_result = predict_eta(new_features_input)
    #     print({"eta": predicted_eta_result})
    # except (IndexError, json.JSONDecodeError):
    #     print("Error: Invalid or missing input data.")
    #     sys.exit(1)
