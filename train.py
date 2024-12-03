import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Conv1D, Flatten, Dense, BatchNormalization, InputLayer
from keras.src.optimizers import Adam
from keras.src.losses import SparseCategoricalCrossentropy, MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import h5py


# Flatten the nested structure to create a flat DataFrame
flat_dataset = []
with h5py.File("dataset.h5", "r") as h5f:
    X = h5f["features"][:]
    y = h5f["labels"][:]

X = np.array(X)
y = np.array(y)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Save the scalar mean and scale
# np.save('scaler_mean.npy', scaler.mean_)
# np.save('scaler_scale.npy', scaler.scale_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Select the first feature for plotting
# X_train_feature = X_train[:, 2]

# Plot the feature against the target variable
# plt.figure(figsize=(10, 6))
# plt.scatter(X_train_feature, y_train, color='blue', alpha=0.5, label='Training Data')
# plt.xlabel("Feature 2 (e.g., OrderPriority)")
# plt.ylabel("ETA (Target Variable)")
# plt.title("Training Data: Feature vs. ETA")
# plt.legend()
# plt.show()

# Build the CNN model
model = Sequential([
    InputLayer(input_shape=(X_train.shape[1], 1)),
    Conv1D(32, kernel_size=2, activation='relu'),
    BatchNormalization(),
    Conv1D(64, kernel_size=2, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Single neuron output for regression
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=MeanSquaredError(),
              metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Plot training & validation loss values
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss (Gradient Descent)')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')
# plt.show()


# Save the entire model (architecture and weights)
# model.save('trained_model.h5')

# Make predictions for a new set of features
new_features = np.array([[47.0, 3, 8.99]])
new_features = scaler.transform(new_features)
new_features = new_features.reshape((1, new_features.shape[1], 1))
predicted_eta = model.predict(new_features)

print(f'Predicted ETA for the new features: {predicted_eta[0][0]}')
