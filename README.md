# ETA Prediction Model using CNN

This project implements a Convolutional Neural Network (CNN) to predict the Estimated Time of Arrival (ETA) based on features from driver and order data. The model is designed to predict ETA for drivers, utilizing various features related to orders and driver status.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Training the Model](#training-the-model)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview
The goal of this project is to predict the Estimated Time of Arrival (ETA) for a driver based on various features such as:
- **Order details**: Order type, order priority, distance to drop-off
- **Driver details**: Distance from pickup, online status, current orders

The model is built using a Convolutional Neural Network (CNN) with dropout and batch normalization layers to prevent overfitting and improve generalization.

---

## Dataset
The dataset is a JSON file containing a list of orders and associated driver information. Each entry includes:
- **Order**: Contains information such as order type, priority, and distance to drop-off.
- **Driver**: Includes features like distance to pickup, online status, current orders, and ETA (the target label).

Example dataset structure:
```json
{
  "dataset": [
    {
      "order": {
        "DistanceToDropOff": 5.5
      },
      "drivers": [
        {
          "AccDistanceFromHereToPickup": 10.8,
          "CurrentOrders": 2,
          "ETA": 15.4
        }
      ]
    }
  ]
}
```

---

## Model Architecture
The CNN model consists of:
1. **Input Layer**: Accepts a 1D array with 6 features per sample.
2. **Convolutional Layers**:
    - Two 1D convolutional layers with ReLU activation to extract patterns from the features.
    - Batch normalization for better training stability.
    - Dropout for regularization.
3. **Output Layer**: A dense layer with a linear activation function to predict the ETA.

---

## Requirements
The following dependencies are required to run the project:
- Python >= 3.7
- TensorFlow >= 2.8
- NumPy
- Matplotlib
- Scikit-learn
- JSON (for data manipulation)

Install dependencies with the following:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## Usage

### 1. Preparing the Dataset
Ensure the dataset is in the correct JSON format. Preprocess the data to extract features and labels:
```python
import json
import numpy as np

# Load dataset
with open('synthetic_data.json', 'r') as file:
    data = json.load(file)

# Flatten the dataset
features_data = []
labels_data = []

for entry in data['dataset']:
    order = entry['order']
    for driver in entry['drivers']:
        features = [
            driver["AccDistanceFromHereToPickup"],
            driver["CurrentOrders"],
            order["DistanceToDropOff"],
        ]
        label = driver["ETA"]
        features_data.append(features)
        labels_data.append(label)

X = np.array(features_data)
y = np.array(labels_data)
```

### 2. Training the Model
To train the model, use the following script:
```bash
python train_model.py
```

### 3. Evaluating the Model
To evaluate the trained model, run:
```bash
python evaluate_model.py
```

---

## Training the Model
The model is trained using the following steps:
1. Load and preprocess the dataset.
2. Normalize the features using `StandardScaler`.
3. Split the data into training and testing sets using `train_test_split`.
4. Build a CNN model with two convolutional layers and dropout/batch normalization.
5. Compile the model with the Adam optimizer and MAE (Mean Absolute Error) loss function.
6. Train the model using the training data.

Example training code:
```python
from keras.src.models import Sequential
from keras.src.layers import Conv1D, Flatten, Dense, BatchNormalization, InputLayer
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
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
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
```

---

## Evaluation
After training the model, evaluate its performance using the test data:
```python
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
```

---

## Results
The model achieves a **Mean Absolute Error (MAE)** on the test dataset. The training and validation losses are monitored to ensure the model is not overfitting.

Example output:
```bash
Test Loss: 4.53
```

The results are visualized using training vs validation loss plots.

---

## Contributing
Contributions are welcome! If you would like to improve the model or add new features:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---