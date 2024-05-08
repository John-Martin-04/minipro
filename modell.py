import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define your custom training data
X_train_custom = np.array([[1], [2], [3], [4], [5], [10], [15], [20], [25], [30]])  # Input integers
y_train_custom = np.array([[5], [6], [6], [7], [15], [22], [25], [35], [45], [60]])  # Output integers

# Define the model architecture
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with your custom data
model.fit(X_train_custom, y_train_custom, epochs=10, batch_size=1)

# Now you can use this model to predict output integers for new input integers
# For example:
input_integer = np.array([[7], [25], [45]])  # Example input integers
predicted_output = model.predict(input_integer)
print(predicted_output)
