# A project of MIS student from Bicol University MIS 
# Scheduling Algorithm with tensorflow with the aid of AI support
# Author Jay Millena, Sir Ramil Berlon Cardaño, and Sir Marbert Panambo Plazo

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate synthetic data for training
# Features: task duration, machine capacity, etc.
# Labels: optimal schedule for each task

# Dummy data - replace this with your actual dataset
# Features
X_train = [[5, 10],
           [3, 8],
           [8, 5],
           [2, 12],
           ...]

# Labels
y_train = [[1, 0, 0],  # Binary representation of machine assignment
           [0, 1, 0],
           [0, 0, 1],
           [1, 0, 0],
           ...]

# Define the machine learning model
model = Sequential()

# Machine learning model
model.add(Dense(units=16, input_dim=2, activation="relu"))
model.add(Dense(units=8, activation="relu"))
model.add(Dense(units=3, activation="softmax"))

# Compile Model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer="adam",
              metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# After training, use the model for predictions
# You can use these predictions to make scheduling decisions

# Generate new task data for prediction
X_new_task = [[6, 9]]

# Make predictions
predictions = model.predict(X_new_task)

# Use tf.argmax instead of deprecated 'tf.argmax.numpy()'
selected_machine = tf.argmax(predictions, axis=1)[0]

# The task is assigned to the selected machine
print(f"The task is scheduled on machine {selected_machine}")

# Version of Tenserflow
# print(f"TensorFlow version: {tf.__version__}")