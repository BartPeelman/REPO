import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

# -------------------------------
# 1. Load and preprocess the data
# -------------------------------
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target.astype(np.int64)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
y_onehot = tf.keras.utils.to_categorical(y, num_classes=3)

# -------------------------------
# 2. Build a simple TensorFlow model
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 3. Train the model
# -------------------------------
model.fit(X, y_onehot, epochs=50, batch_size=16, verbose=0)

# -------------------------------
# 4. Export the model for Triton
# -------------------------------
model_path = "model_repository/example_model/1"
os.makedirs(model_path, exist_ok=True)

# Keras 3 way to export SavedModel
model.export(model_path)

print(f"Model exported to {model_path}")
