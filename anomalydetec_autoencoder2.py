import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# Generate synthetic data for normal transactions
normal_data = np.random.normal(0, 1, (10000, 10))
anomalous_data = np.random.normal(5, 1, (100, 10))

# Combine and label the data
X = np.vstack([normal_data, anomalous_data])
y = np.array([0] * len(normal_data) + [1] * len(anomalous_data))  # 0 = normal, 1 = anomaly

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define Autoencoder
input_dim = X_train.shape[1]
autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(input_dim, activation='linear')  # Reconstruct original input
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Train Autoencoder on normal data only
autoencoder.fit(X_train[y_train == 0], X_train[y_train == 0], epochs=50, batch_size=32, validation_split=0.2)

# Compute reconstruction error
reconstruction = autoencoder.predict(X_test)
reconstruction_error = np.mean((reconstruction - X_test) ** 2, axis=1)

# Set a threshold
threshold = np.percentile(reconstruction_error[y_test == 0], 95)  # 95th percentile of normal errors

# Classify anomalies
y_pred = (reconstruction_error > threshold).astype(int)

# Evaluate
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
