import numpy as np
import pandas as pd  # Import pandas to load the dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load your dataset here (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv('creditcard.csv')  # Ensure your dataset is in CSV format

# Define X_train and X_test
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

# Build the autoencoder model
input_layer = Input(shape=(X_train.shape[1],))
encoded = Dense(14, activation='relu')(input_layer)
encoded = Dense(7, activation='relu')(encoded)
decoded = Dense(14, activation='relu')(encoded)
decoded = Dense(X_train.shape[1], activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Use the autoencoder to detect fraud (high reconstruction error indicates fraud)
reconstructed = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructed, 2), axis=1)
threshold = np.percentile(mse, 95)  # Set threshold for anomalies
fraud = mse > threshold
print(f"Detected fraud cases: {sum(fraud)}")
