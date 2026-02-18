import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load processed data
df = pd.read_csv('data/processed_data.csv')

# Split features and target
X = df.drop('tmsi', axis=1)
y = df['tmsi']

# Train-test split (same as XGBoost)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Neural Networks NEED feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build MLP model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1)  # Regression output
])

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='mae'
)

print("Training Deep Learning Model...")
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate
predictions = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, predictions)

print(f"Deep Learning Model MAE: {mae:.4f}")
