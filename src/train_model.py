import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load processed data
df = pd.read_csv('data/processed_data.csv')

# Split features and target
X = df.drop('tmsi', axis=1)
y = df['tmsi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

print("Training XGBoost Model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"Model Training Complete. Mean Absolute Error: {error:.4f}")

# Save the trained model
joblib.dump(model, 'output/rail_stress_model.pkl')
print("Model saved to output/rail_stress_model.pkl")