import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ML Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# DL Model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ============================
# 1. Load Dataset
# ============================
df = pd.read_csv('data/processed_data.csv')

X = df.drop('tmsi', axis=1)
y = df['tmsi']

# Same split for ALL models (fair comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# 2. Scale data for DL only
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# 3. Models to Compare
# ============================
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ),

    "Deep Learning (MLP)": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=42
    )
}

# ============================
# 4. Train & Evaluate
# ============================
results = []

print("\n🔍 Comparing Models...\n")

for name, model in models.items():

    # DL uses scaled data
    if "Deep Learning" in name:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)

    results.append({
        "Model": name,
        "MAE": round(mae, 4)
    })

# ============================
# 5. Show Results
# ============================
results_df = pd.DataFrame(results).sort_values(by="MAE")

print("\n📊 Model Comparison Results (Lower MAE is Better):\n")
print(results_df)


import matplotlib.pyplot as plt

# ============================
# 6. Plot Comparison Graph
# ============================
plt.figure(figsize=(10, 5))
plt.bar(results_df["Model"], results_df["MAE"])
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Model Comparison for Rail Buckling Prediction")
plt.xticks(rotation=30)
plt.grid(axis='y')

plt.tight_layout()
plt.show()
