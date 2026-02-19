"""
Model Comparison Script
Project: Rail Thermal Buckling Prediction

Purpose:
- Compare classical ML and DL models
- Predict Thermal Misalignment Stress Index (TMSI)
- Offline evaluation (independent of Streamlit visualization)

Dataset:
- data/processed_data.csv
"""

# ============================
# Imports
# ============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ML Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Deep Learning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# ============================
# 1. Load Dataset
# ============================

print("📥 Loading processed dataset...")

df = pd.read_csv("data/processed_data.csv")

X = df.drop("tmsi", axis=1)
y = df["tmsi"]

print(f"✅ Dataset loaded with {df.shape[0]} samples and {df.shape[1]} columns")


# ============================
# 2. Train-Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("🔀 Train-test split completed")


# ============================
# 3. Scaling (ONLY for DL)
# ============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================
# 4. Models to Compare
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
        random_state=42,
        objective="reg:squarederror"
    ),

    "Deep Learning (MLP)": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=42
    )
}


# ============================
# 5. Train & Evaluate
# ============================

results = []

print("\n🔍 Comparing Models...\n")

for name, model in models.items():

    print(f"➡️ Training {name}...")

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

    print(f"   ✅ MAE: {mae:.4f}")


# ============================
# 6. Results Table
# ============================

results_df = pd.DataFrame(results).sort_values(by="MAE")

print("\n📊 Model Comparison Results (Lower MAE is Better):\n")
print(results_df)


# ============================
# 7. Publication-Quality Visualization (IEEE Style)
# ============================

plt.figure(figsize=(10, 5))

bars = plt.bar(
    results_df["Model"],
    results_df["MAE"],
    color=["#70818C", "#A4B5BF", "#B0BFAE", "#D9BB84", "#D9A79C"],
    # color=["#5E7B4C","#A9C77D","#D9C9A2","#775A3A"],
    # color=["#1E319E","#6D7BC7","#E6E7ED","#DACB9D","#B89B47"],

    edgecolor="black"
)

plt.xlabel("Prediction Models", fontsize=11)
plt.ylabel("Mean Absolute Error (MAE)", fontsize=11)
plt.title(
    "Comparison of Machine Learning Models\nfor Rail Thermal Buckling Prediction",
    fontsize=13,
    fontweight="bold"
)

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# ---- Annotate MAE values on bars ----
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.0002,
        f"{height:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

# ---- Save high-resolution figure for IEEE paper ----
plt.savefig(
    "output/model_comparison_mae.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
