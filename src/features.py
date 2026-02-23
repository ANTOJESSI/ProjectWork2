import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import os

# 1. Load your trained model
# Ensure the path matches where your model is saved
model_path = 'output/rail_stress_model.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully.")
else:
    print(f"Error: Model not found at {model_path}. Please run your training script first.")
    exit()

# 2. Set up the plotting environment
# Adjusting figure size for better readability in a research paper
plt.figure(figsize=(10, 6))

# 3. Generate the Feature Importance Plot
# 'importance_type' can be 'weight', 'gain', or 'cover'
# 'weight' (default): Number of times a feature appears in a tree
# 'gain': Average gain of splits which use the feature
xgb.plot_importance(model, 
                   importance_type='weight', 
                   max_num_features=10, 
                   height=0.8, 
                   color='skyblue', 
                   edgecolor='black')

# 4. Customize labels and title for a professional look
plt.title("Key Risk Drivers for Track Buckling (Feature Importance)", fontsize=14, fontweight='bold')
plt.xlabel("F-Score (Importance Weight)", fontsize=12)
plt.ylabel("Environmental and Physical Features", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# 5. Save the figure to your assets folder
if not os.path.exists('assets'):
    os.makedirs('assets')

save_path = 'assets/feature_importance.png'
plt.savefig(save_path, dpi=300)  # High resolution for printing
print(f"Graph saved as {save_path}")

# 6. Display the graph
plt.show()