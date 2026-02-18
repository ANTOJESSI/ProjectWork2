import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import os

# Ensure assets folder exists
if not os.path.exists('assets'):
    os.makedirs('assets')

# 1. Load the model
try:
    model = joblib.load('output/rail_stress_model.pkl')
    print("Model loaded successfully!")
except:
    print("Error: Train the model first by running train_model.py")
    exit()

# 2. Create the Feature Importance Plot
# This tells the 'story' of why the AI thinks a track is dangerous
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, ax=ax, height=0.7, grid=False, 
                   title="Key Risk Drivers for Track Buckling",
                   xlabel="F-Score (Importance Weight)", 
                   ylabel="Environmental & Physical Features")

# Improve layout
plt.tight_layout()

# 3. Save to assets for your paper
plt.savefig('assets/feature_importance.png', dpi=300) # dpi=300 is high quality for printing
print("Graph saved to assets/feature_importance.png")

# 4. Show the plot
plt.show()