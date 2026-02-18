# This is feature engineering + preprocessing.

import pandas as pd
import numpy as np
import os

def prepare_rail_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # 1. Engineering Constants for Rail Steel
    E = 210000  # Young's Modulus (N/mm2)
    alpha = 11.5e-6  # Thermal Expansion Coefficient
    neutral_temp = 35  # Standard Neutral Temp in India (°C)
    
    # 2. Temperature Conversion (Visual Crossing uses Fahrenheit)
    df['temp_c'] = (df['temp'] - 32) * 5/9
    
    # 3. Calculate Rail Temperature
    # In direct sun, rail temp is typically Ambient + 15°C
    df['rail_temp'] = df['temp_c'] + 15
    
    # 4. Physics-Based Stress Calculation (MPa) MegaPascal
    # Stress = E * alpha * (Current_Temp - Neutral_Temp)
    df['stress_mpa'] = E * alpha * (df['rail_temp'] - neutral_temp)
    df['stress_mpa'] = df['stress_mpa'].clip(lower=0) # Only compression causes buckling
    
    # 5. Generate the TMSI (Target Variable)
    # We combine Thermal Stress (70%) with a simulated 'Track Age' factor (30%)
    # Random number between 5 and 40 years
    #     Why divide by 150?

    # 150 MPa ≈ critical stress for buckling

    # This normalizes stress to 0–1 range example:75 MPa → 75/150 = 0.5

    df['track_age'] = np.random.randint(5, 40, size=len(df)) 

    # TMSI = 0.7 × thermal stress factor + 0.3 × age factor
    # 📌 This is NOT machine learning
    #  📌 This is rule-based risk index generation
    df['tmsi'] = (df['stress_mpa'] / 150 * 0.7) + (df['track_age'] / 40 * 0.3)
    df['tmsi'] = df['tmsi'].clip(0, 1) # Keep index between 0 and 1
    
    # Keep only the features we need for training
    features = ['temp_c', 'humidity', 'solarradiation', 'track_age', 'tmsi']
    return df[features]

# Process both your uploaded files
print("Processing datasets...")
data1 = prepare_rail_data('data/india_2023_2026.csv')
data2 = prepare_rail_data('data/feb26_feb27.csv')

# Combine and save
final_df = pd.concat([data1, data2], ignore_index=True)
final_df.to_csv('data/processed_data.csv', index=False)
print(f"Success! Processed {len(final_df)} rows. Saved to data/processed_data.csv")



# After concat (without ignore_index):

# index: 0,1,2,0,1   ❌ (duplicate indexes)

# ignore_index=True ✅
# index: 0,1,2,3,4


# Why we use index=False?

# ML models don’t need row numbers

# Avoids confusion

# Cleaner dataset