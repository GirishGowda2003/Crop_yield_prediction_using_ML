"""
train_model.py - Train Crop Yield Prediction Model
Matches exact dataset headers: State, District , Crop, Crop_Year, Season, Area , Production, Yield, Annual_Rainfall
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🌾 CROP YIELD PREDICTION MODEL TRAINING")
print("   Dataset Headers: State, District , Crop, Crop_Year, Season, Area , Production, Yield, Annual_Rainfall")
print("="*70)

# Load data
print("\n📁 Loading Dataset...")
try:
    df = pd.read_csv('agridata.csv')
    print(f"✅ Dataset loaded: {len(df):,} records")
    
    # --- CRITICAL FIX: Clean Column Names ---
    # Your dataset has 'District ' and 'Area ' (with spaces). 
    # This line removes those spaces to make them 'District' and 'Area'.
    df.columns = df.columns.str.strip()
    print(f"✅ Cleaned Columns: {list(df.columns)}")
    
except FileNotFoundError:
    print("❌ Error: 'agridata.csv' not found!")
    exit()

# Check required columns
required_cols = ['State', 'District', 'Crop', 'Season', 'Area', 'Crop_Year', 'Production', 'Annual_Rainfall']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"\n❌ Error: Missing required columns: {missing_cols}")
    print("   Note: The script attempts to strip spaces from headers automatically.")
    exit()

# Handle Annual_Rainfall (Hierarchical Imputation)
print(f"\n🌧️ Processing Rainfall Data...")
# 1. Fill with District Mean
df['Annual_Rainfall'] = df['Annual_Rainfall'].fillna(
    df.groupby(['State', 'District'])['Annual_Rainfall'].transform('mean')
)
# 2. Fill with State Mean
df['Annual_Rainfall'] = df['Annual_Rainfall'].fillna(
    df.groupby('State')['Annual_Rainfall'].transform('mean')
)
# 3. Fill with Global Mean
df['Annual_Rainfall'] = df['Annual_Rainfall'].fillna(df['Annual_Rainfall'].mean())

print(f"✅ Rainfall data imputed. Range: {df['Annual_Rainfall'].min():.0f} - {df['Annual_Rainfall'].max():.0f} mm")

# Remove missing values
print(f"\n🧹 Cleaning data...")
df = df.dropna(subset=required_cols)

# Convert Area to acres (Assuming input 'Area' is Hectares? Or if it's already Acres, adjust factor)
# Standard Indian datasets often use Hectares. If your 'Area ' is Hectares:
df['Area_Acres'] = df['Area'] * 2.471 
# If your 'Area ' is already Acres, change to: df['Area_Acres'] = df['Area']

# Calculate Yield (if needed, or use provided 'Yield' column)
# We will trust your calculated 'Yield' column but convert to q/acre for uniformity
# Assuming your 'Yield' column is Tonnes/Hectare (common) or Kg/Hectare.
# Let's standardize on calculating fresh to be safe:
df['Yield_kg_ha'] = (df['Production'] * 1000) / df['Area']  # Production in Tonnes -> kg
df['Yield_Q_Acre'] = df['Yield_kg_ha'] / 247.1 # Convert kg/ha -> quintals/acre

# Remove outliers
Q1 = df['Yield_Q_Acre'].quantile(0.25)
Q3 = df['Yield_Q_Acre'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Yield_Q_Acre'] >= Q1 - 1.5 * IQR) & (df['Yield_Q_Acre'] <= Q3 + 1.5 * IQR)]

# Encode categorical variables
le_state = LabelEncoder()
le_district = LabelEncoder()
le_crop = LabelEncoder()
le_season = LabelEncoder()

df['State_Encoded'] = le_state.fit_transform(df['State'])
df['District_Encoded'] = le_district.fit_transform(df['District'])
df['Crop_Encoded'] = le_crop.fit_transform(df['Crop'])
df['Season_Encoded'] = le_season.fit_transform(df['Season'])

# Prepare features (7 features)
X = df[['State_Encoded', 'District_Encoded', 'Crop_Encoded', 'Season_Encoded', 
        'Area_Acres', 'Crop_Year', 'Annual_Rainfall']]
y = df['Yield_Q_Acre']

# Train model
print("\n🤖 Training Random Forest Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"✅ Training completed! R² Score: {r2:.4f}")

# Save models
print("\n💾 Saving model files...")
joblib.dump(model, 'model.pkl')
joblib.dump(le_state, 'state_encoder.pkl')
joblib.dump(le_district, 'district_encoder.pkl')
joblib.dump(le_crop, 'crop_encoder.pkl')
joblib.dump(le_season, 'season_encoder.pkl')
print("✅ All files saved successfully!")
 train_model.py