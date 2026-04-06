"""
predict.py - Standalone Prediction Module
Fixed for headers: 'District ' and 'Area '
"""
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class CropYieldPredictor:
    def __init__(self):
        try:
            self.model = joblib.load('model.pkl')
            self.le_state = joblib.load('state_encoder.pkl')
            self.le_district = joblib.load('district_encoder.pkl')
            self.le_crop = joblib.load('crop_encoder.pkl')
            self.le_season = joblib.load('season_encoder.pkl')
            
            # Load and clean dataset for lookups
            self.df = pd.read_csv('agridata.csv')
            self.df.columns = self.df.columns.str.strip() # Fixes 'District ' -> 'District'
            
            # Pre-fill rainfall
            if 'Annual_Rainfall' in self.df.columns:
                self.df['Annual_Rainfall'] = self.df['Annual_Rainfall'].fillna(
                    self.df.groupby(['State', 'District'])['Annual_Rainfall'].transform('mean')
                )
                self.df['Annual_Rainfall'] = self.df['Annual_Rainfall'].fillna(1000.0)
                
        except Exception as e:
            print(f"Error loading resources: {e}")
            self.model = None

    def get_rainfall(self, state, district):
        """Fetch rainfall from dataset"""
        try:
            val = self.df[(self.df['State'] == state) & (self.df['District'] == district)]['Annual_Rainfall'].mean()
            return val if not pd.isna(val) else 1000.0
        except:
            return 1000.0

    def predict(self, state, district, crop, season, area, year, rainfall=None):
        if not self.model: return {"error": "Model not loaded"}
        
        # Auto-fetch rainfall if not provided
        if rainfall is None:
            rainfall = self.get_rainfall(state, district)
            
        try:
            # Encode
            inputs = np.array([[
                self.le_state.transform([state])[0],
                self.le_district.transform([district])[0],
                self.le_crop.transform([crop])[0],
                self.le_season.transform([season])[0],
                area,
                year,
                rainfall
            ]])
            
            yield_pred = self.model.predict(inputs)[0]
            
            return {
                "yield_q_acre": round(yield_pred, 2),
                "total_production_quintals": round(yield_pred * area, 2),
                "rainfall_used": round(rainfall, 1)
            }
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    p = CropYieldPredictor()
    # Test
    print(p.predict("Haryana", "Hisar", "Wheat", "Rabi", 100, 2024))