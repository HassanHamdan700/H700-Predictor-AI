import joblib
import pickle
import os

files_to_check = [
    # Random Forest
    "football_model_rf_no_odds.pkl",
    "team_encoding_rf_no_odds.pkl",
    "full_match_history_rf_no_odds.pkl",

    # Logistic Regression
    "football_model_no_odds.pkl",
    "team_encoding_no_odds.pkl",
    "full_match_history_no_odds.pkl",

    # XGBoost Pure 26
    "football_model_pure26_no_odd.pkl",
    "team_encoding_pure26_no_odd.pkl",
    "full_match_history_pure26_no_odd.pkl"
]

print("--- FILE INTEGRITY CHECK ---")

for filename in files_to_check:
    if not os.path.exists(filename):
        print(f"❌ MISSING: {filename}")
        continue
        
    try:
        # Try loading as joblib (models)
        if "model" in filename:
            joblib.load(filename)
        # Try loading as pickle (dictionaries/dataframes)
        else:
            with open(filename, 'rb') as f:
                pickle.load(f)
        print(f"✅ OK: {filename}")
    except Exception as e:
        print(f"🔥 CORRUPTED: {filename}")
        print(f"   Error details: {e}") 
        print("   -> ACTION: Delete this file and re-download it.")

print("--- END CHECK ---")