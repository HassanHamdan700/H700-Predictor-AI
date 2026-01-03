import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# --- Page Config ---
st.set_page_config(page_title="Premier League Predictor (No Odds)", layout="wide")

# --- Configuration: Define All 3 Models ---
MODELS = {
    "Random Forest": {
        "model": "football_model_rf_no_odds.pkl",
        "encoding": "team_encoding_rf_no_odds.pkl",
        "history": "full_match_history_rf_no_odds.pkl"
    },
    "Logistic Regression": {
        "model": "football_model_no_odds.pkl",
        "encoding": "team_encoding_no_odds.pkl",
        "history": "full_match_history_no_odds.pkl"
    },
    "XGBoost (Pure 26)": {
        "model": "football_model_pure26_no_odd.pkl",       # Your unique XGBoost Model (OK)
        "encoding": "team_encoding_pure26_no_odd.pkl",    # Your unique XGBoost Encoding (OK)
        # FIX IS HERE: We use the "no_odds" history file because the "pure26" file is broken.
        "history": "full_match_history_no_odds.pkl"       
    }
}

# --- Load Resources (Cached) ---
@st.cache_resource(show_spinner=False)
def load_all_models():
    loaded_data = {}
    for name, config in MODELS.items():
        try:
            # 1. Check if files exist
            if not os.path.exists(config["model"]):
                st.warning(f"⚠️ {name}: Model file not found.")
                continue
            
            # 2. Load Model
            try:
                model = joblib.load(config["model"])
            except Exception as e:
                st.error(f"❌ Error loading {name} model: {e}")
                continue

            # 3. Load Encoding
            try:
                with open(config["encoding"], 'rb') as f:
                    encoding = pickle.load(f)
            except Exception as e:
                st.error(f"❌ Error loading {name} encoding: {e}")
                continue
            
            # 4. Load History
            try:
                with open(config["history"], 'rb') as f:
                    history = pickle.load(f)
            except Exception as e:
                st.error(f"❌ Error loading {name} history: {e}")
                continue
            
            loaded_data[name] = {
                "model": model,
                "encoding": encoding,
                "history": history
            }
            
        except Exception as e:
            st.error(f"⚠️ Unexpected error for {name}: {e}")
            continue
            
    return loaded_data

# Load everything once
model_registry = load_all_models()

# --- Helper Functions ---
def get_rolling_stats(team_name, df, is_home_team):
    """Fetches rolling stats for a team."""
    team_matches = df[df["HomeTeam"] == team_name]
    if team_matches.empty:
        return None
    
    if "Date" in team_matches.columns:
        last_match = team_matches.sort_values("Date", ascending=False).iloc[0]
    else:
        last_match = team_matches.iloc[-1]

    cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]
    rolling_cols = [f"{c}_rolling" for c in cols]
    
    stats = last_match[rolling_cols].to_dict()
    
    if not is_home_team:
        stats = {f"Away_{k}": v for k, v in stats.items()}
        
    return stats

# --- Main Interface ---
st.title("⚽ Premier League Predictor")
st.markdown("### Compare predictions (No Betting Odds)")

if not model_registry:
    st.error("❌ No models loaded successfully. Please check your files.")
else:
    # --- 1. SINGLE INPUT SECTION ---
    # Use the first available model to get the team list
    first_model_key = list(model_registry.keys())[0]
    teams = sorted(list(model_registry[first_model_key]["encoding"].keys()))
    
    st.sidebar.header("Match Details")
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams, index=0)
    with col2:
        away_team = st.selectbox("Away Team", teams, index=1)

    col3, col4 = st.columns(2)
    with col3:
        match_date = st.date_input("Match Date")
    with col4:
        match_time = st.time_input("Match Time")

    # --- 2. PREDICTION LOGIC ---
    if st.button("Run Predictions", type="primary"):
        if home_team == away_team:
            st.error("Home and Away teams cannot be the same!")
        else:
            st.divider()
            
            # Create columns dynamically based on how many models loaded
            cols = st.columns(len(model_registry))
            
            for i, (model_name, assets) in enumerate(model_registry.items()):
                with cols[i]:
                    st.subheader(model_name)
                    
                    # 1. Prepare Features
                    day_code = match_date.weekday()
                    hour = match_time.hour
                    home_code = assets["encoding"].get(home_team, -1)
                    away_code = assets["encoding"].get(away_team, -1)
                    
                    # 2. Get History
                    home_stats = get_rolling_stats(home_team, assets["history"], True)
                    away_stats = get_rolling_stats(away_team, assets["history"], False)
                    
                    if not home_stats or not away_stats:
                        st.warning("Insufficient history data.")
                        continue

                    # 3. Build Input
                    input_data = {
                        "home_code": home_code,
                        "away_code": away_code,
                        "hour": hour,
                        "day_code": day_code
                    }
                    input_data.update(home_stats)
                    input_data.update(away_stats)
                    
                    # 4. Predict
                    try:
                        input_df = pd.DataFrame([input_data])
                        
                        # Handle specific column ordering for XGBoost if needed
                        # (Usually pandas DF aligns by name, but XGBoost can be strict)
                        if "XGBoost" in model_name:
                             # Reorder columns to match model's expected input if needed
                             # For now, we rely on feature names matching training
                             pass

                        prediction = assets["model"].predict(input_df)[0]
                        probs = assets["model"].predict_proba(input_df)[0]
                        
                        # Display
                        winner_map = {2: "Home Win", 1: "Draw", 0: "Away Win"}
                        color_map = {2: "green", 1: "gray", 0: "red"}
                        
                        result_text = winner_map[prediction]
                        result_color = color_map[prediction]
                        
                        st.markdown(f"**Result:** :{result_color}[{result_text}]")
                        
                        st.caption("Confidence:")
                        st.progress(probs[2], text=f"Home: {probs[2]:.1%}")
                        st.progress(probs[1], text=f"Draw: {probs[1]:.1%}")
                        st.progress(probs[0], text=f"Away: {probs[0]:.1%}")
                        
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")