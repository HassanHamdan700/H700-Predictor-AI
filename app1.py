import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# --- Page Config ---
st.set_page_config(page_title="Premier League Super Predictor", layout="centered")

# --- Sidebar: Model Selection ---
st.sidebar.title("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose your Algorithm:",
    ("Random Forest (No Odds)", "Logistic Regression (No Odds)", "XGBoost (NO Odds)")
)

# --- Configuration: Define Paths per Model ---
# Update these filenames to match exactly what you have in your folder
models_config = {
    "Random Forest (No Odds)": {
        "model": "football_model_rf_no_odds.pkl",
        "encoding": "team_encoding_rf_no_odds.pkl",
        "history": "full_match_history_rf_no_odds.pkl",
        "use_odds": False
    },
    "Logistic Regression (No Odds)": {
        "model": "football_model_no_odds.pkl",
        "encoding": "team_encoding_no_odds.pkl",
        "history": "full_match_history_no_odds.pkl",
        "use_odds": False
    },
    "XGBoost (NO Odds)": {
        "model": "football_model_no_odds.pkl",         # Check your specific filename from step 1
        "encoding": "team_encoding_no_odds.pkl",      # Check your specific filename
        "history": "full_match_history_no_odds.pkl",    # Check your specific filename
        "use_odds": False
    }
}

current_config = models_config[model_choice]

# --- Load Resources (Cached) ---
@st.cache_resource(show_spinner=False)
def load_model_assets(config):
    try:
        # Load Model
        model = joblib.load(config["model"])
        
        # Load Encoding
        with open(config["encoding"], 'rb') as f:
            encoding = pickle.load(f)
            
        # Load History
        with open(config["history"], 'rb') as f:
            history = pickle.load(f)
            
        return model, encoding, history
    except FileNotFoundError as e:
        return None, None, None

model, team_encoding, df_history = load_model_assets(current_config)

# --- Helper Functions ---
def get_rolling_stats(team_name, df, is_home_team):
    """Fetches rolling stats for a team."""
    team_matches = df[df["HomeTeam"] == team_name]
    if team_matches.empty:
        return None
    
    # Get last match
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

def get_prob(val):
    return 1 / val if val > 0 else 0

# --- Main Interface ---
st.title(f"⚽ Match Predictor")
st.caption(f"Current Model: **{model_choice}**")

if model is None:
    st.error(f"⚠️ Could not load files for {model_choice}.")
    st.info(f"Looking for: `{current_config['model']}`. Please make sure all .pkl files are in the same folder.")
else:
    # -- inputs --
    col1, col2 = st.columns(2)
    teams = sorted(list(team_encoding.keys()))
    
    with col1:
        home_team = st.selectbox("Home Team", teams, index=0)
    with col2:
        away_team = st.selectbox("Away Team", teams, index=1)

    col3, col4 = st.columns(2)
    with col3:
        match_date = st.date_input("Match Date")
    with col4:
        match_time = st.time_input("Match Time")

    # -- Conditional Odds Input --
    # Only show these fields if the selected model needs them (XGBoost)
    prob_h, prob_d, prob_a = 0, 0, 0
    
    if current_config["use_odds"]:
        st.subheader("Betting Odds")
        c_o1, c_o2, c_o3 = st.columns(3)
        with c_o1:
            odd_h = st.number_input("Home Odds", value=2.50)
        with c_o2:
            odd_d = st.number_input("Draw Odds", value=3.20)
        with c_o3:
            odd_a = st.number_input("Away Odds", value=2.80)
            
        prob_h = get_prob(odd_h)
        prob_d = get_prob(odd_d)
        prob_a = get_prob(odd_a)

    # -- Prediction --
    if st.button("Predict Outcome", type="primary"):
        if home_team == away_team:
            st.warning("Teams cannot be the same.")
        else:
            # 1. Basic Features
            day_code = match_date.weekday()
            hour = match_time.hour
            home_code = team_encoding.get(home_team, -1)
            away_code = team_encoding.get(away_team, -1)
            
            # 2. History
            home_stats = get_rolling_stats(home_team, df_history, True)
            away_stats = get_rolling_stats(away_team, df_history, False)
            
            if not home_stats or not away_stats:
                st.error("Not enough history for these teams.")
            else:
                # 3. Build Input Dictionary
                input_data = {
                    "home_code": home_code,
                    "away_code": away_code,
                    "hour": hour,
                    "day_code": day_code
                }
                
                # Add Odds ONLY if model needs them
                if current_config["use_odds"]:
                    input_data["prob_H"] = prob_h
                    input_data["prob_D"] = prob_d
                    input_data["prob_A"] = prob_a
                
                input_data.update(home_stats)
                input_data.update(away_stats)
                
                # 4. Create DF and Predict
                input_df = pd.DataFrame([input_data])
                
                # Handle XGBoost column order matching if necessary
                # (XGBoost is sensitive to column order; usually Pandas handles names, 
                # but if there's an issue, reindexing might be needed. 
                # For now, we trust the dict-to-df conversion matches training)
                
                try:
                    prediction = model.predict(input_df)[0]
                    probs = model.predict_proba(input_df)[0]
                    
                    # 5. Display
                    st.divider()
                    winner_map = {2: "Home Win", 1: "Draw", 0: "Away Win"}
                    color_map = {2: "green", 1: "gray", 0: "red"}
                    
                    st.markdown(f"### Prediction: :{color_map[prediction]}[{winner_map[prediction]}]")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"Home ({home_team})", f"{probs[2]:.1%}")
                    c2.metric("Draw", f"{probs[1]:.1%}")
                    c3.metric(f"Away ({away_team})", f"{probs[0]:.1%}")
                    
                    st.progress(probs[2], text="Home Confidence")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.write("Debug - Input Cols:", input_df.columns.tolist())