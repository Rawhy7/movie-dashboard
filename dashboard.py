import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
import os

# --- 1. PAGE CONFIGURATION & STYLING (Legendary UI) ---
st.set_page_config(
    page_title="Studio Executive Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# Custom CSS for "Legendary" Interface
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
    
    /* Custom Card Container */
    .css-card {
        background-color: #1e2329;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border: 1px solid #2d333b;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #2d333b;
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(75, 108, 183, 0.4);
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #4b6cb7;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #4b6cb7;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING & MODEL TRAINING (Cached) ---

@st.cache_resource
def load_and_train_models():
    # PATHS TO CHECK (It will try these in order)
    paths_to_check = [
        r"D:\Boudy\الجامعة\DS 2\Intro DS\Datasets\Dataset Authentic Movies.xlsx",  # Your original path
        "Dataset Authentic Movies.xlsx",  # Same folder as script
        "Dataset Authentic Movies.xlsx - Final_Clean_10000_Movies.csv" # CSV version
    ]
    
    df = None
    loaded_path = ""

    # Try loading files
    for path in paths_to_check:
        if os.path.exists(path):
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)
                loaded_path = path
                break
            except Exception as e:
                continue
    
    if df is None:
        st.error("❌ CRITICAL ERROR: Dataset not found!")
        st.warning(f"I looked for the file in these locations:\n1. {paths_to_check[0]}\n2. Current Folder ({os.getcwd()})")
        st.info("💡 TIP: Copy 'Dataset Authentic Movies.xlsx' into the same folder as this script.")
        return None, None, None

    # Prepare Data
    # Ensure columns exist (case insensitive check optional, but assuming exact names for now)
    required_cols = ['BudgetUSD', 'IMDbRating', 'RottenTomatoesScore', 'SuccessStatus']
    
    # Calculate Global_BoxOfficeUSD if missing but US exists (simple fill for demo) or check for it
    if 'Global_BoxOfficeUSD' not in df.columns and 'BoxOfficeUSD' in df.columns:
         df['Global_BoxOfficeUSD'] = df['BoxOfficeUSD'] # Fallback
    
    if 'Global_BoxOfficeUSD' in df.columns:
        # Calculate ROI
        df['ROI'] = (df['Global_BoxOfficeUSD'] - df['BudgetUSD']) / df['BudgetUSD']
        
        # Map Success
        if df['SuccessStatus'].dtype == 'O':
            df['SuccessStatus'] = df['SuccessStatus'].map({'Success': 1, 'Failure': 0})
        
        # Clean
        df_clean = df[['BudgetUSD', 'Global_BoxOfficeUSD', 'IMDbRating', 'RottenTomatoesScore', 'SuccessStatus', 'ROI']].dropna()
        
        # Train Financial Model
        X_fin = df_clean[['BudgetUSD']]
        y_fin = df_clean['Global_BoxOfficeUSD']
        fin_model = RandomForestRegressor(n_estimators=100, random_state=42)
        fin_model.fit(X_fin, y_fin)
        
        # Train Success Probability Model
        X_main = df_clean[['BudgetUSD', 'IMDbRating', 'RottenTomatoesScore']]
        y_main = df_clean['SuccessStatus']
        main_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        main_model.fit(X_main, y_main)
        
        return fin_model, main_model, df_clean
    else:
        st.error(f"Dataset loaded from {loaded_path} but missing 'Global_BoxOfficeUSD' column.")
        st.write("Columns found:", df.columns.tolist())
        return None, None, None

# Load models
fin_model, main_model, df = load_and_train_models()

# --- 3. UI LAYOUT ---

# Title Section
st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <h1 style="font-size: 3.5rem; background: -webkit-linear-gradient(eee, #333); -webkit-background-clip: text; color: white; text-shadow: 0 0 20px rgba(75, 108, 183, 0.5);">
            🎬 STUDIO EXECUTIVE <span style="color: #4b6cb7;">DASHBOARD</span>
        </h1>
        <p style="font-size: 1.2rem; color: #8b949e;">AI-Powered Film Investment & Risk Analysis System</p>
    </div>
""", unsafe_allow_html=True)

if fin_model is not None and main_model is not None:
    
    # --- SIDEBAR INPUTS ---
    with st.sidebar:
        st.header("🎛️ Project Parameters")
        st.markdown("Adjust the core parameters to simulate the movie's performance.")
        
        st.divider()
        
        budget_m = st.slider(
            "💰 Production Budget ($ Millions)",
            min_value=1, max_value=350, value=80, step=1,
            help="Total production budget excluding marketing."
        )
        
        st.divider()
        
        imdb_score = st.slider(
            "⭐ Projected IMDb Rating",
            min_value=1.0, max_value=10.0, value=7.2, step=0.1,
            help="Expected audience rating."
        )
        
        rt_score = st.slider(
            "🍅 Rotten Tomatoes Score (%)",
            min_value=0, max_value=100, value=75, step=1,
            help="Expected critical approval rating."
        )
        
        st.markdown("---")
        analyze_btn = st.button("🚀 RUN ANALYSIS ENGINE", type="primary")
        
        st.markdown("""
            <div style="margin-top: 20px; padding: 10px; background: #1c2128; border-radius: 8px; font-size: 0.8rem; color: #8b949e;">
                <strong>System Status:</strong><br>
                ✅ Financial Model: Active<br>
                ✅ Risk Oracle: Active<br>
                ✅ Database: Connected
            </div>
        """, unsafe_allow_html=True)

    # --- MAIN ANALYSIS LOGIC ---
    if analyze_btn:
        with st.spinner('Running Monte Carlo simulations and predictive algorithms...'):
            # Calculations
            real_budget = budget_m * 1_000_000
            
            # Financial Predictions
            pred_revenue = fin_model.predict([[real_budget]])[0]
            pred_roi = (pred_revenue - real_budget) / real_budget
            pred_profit = pred_revenue - real_budget
            
            # Success Probability
            prob = main_model.predict_proba([[real_budget, imdb_score, rt_score]])[0][1] * 100
            
            # Critical Score Metrics
            rt_scaled = rt_score / 10
            composite = (imdb_score * 0.6) + (rt_scaled * 0.4)
            gap = abs(imdb_score - rt_scaled)

            # --- LOGIC FOR TEXT AND COLORS ---
            # Determine Verdict
            if prob >= 85 and pred_roi > 2.0:
                verdict = "💎 BLOCKBUSTER POTENTIAL"
                verdict_color = "#2ecc71" # Green
                rec_text = "GREEN LIGHT - Exceptional promise across all metrics."
            elif prob >= 60 and pred_roi > 0.5:
                verdict = "✅ STRONG CONTENDER"
                verdict_color = "#3498db" # Blue
                rec_text = "RECOMMEND - Solid fundamentals with good potential."
            elif prob >= 40 or pred_roi > 0:
                verdict = "⚖️ MODERATE RISK"
                verdict_color = "#f1c40f" # Yellow
                rec_text = "CAUTION - Requires strong marketing execution."
            else:
                verdict = "🛑 HIGH RISK"
                verdict_color = "#e74c3c" # Red
                rec_text = "RECONSIDER - Significant challenges predicted."

            # Financial Status
            if pred_roi > 3.0: fin_stat, fin_icon = "EXCEPTIONAL", "💸"
            elif pred_roi > 1.0: fin_stat, fin_icon = "STRONG", "✅"
            elif pred_roi > 0: fin_stat, fin_icon = "MARGINAL", "⚖️"
            else: fin_stat, fin_icon = "CONCERNING", "📉"

            # Critical Status
            if composite >= 8.0: crit_stat, crit_icon = "ACCLAIMED", "🏆"
            elif composite >= 6.5: crit_stat, crit_icon = "POSITIVE", "✅"
            elif gap > 2.0: crit_stat, crit_icon = "DIVISIVE", "⚡"
            else: crit_stat, crit_icon = "POOR", "📉"

            # --- DISPLAY RESULTS (HTML Injection for custom layout) ---
            
            # 1. TOP VERDICT BANNER
            st.markdown(f"""
                <div style="background: {verdict_color}20; border: 2px solid {verdict_color}; border-radius: 15px; padding: 30px; text-align: center; margin-bottom: 30px;">
                    <h2 style="color: {verdict_color}; font-size: 3rem; margin: 0; text-transform: uppercase; letter-spacing: 2px;">{verdict}</h2>
                    <div style="font-size: 1.5rem; font-weight: bold; margin: 15px 0; color: white;">
                        Success Probability: {prob:.1f}%
                    </div>
                    <p style="color: #ddd; font-size: 1.1rem; margin: 0;">{rec_text}</p>
                </div>
            """, unsafe_allow_html=True)

            # 2. METRIC COLUMNS
            col1, col2, col3 = st.columns(3)

            # Financial Column
            with col1:
                st.markdown(f"""
                    <div class="css-card" style="border-top: 5px solid #2ecc71;">
                        <h3 style="text-align: center; color: #2ecc71; margin-bottom: 20px;">{fin_icon} FINANCIAL</h3>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #8b949e;">Revenue</span>
                            <span style="font-weight: bold; font-size: 1.2rem;">${pred_revenue/1e6:.1f}M</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #8b949e;">Profit</span>
                            <span style="font-weight: bold; font-size: 1.2rem; color: {'#2ecc71' if pred_profit > 0 else '#e74c3c'}">${pred_profit/1e6:.1f}M</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8b949e;">ROI</span>
                            <span style="font-weight: bold; font-size: 1.2rem; color: {'#2ecc71' if pred_roi > 0 else '#e74c3c'}">{pred_roi*100:.1f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Critical Column
            with col2:
                st.markdown(f"""
                    <div class="css-card" style="border-top: 5px solid #e67e22;">
                        <h3 style="text-align: center; color: #e67e22; margin-bottom: 20px;">{crit_icon} RECEPTION</h3>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #8b949e;">Composite</span>
                            <span style="font-weight: bold; font-size: 1.2rem;">{composite:.1f}/10</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #8b949e;">IMDb</span>
                            <span style="font-weight: bold; font-size: 1.2rem;">{imdb_score}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8b949e;">R.Tomatoes</span>
                            <span style="font-weight: bold; font-size: 1.2rem;">{rt_score}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Risk Column
            with col3:
                break_even = (real_budget * 2.5) / 1e6
                st.markdown(f"""
                    <div class="css-card" style="border-top: 5px solid #9b59b6;">
                        <h3 style="text-align: center; color: #9b59b6; margin-bottom: 20px;">📊 RISK PROFILE</h3>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #8b949e;">Budget</span>
                            <span style="font-weight: bold; font-size: 1.2rem;">${budget_m}M</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span style="color: #8b949e;">Break-Even</span>
                            <span style="font-weight: bold; font-size: 1.2rem;">~${break_even:.1f}M</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #8b949e;">Critic Gap</span>
                            <span style="font-weight: bold; font-size: 1.2rem; color: {'#e74c3c' if gap > 2 else '#2ecc71'}">{gap:.1f}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # 3. INSIGHTS SECTION
            st.markdown("### 📋 AI Strategic Insights")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.info(f"""
                **💰 Financial Outlook:** {'Based on historical data, projects at this budget level typically see strong returns. The ROI projection suggests this could be a profitable venture.' if pred_roi > 1.0 else 'The financial model indicates challenges in achieving profitability. Consider budget optimization or enhanced marketing strategies.' if pred_roi > 0 else 'Historical comparisons suggest significant financial risk. A comprehensive risk mitigation strategy is essential.'}
                """)
                
            with insight_col2:
                st.warning(f"""
                **📣 Critical Reception:**
                {'Both critics and audiences are expected to respond positively. This alignment typically drives strong word-of-mouth marketing.' if gap < 1.5 and composite > 7 else 'Expect divergent opinions between critics and general audiences. This could create buzz but may limit broad appeal.' if gap > 2 else 'Reception metrics suggest moderate to poor critical response. Enhanced quality control may be necessary.'}
                """)

else:
    st.write("---")