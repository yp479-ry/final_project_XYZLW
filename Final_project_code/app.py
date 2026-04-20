import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

DELAY_COLS = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
              "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
DELAY_LABELS = ["Carrier", "Weather", "NAS", "Security", "Late Aircraft"]

# ── Data loading (cached) ─────────────────────────────────────────────────────
# ============================================================
# TODO: Change this to your local folder containing the CSV files
DATA_DIR = "/Users/linwu/Downloads"
# ============================================================

@st.cache_data
def load_data(directory=DATA_DIR):
    files = (glob.glob(f"{directory}/*ONTIME_REPORTING.csv") or
             glob.glob(f"{directory}/*.csv"))
    if not files:
        st.error(f"No CSV files found in '{directory}'.")
        st.stop()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Only drop rows missing core flight columns — keep on-time flights
    df = df.dropna(subset=["DEP_DELAY", "DEP_DELAY_NEW", "DEP_DEL15"]).copy()
    # Non-delayed flights have NaN in delay cause columns → fill with 0
    for c in DELAY_COLS:
        df[c] = df[c].fillna(0).clip(lower=0)
    return df

# ── Model training (cached) ───────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    X = df[DELAY_COLS].fillna(0)
    y_cls = df["DEP_DEL15"].astype(int)
    y_reg = df["DEP_DELAY_NEW"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10,
                                  random_state=42, n_jobs=-1)
    rfc.fit(X_tr, y_tr)
    lr = LinearRegression().fit(X, y_reg)
    return rfc, lr

# ── Airline metrics (cached) ──────────────────────────────────────────────────
@st.cache_data
def compute_airline_metrics(df):
    perf = df.groupby("OP_UNIQUE_CARRIER").agg(
        Delay_Ratio        = ("DEP_DEL15",     "mean"),
        Avg_Delay          = ("DEP_DELAY_NEW", "mean"),
        CARRIER_DELAY      = ("CARRIER_DELAY",      "mean"),
        WEATHER_DELAY      = ("WEATHER_DELAY",      "mean"),
        NAS_DELAY          = ("NAS_DELAY",          "mean"),
        SECURITY_DELAY     = ("SECURITY_DELAY",     "mean"),
        LATE_AIRCRAFT_DELAY= ("LATE_AIRCRAFT_DELAY","mean"),
    ).reset_index()
    perf["Delay_Ratio (%)"] = perf["Delay_Ratio"] * 100
    perf["Reliability"] = perf["Delay_Ratio (%)"].apply(
        lambda x: "⭐⭐⭐ Excellent" if x < 20
                  else ("⭐⭐ Good" if x < 28 else "⭐ Poor")
    )
    return perf

# ── Load everything ───────────────────────────────────────────────────────────
df = load_data()
rfc, lr_model = train_models(df)
airline_metrics = compute_airline_metrics(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Airplane_silhouette.svg/120px-Airplane_silhouette.svg.png", width=80)
st.sidebar.title("✈️ Flight Delay Predictor")
st.sidebar.markdown("Powered by **Random Forest** + **Linear Regression**")
st.sidebar.markdown("---")

origin_cities  = sorted(df["ORIGIN_CITY_NAME"].dropna().unique())
dest_cities    = sorted(df["DEST_CITY_NAME"].dropna().unique())
airlines       = sorted(df["OP_UNIQUE_CARRIER"].dropna().unique())
months         = list(range(1, 13))
month_names    = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

origin  = st.sidebar.selectbox("Departure City", origin_cities)
dest    = st.sidebar.selectbox("Arrival City",   dest_cities)
month   = st.sidebar.selectbox("Travel Month",   months,
                                format_func=lambda m: month_names[m-1])
predict_btn = st.sidebar.button("🔍 Predict & Compare", use_container_width=True)

# ── Main page ─────────────────────────────────────────────────────────────────
st.title("✈️ U.S. Flight Delay Prediction & Airline Recommendation")
st.markdown(
    "Enter your trip details on the left, then click **Predict & Compare** "
    "to get real-time delay predictions and airline recommendations."
)

if predict_btn:
    # Filter historical data for the chosen route + month
    route_df = df[
        (df["ORIGIN_CITY_NAME"] == origin) &
        (df["DEST_CITY_NAME"]   == dest)   &
        (df["MONTH"]            == month)
    ]

    if route_df.empty:
        st.warning(
            "No historical data found for this route/month combination. "
            "Showing predictions based on overall airline averages."
        )
        route_df = df[df["MONTH"] == month]

    results = []
    for carrier in airlines:
        carrier_df = route_df[route_df["OP_UNIQUE_CARRIER"] == carrier]
        if carrier_df.empty:
            carrier_df = df[df["OP_UNIQUE_CARRIER"] == carrier]
        if carrier_df.empty:
            continue
        X_pred = carrier_df[DELAY_COLS].fillna(0)
        delay_prob    = rfc.predict_proba(X_pred)[:, 1].mean()
        expected_delay = lr_model.predict(X_pred).mean()
        avg_causes    = X_pred.mean().tolist()
        results.append({
            "Airline": carrier,
            "Delay Probability (%)": round(delay_prob * 100, 1),
            "Expected Delay (min)": round(max(expected_delay, 0), 1),
            **{label: round(val, 1) for label, val in zip(DELAY_LABELS, avg_causes)}
        })

    if not results:
        st.error("Could not generate predictions. Please try a different route.")
        st.stop()

    results_df = pd.DataFrame(results).sort_values("Delay Probability (%)", ascending=True)

    # ── Best airline recommendation ───────────────────────────────────────────
    best = results_df.iloc[0]
    st.markdown("---")
    st.subheader("🏆 Recommended Airline")
    col1, col2, col3 = st.columns(3)
    col1.metric("Airline",                best["Airline"])
    col2.metric("Delay Probability",      f"{best['Delay Probability (%)']:.1f}%")
    col3.metric("Expected Delay",         f"{best['Expected Delay (min)']:.1f} min")

    # ── Airline comparison table ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Airline Comparison")
    display_df = results_df[["Airline", "Delay Probability (%)", "Expected Delay (min)"]].copy()
    display_df = display_df.merge(
        airline_metrics[["OP_UNIQUE_CARRIER", "Reliability"]],
        left_on="Airline", right_on="OP_UNIQUE_CARRIER", how="left"
    ).drop(columns="OP_UNIQUE_CARRIER")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    # ── Bar chart comparison ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Delay Probability & Expected Duration by Airline")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71" if a == best["Airline"] else "#3498db"
              for a in results_df["Airline"]]
    ax1.barh(results_df["Airline"], results_df["Delay Probability (%)"], color=colors)
    ax1.set_xlabel("Delay Probability (%)")
    ax1.set_title("Delay Probability by Airline")
    ax1.axvline(x=results_df["Delay Probability (%)"].mean(), color="red",
                linestyle="--", label="Average")
    ax1.legend()

    ax2.barh(results_df["Airline"], results_df["Expected Delay (min)"], color=colors)
    ax2.set_xlabel("Expected Delay (min)")
    ax2.set_title("Expected Delay Duration by Airline")

    green_patch = mpatches.Patch(color="#2ecc71", label="Recommended")
    blue_patch  = mpatches.Patch(color="#3498db", label="Other")
    fig.legend(handles=[green_patch, blue_patch], loc="lower center", ncol=2)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Delay cause donut chart ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"🍩 Delay Cause Breakdown — {best['Airline']}")
    cause_vals = [best.get(label, 0) for label in DELAY_LABELS]
    cause_vals = [max(v, 0.01) for v in cause_vals]  # avoid zero-wedge rendering

    fig2, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        cause_vals, labels=DELAY_LABELS,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.8,
        wedgeprops=dict(width=0.5)
    )
    ax.set_title(f"Avg Delay Cause — {best['Airline']} ({month_names[month-1]})")
    st.pyplot(fig2)

else:
    st.info("👈 Fill in your trip details in the sidebar and click **Predict & Compare**.")
    st.markdown("### 📌 Overall Airline Delay Statistics")
    st.dataframe(
        airline_metrics[["OP_UNIQUE_CARRIER", "Delay_Ratio (%)",
                         "Avg_Delay", "Reliability"]]
        .rename(columns={"OP_UNIQUE_CARRIER": "Airline",
                         "Avg_Delay": "Avg Delay (min)"})
        .sort_values("Delay_Ratio (%)"),
        use_container_width=True
    )