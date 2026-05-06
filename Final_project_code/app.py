import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
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
@st.cache_data
def load_data(directory="Dataset"):
    files = glob.glob(f"{directory}/*.csv")
    if not files:
        st.error(f"No CSV files found in '{directory}'.")
        st.stop()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    clean_cols = ["DEP_DELAY", "DEP_DELAY_NEW", "DEP_DEL15"] + DELAY_COLS
    df = df.dropna(subset=clean_cols).copy()
    for c in DELAY_COLS:
        df[c] = df[c].clip(lower=0)
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
    q1 = perf["Delay_Ratio (%)"].quantile(0.33)
    q2 = perf["Delay_Ratio (%)"].quantile(0.66)

    perf["Reliability"] = perf["Delay_Ratio (%)"].apply(
        lambda x: "⭐⭐⭐ Lower Risk" if x <= q1
                else ("⭐⭐ Medium Risk" if x <= q2 else "⭐ Higher Risk")
    )
    return perf

# ── Load everything ───────────────────────────────────────────────────────────
df = load_data()
rfc, lr_model = train_models(df)
airline_metrics = compute_airline_metrics(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("✈️ Flight Delay Predictor")
st.sidebar.markdown("Powered by **Historical Data + ML Risk Models**")
st.sidebar.markdown("---")

origin_cities  = sorted(df["ORIGIN_CITY_NAME"].dropna().unique())
dest_cities    = sorted(df["DEST_CITY_NAME"].dropna().unique())
airlines       = sorted(df["OP_UNIQUE_CARRIER"].dropna().unique())
months         = list(range(1, 13))
month_names    = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

origin  = st.sidebar.selectbox("Departure City", origin_cities)
dest    = st.sidebar.selectbox("Arrival City",   dest_cities)
month   = st.sidebar.selectbox(
    "Travel Month",
    months,
    format_func=lambda m: month_names[m-1]
)

risk_threshold = st.sidebar.slider(
    "Delay Risk Threshold (%)",
    min_value=10,
    max_value=90,
    value=75,
    step=5,
    help="Flights with predicted delay probability above this threshold will be marked as High Risk."
)
st.sidebar.caption(

    "Lower threshold = more conservative. Higher threshold = more flexible."

)

predict_btn = st.sidebar.button("🔍 Predict & Compare", width="stretch")

# ── Main page ─────────────────────────────────────────────────────────────────
st.title("✈️ U.S. Flight Delay Prediction & Airline Recommendation")
st.markdown(
    "Enter your trip details on the left, then click **Predict & Compare** "
    "to compare airline delay risk based on historical route and month patterns."
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
    results_df["Risk Level"] = np.where(
        results_df["Delay Probability (%)"] >= risk_threshold,
        "High Risk",
        "Acceptable Risk"
    )
    # ── Best airline recommendation ───────────────────────────────────────────
    acceptable_df = results_df[results_df["Risk Level"] == "Acceptable Risk"]

    if not acceptable_df.empty:
        best = acceptable_df.iloc[0]
    else:
        best = results_df.iloc[0]
        st.warning(
            "All available airlines are above your selected risk threshold. "
            "This is not a low-risk recommendation; it is the lowest-risk option among the available airlines."
        )

    st.markdown("---")
    st.subheader("🏆 Lowest-Risk Airline Option")

    # Smaller unified metric cards
    risk_color = "#2ecc71" if best["Risk Level"] == "Acceptable Risk" else "#ff4b4b"

    metric_html = f"""
    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:18px; margin-top:10px;">

    <div style="padding:18px 20px; border-radius:16px; background-color:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10);">
        <div style="font-size:15px; color:#A0A4AA; font-weight:600;">Airline</div>
        <div style="font-size:28px; color:white; font-weight:700; margin-top:8px;">{best["Airline"]}</div>
    </div>

    <div style="padding:18px 20px; border-radius:16px; background-color:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10);">
        <div style="font-size:15px; color:#A0A4AA; font-weight:600;">Delay Probability</div>
        <div style="font-size:28px; color:white; font-weight:700; margin-top:8px;">{best["Delay Probability (%)"]:.1f}%</div>
    </div>

    <div style="padding:18px 20px; border-radius:16px; background-color:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10);">
        <div style="font-size:15px; color:#A0A4AA; font-weight:600;">Expected Delay</div>
        <div style="font-size:28px; color:white; font-weight:700; margin-top:8px;">{best["Expected Delay (min)"]:.1f} min</div>
    </div>

    <div style="padding:18px 20px; border-radius:16px; background-color:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10);">
        <div style="font-size:15px; color:#A0A4AA; font-weight:600;">Risk Level</div>
        <div style="display:inline-block; font-size:18px; color:{risk_color}; font-weight:700; margin-top:10px; padding:5px 10px; border-radius:999px; background-color:{risk_color}22; border:1px solid {risk_color}; white-space:nowrap;">
        {best["Risk Level"]}
        </div>
    </div>

    </div>
    """

    st.markdown(metric_html, unsafe_allow_html=True)
    # ── Recommendation explanation ───────────────────────────────────────────────
    cause_summary = pd.DataFrame({
        "Cause": DELAY_LABELS,
        "Average Delay Minutes": [best.get(label, 0) for label in DELAY_LABELS]
    }).sort_values("Average Delay Minutes", ascending=False)

    top_cause = cause_summary.iloc[0]["Cause"]
    top_cause_value = cause_summary.iloc[0]["Average Delay Minutes"]

    st.markdown(
        f"""
        <div style="margin-top:18px; padding:18px 22px; border-radius:16px; background-color:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10);">
            <div style="font-size:18px; font-weight:700; color:white; margin-bottom:8px;">
                💡 Why this option?
            </div>
            <div style="font-size:15px; color:#C9CDD3; line-height:1.6;">
                <b>{best["Airline"]}</b> has the lowest predicted delay probability among the available airlines
                for the selected route and month. Its predicted delay probability is
                <b>{best["Delay Probability (%)"]:.1f}%</b>, with an expected delay of
                <b>{best["Expected Delay (min)"]:.1f} minutes</b>. The largest estimated delay contributor is
                <b>{top_cause}</b> at about <b>{top_cause_value:.1f} minutes</b>.
            </div>
            <div style="margin-top:14px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.10); font-size:13px; color:#9AA3B2;">
                ⚙️ Model pipeline: historical route/month filtering → delay probability estimation → expected delay duration → airline risk comparison.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    

    # ── Airline comparison table ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Airline Comparison")
    display_df = results_df[
        ["Airline", "Delay Probability (%)", "Expected Delay (min)", "Risk Level"]
    ].copy()
    display_df = display_df.merge(
        airline_metrics[["OP_UNIQUE_CARRIER", "Reliability"]],
        left_on="Airline", right_on="OP_UNIQUE_CARRIER", how="left"
    ).drop(columns="OP_UNIQUE_CARRIER")
    st.dataframe(display_df.reset_index(drop=True), width="stretch")

    # ── Airline comparison charts ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Delay Risk and Expected Delay by Airline")

    chart_df = results_df.copy()
    chart_df["Recommendation"] = np.where(
        chart_df["Airline"] == best["Airline"],
        "Recommended",
        "Other"
    )

    chart_df["DelayProbability"] = chart_df["Delay Probability (%)"]
    chart_df["ExpectedDelay"] = chart_df["Expected Delay (min)"]

    threshold_df = pd.DataFrame({
        "Risk Threshold": [risk_threshold]
    })

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Delay Probability")

        prob_bars = alt.Chart(chart_df).mark_bar(
            cornerRadiusEnd=6
        ).encode(
            y=alt.Y(
                "Airline:N",
                sort=alt.EncodingSortField(
                    field="DelayProbability",
                    order="ascending"
                ),
                title=None
            ),
            x=alt.X(
                "DelayProbability:Q",
                title="Delay Probability (%)",
                scale=alt.Scale(domain=[0, 100])
            ),
            color=alt.Color(
                "Recommendation:N",
                scale=alt.Scale(
                    domain=["Recommended", "Other"],
                    range=["#2ecc71", "#4aa3df"]
                ),
                legend=alt.Legend(title=None)
            ),
            tooltip=[
                alt.Tooltip("Airline:N", title="Airline"),
                alt.Tooltip("DelayProbability:Q", title="Delay Probability (%)", format=".1f"),
                alt.Tooltip("ExpectedDelay:Q", title="Expected Delay (min)", format=".1f"),
                alt.Tooltip("Risk Level:N", title="Risk Level")
            ]
        ).properties(
            height=360
        )

        threshold_line = alt.Chart(threshold_df).mark_rule(
            color="#ff4b4b",
            strokeDash=[6, 4],
            size=2
        ).encode(
            x="Risk Threshold:Q"
        )

        threshold_label = alt.Chart(threshold_df).mark_text(
            text=f"Threshold: {risk_threshold}%",
            align="left",
            dx=6,
            dy=-8,
            color="#ff4b4b",
            fontSize=12
        ).encode(
            x="Risk Threshold:Q",
            y=alt.value(10)
        )

        prob_chart = (prob_bars + threshold_line + threshold_label).configure_view(
            strokeWidth=0
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=13
        )

        st.altair_chart(prob_chart, width="stretch")

    with right_col:
        st.markdown("#### Expected Delay Duration")

        delay_chart = alt.Chart(chart_df).mark_bar(
            cornerRadiusEnd=6
        ).encode(
            y=alt.Y(
                "Airline:N",
                sort=alt.EncodingSortField(
                    field="ExpectedDelay",
                    order="ascending"
                ),
                title=None
            ),
            x=alt.X(
                "ExpectedDelay:Q",
                title="Expected Delay (minutes)"
            ),
            color=alt.Color(
                "Recommendation:N",
                scale=alt.Scale(
                    domain=["Recommended", "Other"],
                    range=["#2ecc71", "#4aa3df"]
                ),
                legend=alt.Legend(title=None)
            ),
            tooltip=[
                alt.Tooltip("Airline:N", title="Airline"),
                alt.Tooltip("ExpectedDelay:Q", title="Expected Delay (min)", format=".1f"),
                alt.Tooltip("DelayProbability:Q", title="Delay Probability (%)", format=".1f"),
                alt.Tooltip("Risk Level:N", title="Risk Level")
            ]
        ).properties(
            height=360
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=13
        )

        st.altair_chart(delay_chart, width="stretch")

    # ── Delay cause breakdown chart ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"🍩 Delay Cause Breakdown — {best['Airline']}")

    cause_vals = [best.get(label, 0) for label in DELAY_LABELS]

    cause_df = pd.DataFrame({
        "Cause": DELAY_LABELS,
        "Average Delay Minutes": cause_vals
    })

    cause_df["Average Delay Minutes"] = cause_df["Average Delay Minutes"].clip(lower=0.01)
    cause_df["Percent"] = (
        cause_df["Average Delay Minutes"] /
        cause_df["Average Delay Minutes"].sum() * 100
    )

    donut_col, bar_col = st.columns([1, 1.25])

    with donut_col:
        st.markdown("#### Cause Share")

        donut_chart = alt.Chart(cause_df).mark_arc(
            innerRadius=70,
            outerRadius=125
        ).encode(
            theta=alt.Theta("Average Delay Minutes:Q"),
            color=alt.Color(
                "Cause:N",
                legend=alt.Legend(title="Delay Cause", orient="bottom")
            ),
            tooltip=[
                alt.Tooltip("Cause:N", title="Cause"),
                alt.Tooltip("Average Delay Minutes:Q", title="Avg Delay Minutes", format=".1f"),
                alt.Tooltip("Percent:Q", title="Share (%)", format=".1f")
            ]
        ).properties(
            height=330
        ).configure_view(
            strokeWidth=0
        )

        st.altair_chart(donut_chart, width="stretch")

    with bar_col:
        st.markdown("#### Average Minutes by Cause")

        cause_bar = alt.Chart(cause_df).mark_bar(
            cornerRadiusEnd=6
        ).encode(
            y=alt.Y(
                "Cause:N",
                sort=alt.EncodingSortField(
                    field="Average Delay Minutes",
                    order="descending"
                ),
                title=None
            ),
            x=alt.X(
                "Average Delay Minutes:Q",
                title="Average Delay Minutes"
            ),
            color=alt.Color(
                "Cause:N",
                legend=None
            ),
            tooltip=[
                alt.Tooltip("Cause:N", title="Cause"),
                alt.Tooltip("Average Delay Minutes:Q", title="Avg Delay Minutes", format=".1f"),
                alt.Tooltip("Percent:Q", title="Share (%)", format=".1f")
            ]
        ).properties(
            height=330
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=13
        )

        st.altair_chart(cause_bar, width="stretch")

else:
    st.info("👈 Fill in your trip details in the sidebar and click **Predict & Compare**.")
    st.markdown("### 📌 Overall Airline Delay Statistics")
    st.dataframe(
        airline_metrics[["OP_UNIQUE_CARRIER", "Delay_Ratio (%)",
                        "Avg_Delay", "Reliability"]]
        .rename(columns={"OP_UNIQUE_CARRIER": "Airline",
                        "Avg_Delay": "Avg Delay (min)"})
        .sort_values("Delay_Ratio (%)"),
        width="stretch"
    )