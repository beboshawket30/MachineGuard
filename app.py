import math
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ==========================================
# 1. Page Config & Custom Styling (Dark Theme)
# ==========================================
st.set_page_config(
    page_title="Predictive Maintenance", page_icon="⚙️", layout="wide"
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #0d0d0f;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #141417 !important;
        border-right: 1px solid #222228;
    }

    .main-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        border-left: 5px solid #ff1744;
        padding-left: 18px;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #9e9e9e;
        font-size: 1.05rem;
        padding-left: 23px;
        margin-bottom: 30px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* HIGH SPECIFICITY INPUTS & SELECTBOX HOVER EFFECT */
    div[data-testid="stNumberInput"] > div,
    div[data-testid="stSelectbox"] > div,
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div {
        background-color: #141417 !important;
        border: 1px solid #2a2a30 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease-in-out !important;
    }

    /* HOVER STATE */
    div[data-testid="stNumberInput"]:hover > div,
    div[data-testid="stSelectbox"]:hover > div,
    div[data-baseweb="input"]:hover,
    div[data-baseweb="select"] > div:hover {
        border-color: #ff1744 !important;
        box-shadow: 0 0 15px rgba(255, 23, 68, 0.45) !important;
        transform: translateY(-3px) !important;
    }

    /* FOCUS / ACTIVE STATE */
    div[data-testid="stNumberInput"]:focus-within > div,
    div[data-testid="stSelectbox"]:focus-within > div {
        border-color: #ff1744 !important;
        box-shadow: 0 0 20px rgba(255, 23, 68, 0.6) !important;
    }

    /* POPUP MODAL STYLING */
    div[data-testid="stDialog"] {
        background-color: rgba(0, 0, 0, 0.85) !important;
        backdrop-filter: blur(4px);
    }

    div[data-testid="stDialog"] > div[role="dialog"] {
        background-color: #141417 !important;
        border: 1px solid #2a2a30 !important;
        max-width: 850px !important;
        width: 85% !important;
        border-radius: 12px !important;
        padding: 30px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8) !important;
    }

    div[data-testid="stDialog"] h2 {
        color: #ffffff !important;
        font-family: 'Playfair Display', serif !important;
    }

    .protocol-card {
        background: #18181c;
        border: 1px solid #ff1744;
        border-radius: 8px;
        padding: 16px;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(255, 23, 68, 0.15);
    }
    .protocol-title {
        color: #ff6b6b;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 8px;
        text-transform: uppercase;
    }
    .protocol-desc {
        color: #cccccc;
        font-size: 0.85rem;
        line-height: 1.4;
        margin-bottom: 8px;
    }
    .protocol-target {
        color: #999999;
        font-size: 0.8rem;
        border-top: 1px solid #2a2a30;
        padding-top: 8px;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #b71c1c 0%, #ff1744 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 12px 28px !important;
        width: 100%;
        box-shadow: 0 4px 15px rgba(255, 23, 68, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        box-shadow: 0 6px 20px rgba(255, 23, 68, 0.6) !important;
        transform: translateY(-2px);
    }

    .result-card-danger {
        background: linear-gradient(135deg, #2a0808 0%, #4a0d0d 100%);
        border: 1px solid #ff1744;
        border-radius: 8px;
        padding: 25px;
        text-align: center;
        color: #ff6b6b;
        box-shadow: 0 4px 20px rgba(255, 23, 68, 0.25);
        margin-top: 15px;
    }
    .result-card-safe {
        background: linear-gradient(135deg, #0d1e15 0%, #133322 100%);
        border: 1px solid #00e676;
        border-radius: 8px;
        padding: 25px;
        text-align: center;
        color: #00e676;
        box-shadow: 0 4px 20px rgba(0, 230, 118, 0.15);
        margin-top: 15px;
    }

    .custom-alert-danger {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ff4d4d;
        text-align: center;
        margin-bottom: 5px;
    }
    .custom-alert-safe {
        font-size: 1.6rem;
        font-weight: 800;
        color: #00e676;
        text-align: center;
        margin-bottom: 5px;
    }

    /* TEAM SECTION STYLING */
    .team-container {
        margin-top: 60px;
        padding: 40px 20px;
        background-color: #141417;
        border: 1px solid #222228;
        border-radius: 12px;
        text-align: center;
    }
    .team-header {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
    }
    .team-sub {
        color: #a0a0ab;
        font-size: 0.95rem;
        margin-bottom: 35px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 25px;
        max-width: 1000px;
        margin: 0 auto;
    }
    .team-card {
        background: #18181c;
        border: 1px solid #2a2a30;
        border-radius: 12px;
        padding: 25px 15px;
        display: flex;
        flex-direction: column;
        align-items: center;
        transition: all 0.3s ease;
    }
    .team-card:hover {
        border-color: #ff1744;
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(255, 23, 68, 0.25);
    }
    .team-img-wrapper {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        overflow: hidden;
        border: 3px solid #ff1744;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
    }
    .team-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    .team-card:hover .team-img {
        transform: scale(1.15);
    }
    .team-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .team-role {
        font-size: 0.82rem;
        color: #ff6b6b;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# 2. Data Loading & Training (Cached)
# ==========================================
@st.cache_resource
def load_and_train():
  df = pd.read_csv("ai4i2020.csv")

  df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
  df["power_watts"] = (
      df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
  )
  df["overstrain_index"] = df["Torque [Nm]"] * df["Tool wear [min]"]
  df["torque_rpm_ratio"] = df["Torque [Nm]"] / (
      df["Rotational speed [rpm]"] + 1e-5
  )

  df_clean = df.drop(columns=["UDI", "Product ID"], errors="ignore")
  type_map = {"L": 1, "M": 2, "H": 3}
  if "Type" in df_clean.columns:
    df_clean["type_encoded"] = df_clean["Type"].map(type_map)
    df_clean = df_clean.drop(columns=["Type"])

  df_clean.columns = [re.sub(r"[\[\]<]", "", col) for col in df_clean.columns]

  drop_cols = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
  cols_to_drop = [col for col in drop_cols if col in df_clean.columns]

  X = df_clean.drop(columns=cols_to_drop)
  y = df_clean["Machine failure"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  scaler = RobustScaler()
  X_train_scaled = scaler.fit_transform(X_train)

  svc_model = SVC(kernel="rbf", class_weight="balanced", random_state=42)
  svc_model.fit(X_train_scaled, y_train)

  scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
  xgb_model = XGBClassifier(
      scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss"
  )
  xgb_model.fit(X_train, y_train)

  return scaler, svc_model, xgb_model, X.columns.tolist(), df


try:
  with st.spinner("⏳ Loading AI Pipeline..."):
    scaler, svc_model, xgb_model, feature_names, raw_df = load_and_train()
except Exception as e:
  st.error(f"❌ Error loading dataset: {e}")
  st.stop()


# ==========================================
# 3. Helper Function: Clean Plotly Theme Engine
# ==========================================
def apply_custom_chart_theme(
    fig, height=450, title="", show_legend=True, margin=None
):
  fig.update_layout(
      title_text=title if title else "",
      title_font=dict(size=16, family="Inter, sans-serif", color="#FFFFFF"),
      title_x=0,
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      font=dict(family="Inter, sans-serif", color="#A0A0AB", size=12),
      height=height,
      margin=margin if margin else dict(l=20, r=20, t=30 if title else 20, b=20),
      legend=dict(
          orientation="h",
          yanchor="bottom",
          y=1.02,
          xanchor="right",
          x=1,
          font=dict(size=12, color="#E0E0E0"),
      )
      if show_legend
      else None,
      hoverlabel=dict(
          bgcolor="#18181C",
          font_size=13,
          font_family="Inter, sans-serif",
          bordercolor="#FF1744",
      ),
  )
  fig.update_xaxes(
      showgrid=True,
      gridwidth=1,
      gridcolor="#22222A",
      zeroline=False,
      linecolor="#33333F",
  )
  fig.update_yaxes(
      showgrid=True,
      gridwidth=1,
      gridcolor="#22222A",
      zeroline=False,
      linecolor="#33333F",
  )
  return fig


# ==========================================
# 4. Popup Modal Dialog Function
# ==========================================
@st.dialog("📋 System Diagnostic Report", width="large")
def show_results_popup(is_failure, strategy_name):
  if is_failure:
    siren_url = "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg"
    st.markdown(
        f'<audio autoplay loop hidden src="{siren_url}"></audio>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='custom-alert-danger'>🚨 RUN! Fix it before we lose"
        " all our money! 💸</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="result-card-danger">
            <h2 style="margin:0; font-family:'Playfair Display', serif;">⚠️ HIGH RISK: Breakdown Imminent!</h2>
            <p style="margin-top:10px; margin-bottom:8px; color:#ffb3b3; font-size:1.05rem;">Active Strategy: <b>{strategy_name}</b></p>
            <p style="margin:0; font-size:1rem; color:#ffd1d1; font-style: italic;">
                Inspect the sensors immediately! Stopping the line now is cheaper than replacing the entire engine! 🛠️💥
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        "<div class='custom-alert-safe'>✨ All Good! Grab a coffee and"
        " relax! ☕</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="result-card-safe">
            <h2 style="margin:0; font-family:'Playfair Display', serif;">✅ OPTIMAL: Smooth Operations</h2>
            <p style="margin-top:10px; margin-bottom:8px; color:#b3ffcc; font-size:1.05rem;">Active Strategy: <b>{strategy_name}</b></p>
            <p style="margin:0; font-size:1rem; color:#d4edda; font-style: italic;">
                The machine is chilling and working like a charm. Zero drama, zero financial loss today! 😎
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# 5. Header & Sidebar Strategy Setup
# ==========================================
st.markdown(
    "<div class='main-title'>Predictive Maintenance</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-title'>Early Warning System — Industrial AI & Smart"
    " Manufacturing</div>",
    unsafe_allow_html=True,
)

st.sidebar.header("🎯 Deployment Strategy")

option_1 = "Option 1: High Efficiency (XGBoost Default)"
option_2 = "Option 2 ⭐ Recommended: The Balanced (XGBoost @ 0.07)"
option_3 = "Option 3: Extreme Safety (Support Vector Classifier)"

selected_option = st.sidebar.radio(
    "Select Strategy:", (option_1, option_2, option_3), index=1
)

if selected_option == option_1:
  st.sidebar.markdown(
      """
    <div class="protocol-card">
        <div class="protocol-title">⚡ High Efficiency Protocol</div>
        <div class="protocol-desc"><b>Strategy:</b> Minimizes False Alarms & Unnecessary Inspections. Keeps lines running continuously.</div>
        <div class="protocol-target"><b>Best Suited For:</b> High-volume assembly lines & continuous manufacturing where stopping causes massive immediate financial loss.</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

elif selected_option == option_2:
  st.sidebar.markdown(
      """
    <div class="protocol-card">
        <div class="protocol-title">⚖️ The Balanced Protocol</div>
        <div class="protocol-desc"><b>Strategy:</b> Golden Middle — Early Failure Detection with Controlled Inspections. Optimal ROI trade-off.</div>
        <div class="protocol-target"><b>Best Suited For:</b> Standard manufacturing plants seeking highest overall ROI and smooth day-to-day workflow.</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

elif selected_option == option_3:
  st.sidebar.markdown(
      """
    <div class="protocol-card">
        <div class="protocol-title">🛡️ Extreme Safety Protocol</div>
        <div class="protocol-desc"><b>Strategy:</b> Zero Tolerance for Unnoticed Machine Failures. Prioritizes catching every potential breakdown risk.</div>
        <div class="protocol-target"><b>Best Suited For:</b> Ultra-critical, high-cost heavy machinery where a single failure leads to catastrophic damage or safety hazards.</div>
    </div>
    """,
      unsafe_allow_html=True,
  )

# ==========================================
# 6. Inputs & Diagnostics
# ==========================================
st.subheader("📥 Real-Time Sensor Inputs")

col1, col2, col3 = st.columns(3)

with col1:
  product_type = st.selectbox(
      "Product Type", options=["L", "M", "H"], index=0
  )
  air_temp = st.number_input(
      "Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0
  )

with col2:
  process_temp = st.number_input(
      "Process Temperature [K]",
      min_value=300.0,
      max_value=320.0,
      value=310.0,
  )
  rot_speed = st.number_input(
      "Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500
  )

with col3:
  torque = st.number_input(
      "Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0
  )
  tool_wear = st.number_input(
      "Tool Wear [min]", min_value=0, max_value=300, value=100
  )

temp_diff = process_temp - air_temp
power_watts = torque * rot_speed * (2 * np.pi / 60)
overstrain_index = torque * tool_wear
torque_rpm_ratio = torque / (rot_speed + 1e-5)
type_encoded = {"L": 1, "M": 2, "H": 3}[product_type]

input_dict = {
    "Air temperature K": air_temp,
    "Process temperature K": process_temp,
    "Rotational speed rpm": rot_speed,
    "Torque Nm": torque,
    "Tool wear min": tool_wear,
    "temp_diff": temp_diff,
    "power_watts": power_watts,
    "overstrain_index": overstrain_index,
    "torque_rpm_ratio": torque_rpm_ratio,
    "type_encoded": type_encoded,
}

input_df = pd.DataFrame([input_dict])[feature_names]

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Run Diagnostics"):
  is_failure = False

  if selected_option == option_1:
    prob = xgb_model.predict_proba(input_df)[0][1]
    is_failure = prob >= 0.50

  elif selected_option == option_2:
    prob = xgb_model.predict_proba(input_df)[0][1]
    is_failure = prob >= 0.07

  elif selected_option == option_3:
    input_scaled = scaler.transform(input_df)
    is_failure = svc_model.predict(input_scaled)[0] == 1

  show_results_popup(is_failure, selected_option)

# ==========================================
# 7. Visual Analytics Section
# ==========================================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.subheader("📊 Fleet Data Analytics & Sensor Insights")

tab1, tab2, tab3 = st.tabs([
    "📈 Operating Zone Scatter",
    "🍩 Failure Modes Donut",
    "🔍 Deep Sensor Analytics & Distributions",
])

# ------------------------------------------
# Tab 1: Scatter Plot
# ------------------------------------------
with tab1:
  st.markdown(
      "##### 🗺️ Operating Zone vs. Historical Failures (Current Input Highlighted)"
  )

  plot_df = raw_df.copy()
  plot_df["Status"] = plot_df["Machine failure"].map(
      {0: "Normal Operation", 1: "Failure Breakdown"}
  )

  available_cols = [
      c
      for c in plot_df.columns
      if c not in ["UDI", "Product ID", "Machine failure", "Status"]
      and pd.api.types.is_numeric_dtype(plot_df[c])
  ]

  col_x, col_y = st.columns(2)
  with col_x:
    x_axis = st.selectbox(
        "Select X-Axis Feature:",
        options=available_cols,
        index=0 if len(available_cols) > 0 else 0,
    )
  with col_y:
    y_axis = st.selectbox(
        "Select Y-Axis Feature:",
        options=available_cols,
        index=3 if len(available_cols) > 3 else 0,
    )

  fig_scatter = px.scatter(
      plot_df,
      x=x_axis,
      y=y_axis,
      color="Status",
      title="",
      color_discrete_map={
          "Normal Operation": "#00E676",
          "Failure Breakdown": "#FF1744",
      },
      opacity=0.35,
  )

  fig_scatter.update_traces(
      marker=dict(size=7, line=dict(width=0.5, color="#141417"))
  )

  def get_input_val(col_name):
    cleaned = re.sub(r"[\[\]<]", "", col_name)
    return input_dict.get(cleaned, 0)

  current_x = get_input_val(x_axis)
  current_y = get_input_val(y_axis)

  fig_scatter.add_scatter(
      x=[current_x],
      y=[current_y],
      mode="markers",
      marker=dict(size=36, color="rgba(255, 234, 0, 0.3)", symbol="circle"),
      showlegend=False,
      hoverinfo="skip",
  )

  fig_scatter.add_scatter(
      x=[current_x],
      y=[current_y],
      mode="markers",
      marker=dict(
          size=20,
          color="#FFEA00",
          symbol="star",
          line=dict(width=2, color="#FFFFFF"),
      ),
      name="📍 Current Active Input",
  )

  apply_custom_chart_theme(fig_scatter, height=480)
  st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------
# Tab 2: Donut Chart
# ------------------------------------------
with tab2:
  st.markdown("##### 🍕 Failure Modes Distribution Breakdown")

  failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
  existing_fail_cols = [c for c in failure_cols if c in raw_df.columns]

  if existing_fail_cols:
    failure_counts = raw_df[existing_fail_cols].sum().reset_index()
    failure_counts.columns = ["Failure Type", "Count"]

    type_names = {
        "TWF": "Tool Wear (TWF)",
        "HDF": "Heat Dissipation (HDF)",
        "PWF": "Power Loss (PWF)",
        "OSF": "Overstrain (OSF)",
        "RNF": "Random Failure (RNF)",
    }
    failure_counts["Failure Name"] = failure_counts["Failure Type"].map(
        lambda x: type_names.get(x, x)
    )
    total_failures = int(failure_counts["Count"].sum())

    col_pie, col_stats = st.columns([1.2, 0.8], gap="large")

    with col_pie:
      fig_pie = px.pie(
          failure_counts,
          values="Count",
          names="Failure Name",
          title="",
          hole=0.65,
          color_discrete_sequence=[
              "#FF1744",
              "#FF9100",
              "#FFEA00",
              "#D500F9",
              "#00E5FF",
          ],
      )

      fig_pie.update_traces(
          textposition="inside",
          textinfo="percent",
          hovertemplate=(
              "<b>%{label}</b><br>Count: <b>%{value}</b><br>Percentage:"
              " <b>%{percent}</b><extra></extra>"
          ),
          marker=dict(line=dict(color="#0D0D0F", width=3)),
      )

      fig_pie.add_annotation(
          text=(
              f"<span style='font-size:26px; font-weight:800;"
              f" color:#FFFFFF;'>{total_failures}</span><br><span"
              " style='font-size:12px; color:#A0A0AB;'>Total Failures</span>"
          ),
          x=0.5,
          y=0.5,
          showarrow=False,
      )

      apply_custom_chart_theme(fig_pie, height=450, show_legend=True)
      st.plotly_chart(fig_pie, use_container_width=True)

    with col_stats:
      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("###### 📊 Detailed Failure Metrics:")
      for _, row in failure_counts.iterrows():
        pct = (
            (row["Count"] / total_failures * 100) if total_failures > 0 else 0
        )
        st.markdown(
            f"• **{row['Failure Name']}:** `{row['Count']}` cases"
            f" (**{pct:.1f}%**)"
        )

  else:
    st.info("No failure mode columns found in dataset.")

# ------------------------------------------
# Tab 3: Feature Importance & Box Plots
# ------------------------------------------
with tab3:
  st.markdown("##### 🎯 Feature Importance (AI Sensor Weighting)")

  importances = xgb_model.feature_importances_
  fi_df = pd.DataFrame(
      {"Sensor Feature": feature_names, "Importance Rating": importances}
  ).sort_values(by="Importance Rating", ascending=True)

  fig_fi = px.bar(
      fi_df,
      x="Importance Rating",
      y="Sensor Feature",
      orientation="h",
      title="",
      color="Importance Rating",
      color_continuous_scale=[[0, "#2A0808"], [0.5, "#D32F2F"], [1, "#FF5252"]],
  )

  fig_fi.update_traces(
      marker_line_color="#141417", marker_line_width=1, opacity=0.9
  )
  fig_fi.update_coloraxes(showscale=False)

  apply_custom_chart_theme(fig_fi, height=380, show_legend=False)
  st.plotly_chart(fig_fi, use_container_width=True)

  st.markdown("<br><hr>", unsafe_allow_html=True)

  st.markdown("##### 📦 Interactive Sensor Outlier & Range Distributions")

  exclude_cols = [
      "UDI",
      "Machine failure",
      "TWF",
      "HDF",
      "PWF",
      "OSF",
      "RNF",
      "type_encoded",
  ]
  numerical_cols = [
      col
      for col in raw_df.select_dtypes(include=[np.number]).columns
      if col not in exclude_cols
  ]

  num_features = len(numerical_cols)
  ncols = 2
  nrows = math.ceil(num_features / ncols)

  fig_box = sp.make_subplots(
      rows=nrows,
      cols=ncols,
      subplot_titles=[f"<b>{col}</b>" for col in numerical_cols],
      vertical_spacing=0.12,
      horizontal_spacing=0.1,
  )

  for i, col in enumerate(numerical_cols):
    r = (i // ncols) + 1
    c = (i % ncols) + 1

    fig_box.add_trace(
        go.Box(
            y=raw_df[col],
            name="",
            marker_color="#FF1744",
            boxpoints="outliers",
            pointpos=0,
            jitter=0.3,
            line=dict(width=1.5, color="#FF5252"),
            fillcolor="rgba(211, 47, 47, 0.25)",
        ),
        row=r,
        col=c,
    )

  fig_box.update_layout(
      title_text="",
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      font=dict(family="Inter, sans-serif", color="#A0A0AB", size=11),
      height=300 * nrows,
      showlegend=False,
      margin=dict(l=30, r=30, t=50, b=30),
  )
  fig_box.update_xaxes(showgrid=False, showticklabels=False)
  fig_box.update_yaxes(
      showgrid=True, gridcolor="#22222A", gridwidth=1, linecolor="#33333F"
  )

  for annotation in fig_box["layout"]["annotations"]:
    annotation["font"] = dict(color="#FFFFFF", size=13)

  st.plotly_chart(fig_box, use_container_width=True)


# ==========================================
# 8. Meet Our Team Section (Fail-Safe Version)
# ==========================================
team_html = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
.linkedin-icon {
    color: #0a66c2;
    font-size: 1.4rem;
    margin-top: 10px;
    display: inline-block;
    transition: all 0.3s ease;
    text-decoration: none;
}
.linkedin-icon:hover {
    color: #004182;
    transform: scale(1.25);
    filter: drop-shadow(0 0 8px rgba(10, 102, 194, 0.6));
}
</style>

<div class="team-container">
    <div class="team-header">Meet Our Team</div>
    <div class="team-sub">The Engineers Behind This Predictive Maintenance Platform</div>
    
    <div class="team-grid">
        <!-- Member 1: Abanoub Shawket -->
        <div class="team-card">
            <div class="team-img-wrapper">
                <img src="https://i.postimg.cc/qBGDDmwy/487565599-1000372732069969-5213268035757435117-n.jpg" alt="Abanoub Shawket" class="team-img">
            </div>
            <div class="team-name">Abanoub Shawket</div>
            <div class="team-role">Software / ML Engineer</div>
            <a href="https://www.linkedin.com/in/bebo-shawket" target="_blank" class="linkedin-icon" title="LinkedIn Profile">
                <i class="fab fa-linkedin"></i>
            </a>
        </div>

        <!-- Member 2: Mohamed Bahaa -->
        <div class="team-card">
            <div class="team-img-wrapper">
                <img src="https://i.postimg.cc/qqjWnQqC/Whats-App-Image-2026-08-14-at-3-59-07-PM.jpg" alt="Mohamed Bahaa" class="team-img">
            </div>
            <div class="team-name">Mohamed Bahaa</div>
            <div class="team-role">Software Engineer</div>
            <a href="https://www.linkedin.com/in/mohamed-bahaa-a20736344?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank" class="linkedin-icon" title="LinkedIn Profile">
                <i class="fab fa-linkedin"></i>
            </a>
        </div>

        <!-- Member 3: Shahd Emad -->
        <div class="team-card">
            <div class="team-img-wrapper">
                <img src="https://i.postimg.cc/g2nby9vj/Whats-App-Image-2026-08-13-at-11-51-37-PM.jpg" alt="Shahd Emad" class="team-img">
            </div>
            <div class="team-name">Shahd Emad</div>
            <div class="team-role">ML Engineer</div>
            <a href="https://www.linkedin.com/in/shahd-emad-821108411?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank" class="linkedin-icon" title="LinkedIn Profile">
                <i class="fab fa-linkedin"></i>
            </a>
        </div>

        <!-- Member 4: Salma Omar -->
        <div class="team-card">
            <div class="team-img-wrapper">
                <img src="https://i.postimg.cc/yYn7XXtL/Whats-App-Image-2026-08-13-at-11-51-17-PM.jpg" alt="Salma Omar" class="team-img">
            </div>
            <div class="team-name">Salma Omar</div>
            <div class="team-role">Cybersecurity Engineer</div>
            <a href="https://www.linkedin.com/in/salma-omar-753925380?utm_source=share_via&utm_content=profile&utm_medium=member_ios" target="_blank" class="linkedin-icon" title="LinkedIn Profile">
                <i class="fab fa-linkedin"></i>
            </a>
        </div>
    </div>
</div>
"""

# السطر ده بيضمن مسح أي مسافات جانبة بتجبر Streamlit يعرض الكود كـ Text
clean_team_html = "\n".join([line.strip() for line in team_html.splitlines()])

st.markdown(clean_team_html, unsafe_allow_html=True)