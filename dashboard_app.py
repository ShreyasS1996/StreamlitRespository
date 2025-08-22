# app.py
# Streamlit Engine Emissions Analysis & Real-Time Style Dashboard
# Author: Your Name
# How to run: streamlit run app.py

import io
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# --------- SETTINGS ---------
# ----------------------------
st.set_page_config(
    page_title="Engine Test Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Small CSS for a live-look status pill and tighter visuals
st.markdown("""
<style>
div.block-container { padding-top: 1.2rem; }
.status-ok {background:#16a34a;color:white;padding:4px 10px;border-radius:20px;font-weight:600;}
.status-warn {background:#f59e0b;color:#111827;padding:4px 10px;border-radius:20px;font-weight:600;}
.kpi {border:1px solid #E5E7EB;border-radius:10px;padding:12px 16px;background:#F9FAFB;}
h1,h2,h3 { font-weight:700; }
.small { color:#6B7280; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# -------- UTILITIES ---------
# ----------------------------

@st.cache_data(show_spinner=False)
def load_excel(uploaded_file, sheet=None) -> pd.DataFrame:
    """Load Excel sheet into DataFrame, with some type parsing."""
    if sheet is None:
        # If 'Test_Data' exists, prefer it
        xls = pd.ExcelFile(uploaded_file)
        pick = "Test_Data" if "Test_Data" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(uploaded_file, sheet_name=pick, engine="openpyxl")
    else:
        df = pd.read_excel(uploaded_file, sheet_name=sheet, engine="openpyxl")

    # Try to parse timestamp column(s)
    for c in df.columns:
        if "timestamp" in c.lower() or "time" in c.lower() or "date" in c.lower():
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass
    return df


def ensure_columns(df: pd.DataFrame) -> dict:
    """
    Map column names to our internal names, tolerant to different labels.
    Returns a dict with keys we use and actual df column names, None if missing.
    """
    name_map = {
        "engine_number": ["Engine_Number", "Engine No", "EngineID", "EngineId", "Engine"],
        "batch_number":  ["Batch_Number", "Batch", "BatchID", "Batch Id"],
        "fuel_type":     ["Fuel_Type", "Fuel", "Engine_Type"],
        "test_cycle":    ["Test_Cycle", "Cycle", "Emission_Cycle"],
        "test_procedure":["Test_Procedure", "Procedure", "Proc_Type"],
        "timestamp":     ["Test_Timestamp", "Timestamp", "Time", "Date"],
        "rpm":           ["Engine_Speed_RPM", "RPM", "Speed_RPM"],
        "load_pct":      ["Engine_Load_%", "Load_%", "LoadPercent"],
        "nox":           ["NOx_(g_per_kWh)", "NOx", "NOx_gkWh"],
        "pm":            ["PM_(mg_per_kWh)", "PM_mgkWh", "PM"],
        "co":            ["CO_(g_per_kWh)", "CO", "CO_gkWh"],
        "hc":            ["HC_(g_per_kWh)", "HC", "HC_gkWh"],
        "egt":           ["Exhaust_Gas_Temp_C", "EGT_C", "EGT"],
        "o2":            ["O2_(vol_%)", "O2"],
        "afr":           ["AFR_(air_to_fuel)", "AFR"],
        "lambda":        ["Lambda_(ratio)", "Lambda"],
        # Optional pre-provided curves
        "torque":        ["Torque_Nm", "Torque (Nm)"],
        "power":         ["Power_kW", "Power (kW)"]
    }
    resolved = {}
    for k, candidates in name_map.items():
        resolved[k] = next((c for c in candidates if c in df.columns), None)
    return resolved


def compute_curves(df, cols, rated_torque_nm=None):
    """
    Ensure Torque_Nm and Power_kW exist.
    - If torque present: use it.
    - Else if load_pct and rated_torque provided: torque = load% * rated_torque
    - Power_kW = Torque(Nm) * RPM / 9550 (SI)
    """
    rpm_col = cols["rpm"]
    torque_col = cols["torque"]
    power_col = cols["power"]
    load_col = cols["load_pct"]

    df = df.copy()

    # Compute Torque if missing and we have load% + rated torque
    if torque_col is None:
        if load_col is not None and rated_torque_nm is not None:
            df["Torque_Nm"] = (df[load_col].astype(float) / 100.0) * float(rated_torque_nm)
            torque_col = "Torque_Nm"

    # Compute Power if missing but have torque & rpm
    if power_col is None and torque_col and rpm_col:
        df["Power_kW"] = (df[torque_col].astype(float) * df[rpm_col].astype(float)) / 9550.0
        power_col = "Power_kW"
    return df, {"torque": torque_col, "power": power_col}


def compliance_columns(df, cols, limits):
    """
    Add pass/fail columns for regulated pollutants using user limits.
    """
    df = df.copy()
    for pol, lim in limits.items():
        c = cols.get(pol)
        if c and lim is not None:
            df[f"{pol.upper()}_Pass"] = np.where(df[c] <= lim, "Pass", "Fail")
    if all(cols.get(p) for p in ["nox", "co", "hc", "pm"]):
        df["Overall_Pass"] = "Pass"
        for pol in ["nox", "co", "hc", "pm"]:
            if f"{pol.upper()}_Pass" in df.columns:
                df["Overall_Pass"] = np.where(df[f"{pol.upper()}_Pass"] == "Fail", "Fail", df["Overall_Pass"])
    return df


def filtered_df(df, cols, filters):
    """
    Apply sidebar filters including date range.
    """
    out = df.copy()
    if filters.get("fuel_type"):
        out = out[out[cols["fuel_type"]].isin(filters["fuel_type"])]
    if filters.get("engine_number"):
        out = out[out[cols["engine_number"]].isin(filters["engine_number"])]
    if filters.get("batch_number"):
        out = out[out[cols["batch_number"]].isin(filters["batch_number"])]
    if filters.get("test_cycle"):
        out = out[out[cols["test_cycle"]].isin(filters["test_cycle"])]
    if filters.get("test_procedure"):
        out = out[out[cols["test_procedure"]].isin(filters["test_procedure"])]
    # Date range
    if cols["timestamp"] and filters.get("date_range"):
        start, end = filters["date_range"]
        out = out[(out[cols["timestamp"]] >= start) & (out[cols["timestamp"]] <= end)]
    return out


def kpi_card(label, value, delta=None, helptext=None, good=True):
    color_cls = "status-ok" if good else "status-warn"
    st.markdown(f"""
    <div class="kpi">
        <div style="display:flex;align-items:center;gap:10px;">
            <span class="{color_cls}">{'OK' if good else 'ATTN'}</span>
            <div>
                <div style="font-size:0.95rem;color:#6B7280;">{label}</div>
                <div style="font-size:1.6rem;font-weight:700;">{value}</div>
                {"<div class='small'>"+str(helptext)+"</div>" if helptext else ""}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------
# --------- SIDEBAR ----------
# ----------------------------
st.sidebar.header("1) Upload your Excel")
uploaded = st.sidebar.file_uploader("Excel (.xlsx) with Test_Data", type=["xlsx"])
sheet_name = st.sidebar.text_input("Sheet name (optional, blank tries 'Test_Data' first)", "")

st.sidebar.header("2) Derived Curves (if not present)")
rated_torque_nm = st.sidebar.number_input("Rated torque (Nm) for load-based torque calc", min_value=0.0, value=400.0, step=10.0)

st.sidebar.header("3) Compliance Limits")
nox_lim = st.sidebar.number_input("NOx limit (g/kWh)", min_value=0.0, value=0.50, step=0.05)
co_lim  = st.sidebar.number_input("CO limit (g/kWh)",  min_value=0.0, value=1.50, step=0.10)
hc_lim  = st.sidebar.number_input("HC limit (g/kWh)",  min_value=0.0, value=0.15, step=0.01)
pm_lim  = st.sidebar.number_input("PM limit (mg/kWh)", min_value=0.0, value=10.0, step=0.5)

st.sidebar.header("4) Real-time look")
simulate_live = st.sidebar.toggle("Show refresh indicator (visual only)")
refresh_btn = st.sidebar.button("🔄 Refresh now")

# ----------------------------
# --------- MAIN UI ----------
# ----------------------------

st.title("🧪 Engine Test – Real‑Time Style Dashboard")

if not uploaded:
    st.info("Upload your Excel workbook to begin. Tip: use the `Engine_Emissions_Test_Data_100.xlsx` you generated earlier.")
    st.stop()

# Load data
df = load_excel(uploaded, sheet=sheet_name if sheet_name.strip() else None)
cols = ensure_columns(df)

# Validate minimum columns
required = ["engine_number", "batch_number", "fuel_type", "test_cycle", "rpm", "load_pct"]
missing = [k for k in required if cols[k] is None]
if missing:
    st.error(f"Your file is missing required columns for: {missing}. Please check column names in the Help.")
    st.write("Detected columns:", list(df.columns))
    st.stop()

# Compute torque/power if needed
df, curve_cols = compute_curves(df, cols, rated_torque_nm=rated_torque_nm)
cols.update(curve_cols)

# Add compliance columns
limits = {"nox": nox_lim, "co": co_lim, "hc": hc_lim, "pm": pm_lim}
df = compliance_columns(df, cols, limits)

# Sidebar filters from current data
st.sidebar.header("5) Filters")
fuel_vals = sorted(df[cols["fuel_type"]].dropna().unique())
eng_vals  = sorted(df[cols["engine_number"]].dropna().unique())
bat_vals  = sorted(df[cols["batch_number"]].dropna().unique())
cyc_vals  = sorted(df[cols["test_cycle"]].dropna().unique()) if cols["test_cycle"] else []
proc_vals = sorted(df[cols["test_procedure"]].dropna().unique()) if cols["test_procedure"] else []

date_range = None
if cols["timestamp"] and pd.api.types.is_datetime64_any_dtype(df[cols["timestamp"]]):
    min_d = pd.to_datetime(df[cols["timestamp"]]).min()
    max_d = pd.to_datetime(df[cols["timestamp"]]).max()
    date_range = st.sidebar.date_input("Test date range", value=(min_d.date(), max_d.date()))
    # Convert back to Timestamp
    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_range = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))

filters = {
    "fuel_type": st.sidebar.multiselect("Engine type (fuel)", options=fuel_vals, default=fuel_vals),
    "engine_number": st.sidebar.multiselect("Engine numbers", options=eng_vals, default=[]),
    "batch_number": st.sidebar.multiselect("Batch number", options=bat_vals, default=[]),
    "test_cycle": st.sidebar.multiselect("Test cycle type", options=cyc_vals, default=cyc_vals),
    "test_procedure": st.sidebar.multiselect("Test procedure", options=proc_vals, default=proc_vals),
    "date_range": date_range
}

# Apply filters
df_f = filtered_df(df, cols, filters)
if df_f.empty:
    st.warning("No rows after filters. Loosen your selections.")
    st.stop()

# ----------------------------
# ------- KPI SECTION --------
# ----------------------------
live_badge = "🟢 LIVE" if simulate_live else "⏺️ Snapshot"
st.subheader(f"Overview {live_badge}")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    kpi_card("Records (filtered)", f"{len(df_f):,}", helptext="Rows in current selection")
with col2:
    unique_eng = df_f[cols["engine_number"]].nunique()
    kpi_card("Engines (filtered)", f"{unique_eng}", helptext="Unique engine IDs")
with col3:
    if "Overall_Pass" in df_f.columns:
        pass_rate = (df_f["Overall_Pass"] == "Pass").mean()*100
        kpi_card("Overall Pass Rate", f"{pass_rate:.1f} %", good=pass_rate>=95, helptext="All pollutants")
with col4:
    if cols["nox"]:
        kpi_card("Avg NOx (g/kWh)", f"{df_f[cols['nox']].mean():.3f}", good=df_f[cols["nox"]].mean()<=nox_lim)
with col5:
    if cols["pm"]:
        kpi_card("Avg PM (mg/kWh)", f"{df_f[cols['pm']].mean():.1f}", good=df_f[cols["pm"]].mean()<=pm_lim)

# ----------------------------
# ------- CURVES TAB ---------
# ----------------------------
st.markdown("### Torque & Power Curves")

if cols["torque"] or cols["power"]:
    # Engine selector for curves
    eng_pick = st.multiselect("Select engine(s) for curve overlay", options=eng_vals, default=eng_vals[:3])
    curve_df = df_f[df_f[cols["engine_number"]].isin(eng_pick)] if eng_pick else df_f.copy()

    # Sort by RPM for nicer lines
    curve_df = curve_df.sort_values(by=cols["rpm"])

    c1, c2 = st.columns(2)
    with c1:
        if cols["torque"]:
            fig_t = px.line(
                curve_df, x=cols["rpm"], y=cols["torque"], color=cols["engine_number"],
                title="Torque Curve (Nm) vs RPM", markers=True,
                labels={cols["rpm"]: "RPM", cols["torque"]: "Torque (Nm)"}
            )
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Torque not available. Provide 'Engine_Load_%' and set a rated torque in sidebar to compute it.")

    with c2:
        if cols["power"]:
            fig_p = px.line(
                curve_df, x=cols["rpm"], y=cols["power"], color=cols["engine_number"],
                title="Power Curve (kW) vs RPM", markers=True,
                labels={cols["rpm"]: "RPM", cols["power"]: "Power (kW)"}
            )
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Power not available. Power is computed from Torque and RPM; ensure torque is available.")

else:
    st.info("No torque/power columns found and cannot compute from the provided data. "
            "Add 'Engine_Load_%' and set a rated torque to synthesize curves.")

# ----------------------------
# ------ EMISSIONS TAB -------
# ----------------------------
st.markdown("### Emissions Analysis")

em_col1, em_col2 = st.columns(2)

with em_col1:
    # Box distributions by fuel
    if cols["fuel_type"] and cols["nox"]:
        fig_box = px.box(
            df_f, x=cols["fuel_type"], y=cols["nox"], color=cols["fuel_type"],
            points="outliers", title="NOx Distribution by Fuel Type",
            labels={cols["fuel_type"]: "Fuel Type", cols["nox"]: "NOx (g/kWh)"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

with em_col2:
    # NOx vs EGT with OLS trendline (requires statsmodels)
    if cols["egt"] and cols["nox"]:
        fig_sc = px.scatter(
            df_f, x=cols["egt"], y=cols["nox"], color=cols["fuel_type"],
            trendline="ols", title="NOx vs Exhaust Gas Temperature (with Trendline)",
            labels={cols["egt"]: "EGT (°C)", cols["nox"]: "NOx (g/kWh)"}
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ----------------------------
# ------- BATCH CHART --------
# ----------------------------
st.markdown("### Batch Quality: Pass Rate & Mean NOx")

if cols["batch_number"] and cols["nox"]:
    g = df_f.groupby(cols["batch_number"]).agg(
        mean_nox=(cols["nox"], "mean"),
        pass_rate=("Overall_Pass", lambda s: (s == "Pass").mean() * 100 if "Overall_Pass" in df_f.columns else np.nan),
        count=(cols["nox"], "size")
    ).reset_index()

    fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
    fig_combo.add_trace(
        go.Bar(x=g[cols["batch_number"]], y=g["pass_rate"], name="Pass Rate (%)", marker_color="#6BB700"),
        secondary_y=False
    )
    fig_combo.add_trace(
        go.Scatter(x=g[cols["batch_number"]], y=g["mean_nox"], name="Mean NOx (g/kWh)",
                   mode="lines+markers", line=dict(color="#8A2BE2", width=3)),
        secondary_y=True
    )
    # NOx limit line
    fig_combo.add_hline(y=nox_lim, line_dash="dash", line_color="red", secondary_y=True)

    fig_combo.update_layout(title="Pass Rate by Batch (bars) + Mean NOx (line)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_combo.update_xaxes(title_text="Batch Number")
    fig_combo.update_yaxes(title_text="Pass Rate (%)", secondary_y=False, range=[0, 105])
    fig_combo.update_yaxes(title_text="Mean NOx (g/kWh)", secondary_y=True)
    st.plotly_chart(fig_combo, use_container_width=True)

# ----------------------------
# -------- DATA VIEW ---------
# ----------------------------
st.markdown("### Data (Filtered)")
st.dataframe(df_f.head(300), use_container_width=True, height=360)

# Download filtered
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered data (CSV)", data=csv, file_name="filtered_engine_data.csv", mime="text/csv")

# Optional: simple refresh indicator (visual only)
if simulate_live or refresh_btn:
    st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
