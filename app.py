import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

st.set_page_config(page_title="Powertrain Emissions & Performance Analytics", layout="wide")

# =============== Helpers ===============

@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def safe_num(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def headroom(measured, limit, lower_is_better=True):
    if pd.isna(measured) or pd.isna(limit):
        return np.nan
    return (limit - measured) if lower_is_better else (measured - limit)

def percent(x):
    return f"{100.0*x:.1f}%"

def _colpick(label, options, key, help_txt=None, required=False, default=None):
    if default and default in options:
        return st.selectbox(label, ["<none>"] + options, index=options.index(default)+1, key=key, help=help_txt)
    return st.selectbox(label, ["<none>"] + options, key=key, help=help_txt)

def available(df, name):
    return name in df.columns

# Regulatory limits (example values; adjust as needed)
REG_LIMITS = {
    "BS VI": {  # Diesel LCV example (g/kWh) — replace with your exact class/use-case
        "NOx": 0.46,
        "CO": 1.5,
        "HC": 0.13,
        "PM": 0.01,
    },
    "Euro VI": {  # Heavy-duty diesel (g/kWh)
        "NOx": 0.4,
        "CO": 1.5,
        "HC": 0.13,
        "PM": 0.01,
    },
    "EPA 2010": {  # US HDD (g/bhp-hr); converted to g/kWh (1 bhp-hr = 0.7457 kWh)
        "NOx": 0.2/0.7457,   # ≈ 0.268
        "CO": 15.5/0.7457,   # ≈ 20.79 (note: CO often in mg/m3 tailpipe – verify for your case)
        "HC": 0.14/0.7457,   # ≈ 0.188
        "PM": 0.01/0.7457,   # ≈ 0.0134
    },
}

# =============== Sidebar: Data & Mapping ===============

st.sidebar.title("Data & Setup")

uploaded = st.sidebar.file_uploader("Upload raw CSV", type=["csv"])
if uploaded:
    df = read_csv(uploaded)
else:
    # Minimal demo data if no file provided
    rng = np.random.default_rng(42)
    n = 2500
    rpm = rng.integers(800, 4200, n)
    torque = rng.uniform(20, 320, n)  # Nm
    power_kw = (rpm * torque) / 9550.0
    afr = rng.uniform(12, 20, n)
    egr = rng.uniform(0, 20, n)  # %
    inj_t = rng.uniform(-5, +10, n)  # crank° ATDC (+ retarded)
    egt = rng.uniform(180, 550, n)  # °C
    fuel_gps = rng.uniform(0.2, 3.5, n)  # g/s
    # Pollutants (synthetic)
    nox = 0.0008 * (egt**1.1) + 0.02*(afr-14.7) - 0.01*egr + 0.002*inj_t + rng.normal(0, 0.05, n)
    co = 0.25 + 0.02*(14.7-afr) + 0.002*np.maximum(0, -inj_t) + rng.normal(0, 0.03, n)
    pm = 0.002 + 0.00002*torque + 0.00001*np.maximum(0, 13.5-afr) + rng.normal(0, 0.001, n)
    ts = pd.date_range("2025-01-01", periods=n, freq="min")
    batch = rng.choice(list("ABCDEFG"), size=n)
    test_id = rng.integers(1, 12, n)

    df = pd.DataFrame({
        "timestamp": ts, "test_id": test_id, "batch": batch,
        "rpm": rpm, "torque_Nm": torque, "power_kW": power_kw,
        "AFR": afr, "EGR_pct": egr, "inj_timing_deg": inj_t,
        "EGT_C": egt, "fuel_flow_gps": fuel_gps,
        "NOx_gpkWh": np.clip(nox, 0, None),
        "CO_gpkWh": np.clip(co, 0, None),
        "PM_gpkWh": np.clip(pm, 0, None),
        # After-treatment (optional synthetic)
        "NOx_in_gpkWh": np.clip(nox*1.8, 0, None),
        "NOx_out_gpkWh": np.clip(nox*0.7, 0, None),
        "PM_in_gpkWh": np.clip(pm*1.7, 0, None),
        "PM_out_gpkWh": np.clip(pm*0.6, 0, None),
        "CO_in_gpkWh": np.clip(co*1.2, 0, None),
        "CO_out_gpkWh": np.clip(co*0.8, 0, None),
    })

st.sidebar.caption("Tip: Your CSV can have any column names. Map them below.")

cols = list(df.columns)

st.sidebar.subheader("Map required fields")
c_rpm   = _colpick("RPM", cols, "rpm", default="rpm")
c_trq   = _colpick("Torque [Nm]", cols, "trq", default="torque_Nm")
c_pwr   = _colpick("Power [kW]", cols, "pwr", default="power_kW")
c_fuel  = _colpick("Fuel flow [g/s]", cols, "fuel", default="fuel_flow_gps")

st.sidebar.subheader("Map pollutant columns (g/kWh)")
c_nox   = _colpick("NOx", cols, "nox", default="NOx_gpkWh")
c_co    = _colpick("CO", cols, "co", default="CO_gpkWh")
c_hc    = _colpick("HC (optional)", cols, "hc")
c_pm    = _colpick("PM", cols, "pm", default="PM_gpkWh")

st.sidebar.subheader("Map important parameters")
c_afr   = _colpick("AFR", cols, "afr", default="AFR")
c_egr   = _colpick("EGR [%]", cols, "egr", default="EGR_pct")
c_inj   = _colpick("Injection timing [°]", cols, "inj", default="inj_timing_deg")
c_egt   = _colpick("Exhaust Gas Temp [°C]", cols, "egt", default="EGT_C")

st.sidebar.subheader("Map time & grouping (optional)")
c_time  = _colpick("Timestamp", cols, "time", default="timestamp")
c_batch = _colpick("Batch", cols, "batch", default="batch")
c_test  = _colpick("Test ID", cols, "test", default="test_id")

st.sidebar.subheader("After-treatment (optional; g/kWh)")
c_nox_in  = _colpick("NOx In", cols, "nox_in", default="NOx_in_gpkWh")
c_nox_out = _colpick("NOx Out", cols, "nox_out", default="NOx_out_gpkWh")
c_pm_in   = _colpick("PM In", cols, "pm_in", default="PM_in_gpkWh")
c_pm_out  = _colpick("PM Out", cols, "pm_out", default="PM_out_gpkWh")
c_co_in   = _colpick("CO In", cols, "co_in", default="CO_in_gpkWh")
c_co_out  = _colpick("CO Out", cols, "co_out", default="CO_out_gpkWh")

# Clean mapping to None for "<none>"
def m(sel): return None if sel == "<none>" else sel

rpm_col, trq_col, pwr_col, fuel_col = map(m, [c_rpm, c_trq, c_pwr, c_fuel])
nox_col, co_col, hc_col, pm_col     = map(m, [c_nox, c_co, c_hc, c_pm])
afr_col, egr_col, inj_col, egt_col  = map(m, [c_afr, c_egr, c_inj, c_egt])
time_col, batch_col, test_col       = map(m, [c_time, c_batch, c_test])
nox_in_col, nox_out_col = map(m, [c_nox_in, c_nox_out])
pm_in_col,  pm_out_col  = map(m, [c_pm_in,  c_pm_out])
co_in_col,  co_out_col  = map(m, [c_co_in,  c_co_out])

st.sidebar.divider()
std_choice = st.sidebar.selectbox("Regulatory Standard for Compliance", list(REG_LIMITS.keys()), index=0)
limits = REG_LIMITS[std_choice]

# =============== Tabs ===============

st.title("Powertrain Emissions, Performance & Calibration Analytics")

tabs = st.tabs([
    "🔎 Data Preview",
    "✅ Compliance",
    "⚙️ Performance",
    "🔗 Correlation & RCA",
    "📈 Trend & Stability",
    "🧠 Optimization & Calibration",
    "📊 Reporting"
])

# =============== Data Preview ===============
with tabs[0]:
    st.subheader("Uploaded Data")
    st.dataframe(df.head(200), use_container_width=True)
    st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

    # Quick profile of mapped columns
    mapped_cols = {
        "RPM": rpm_col, "Torque": trq_col, "Power_kW": pwr_col, "Fuel_gps": fuel_col,
        "NOx": nox_col, "CO": co_col, "HC": hc_col, "PM": pm_col,
        "AFR": afr_col, "EGR_%": egr_col, "Inj_timing_deg": inj_col, "EGT_C": egt_col,
        "Timestamp": time_col, "Batch": batch_col, "Test_ID": test_col
    }
    st.write("**Column Mapping**")
    st.json({k: v if v else "—" for k, v in mapped_cols.items()})

# =============== Compliance ===============
with tabs[1]:
    st.subheader("Compliance Analysis")
    pols = ["NOx", "CO", "HC", "PM"]
    chosen = []
    for p, col in zip(pols, [nox_col, co_col, hc_col, pm_col]):
        if col:
            chosen.append((p, col))
    if not chosen:
        st.warning("Please map at least one pollutant column in the sidebar.")
    else:
        # Compute pass/fail vs selected standard
        rows = []
        for pname, pcol in chosen:
            limit = limits.get(pname)
            if limit is None:
                st.info(f"No limit set for {pname} under {std_choice}. Edit REG_LIMITS in app.")
                continue
            x = df[pcol].astype(float)
            passes = (x <= limit).sum()
            fails = (x > limit).sum()
            head = (limit - x).median()
            rows.append({
                "Pollutant": pname,
                "Limit (g/kWh)": round(limit, 4),
                "Pass count": int(passes),
                "Fail count": int(fails),
                "Pass rate": passes / max(1, (passes+fails)),
                "Median headroom (g/kWh)": round(head, 4)
            })
        if rows:
            summ = pd.DataFrame(rows)
            summ["Pass rate"] = summ["Pass rate"].apply(lambda r: f"{r*100:.1f}%")
            st.dataframe(summ, use_container_width=True)

            # Visual: compliance vs headroom
            fig = px.bar(pd.DataFrame(rows),
                         x="Pollutant", y="Pass rate",
                         hover_data=["Limit (g/kWh)", "Median headroom (g/kWh)"],
                         title=f"Pass Rate by Pollutant — {std_choice}")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.box(
                pd.DataFrame({name: df[col] for name, col in chosen}),
                title="Pollutant Distributions (g/kWh)")
            st.plotly_chart(fig2, use_container_width=True)

# =============== Performance ===============
with tabs[2]:
    st.subheader("Performance Analysis")
    if not (rpm_col and (trq_col or pwr_col)):
        st.warning("Map RPM and either Torque or Power to view performance curves.")
    else:
        # Power vs RPM (compute from torque if needed)
        if not pwr_col and trq_col:
            power_kW = (df[trq_col].astype(float) * df[rpm_col].astype(float)) / 9550.0
        else:
            power_kW = df[pwr_col].astype(float)

        fig1 = px.scatter(df, x=rpm_col, y=power_kW, trendline="lowess",
                          labels={rpm_col: "RPM", "y": "Power (kW)"},
                          title="Power vs RPM")
        st.plotly_chart(fig1, use_container_width=True)

        if trq_col:
            fig2 = px.scatter(df, x=rpm_col, y=trq_col, trendline="lowess",
                              labels={rpm_col: "RPM", trq_col: "Torque (Nm)"},
                              title="Torque vs RPM")
            st.plotly_chart(fig2, use_container_width=True)

        # BSFC map
        if fuel_col:
            # BSFC = (fuel_gps * 3600) / power_kW
            bsfc = (df[fuel_col].astype(float) * 3600.0) / np.where(power_kW <= 0, np.nan, power_kW)
            data = df[[rpm_col, trq_col]].copy() if trq_col else df[[rpm_col]].copy()
            data["BSFC_gpkWh"] = bsfc

            st.markdown("**BSFC Map (g/kWh)**")
            if trq_col:
                # Create a grid by binning RPM & Torque
                Rbins = st.slider("RPM bins", min_value=8, max_value=60, value=24, step=1)
                Tbms = st.slider("Torque bins", min_value=8, max_value=60, value=24, step=1)
                d = data.dropna().copy()
                d["rpm_bin"] = pd.cut(d[rpm_col], bins=Rbins)
                d["trq_bin"] = pd.cut(d[trq_col], bins=Tbms)
                pivot = d.groupby(["rpm_bin", "trq_bin"])["BSFC_gpkWh"].mean().unstack()
                # Convert bins to midpoints
                rpm_centers = pivot.index.map(lambda c: 0.5*(c.left+c.right))
                trq_centers = pivot.columns.map(lambda c: 0.5*(c.left+c.right))
                heat = go.Figure(data=go.Heatmap(
                    x=trq_centers, y=rpm_centers, z=pivot.values, colorbar_title="g/kWh"))
                heat.update_layout(xaxis_title="Torque (Nm)", yaxis_title="RPM",
                                   title="BSFC Heatmap")
                st.plotly_chart(heat, use_container_width=True)
            else:
                st.info("Torque not mapped — showing BSFC vs RPM.")
                fig3 = px.scatter(data, x=rpm_col, y="BSFC_gpkWh", trendline="lowess",
                                  title="BSFC vs RPM")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Map Fuel flow to enable BSFC.")

# =============== Correlation & RCA ===============
with tabs[3]:
    st.subheader("Correlation & Root Cause")
    pairs = []
    if nox_col and egt_col: pairs.append(("NOx vs EGT", egt_col, nox_col))
    if co_col and afr_col:  pairs.append(("CO vs AFR", afr_col, co_col))
    if pm_col and trq_col:  pairs.append(("PM vs Load (Torque)", trq_col, pm_col))

    if pairs:
        cols2 = st.columns(len(pairs))
        for i, (title, xcol, ycol) in enumerate(pairs):
            with cols2[i]:
                fig = px.scatter(df, x=xcol, y=ycol, trendline="ols", title=title)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Map NOx–EGT, CO–AFR, and PM–Torque to see the classic relationships.")

    # Full correlation matrix (numeric)
    num = df.select_dtypes(include=[np.number]).copy()
    if len(num.columns) >= 2:
        corr = num.corr(numeric_only=True).round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

    # Simple driver analysis: predict NOx/CO/PM from controls
    st.markdown("**Key Drivers via Standardized Linear Regression**")
    drivers_target = st.selectbox("Target pollutant", [p for p, c in [("NOx", nox_col), ("CO", co_col), ("PM", pm_col)] if c], index=0)
    target_col = {"NOx": nox_col, "CO": co_col, "PM": pm_col}[drivers_target]
    feature_cols = [c for c in [afr_col, egr_col, inj_col, egt_col, rpm_col, trq_col, pwr_col] if c]
    if len(feature_cols) >= 2 and target_col:
        data = df[feature_cols + [target_col]].dropna()
        if len(data) > 20:
            X = data[feature_cols].astype(float).values
            y = data[target_col].astype(float).values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            lr = LinearRegression().fit(Xs, y)
            coefs = pd.Series(lr.coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)
            st.write("**Coefficient (standardized) — sign shows direction, magnitude shows influence**")
            st.dataframe(coefs.to_frame("coef").round(3))
            # Mutual info (nonlinear signal)
            try:
                mi = mutual_info_regression(Xs, y, random_state=0)
                mi_s = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
                st.write("**Mutual Information (nonlinear influence)**")
                st.dataframe(mi_s.to_frame("MI").round(3))
            except Exception:
                pass
        else:
            st.info("Not enough rows after dropping NAs for driver analysis.")
    else:
        st.info("Map at least two features among AFR, EGR, Inj timing, EGT, RPM, Torque/Power plus a target pollutant.")

# =============== Trend & Stability ===============
with tabs[4]:
    st.subheader("Trend & Stability")
    if time_col and nox_col:
        d = df[[time_col, nox_col]].dropna().sort_values(time_col).copy()
        fig = px.line(d, x=time_col, y=nox_col, title="NOx over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Map Timestamp and NOx to see time trends.")

    # Batch-to-batch variation
    if batch_col and nox_col:
        fig = px.box(df.dropna(subset=[batch_col, nox_col]), x=batch_col, y=nox_col,
                     title="Batch-to-Batch Variation — NOx")
        st.plotly_chart(fig, use_container_width=True)

    # Control charts (Individuals & Moving Range)
    st.markdown("**Individuals & Moving Range Control Chart (NOx)**")
    if time_col and nox_col:
        d = df[[time_col, nox_col]].dropna().sort_values(time_col).copy()
        y = d[nox_col].astype(float).values
        if len(y) >= 10:
            mean = np.nanmean(y)
            mr = np.abs(np.diff(y))
            mrbar = np.nanmean(mr) if len(mr) else np.nan
            d2 = 1.128  # constant for MR(2)
            sigma = mrbar / d2 if (mrbar and not np.isnan(mrbar)) else np.nan
            UCL = mean + 3*sigma if sigma else np.nan
            LCL = mean - 3*sigma if sigma else np.nan

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d[time_col], y=y, mode="lines+markers", name="NOx"))
            fig.add_hline(y=mean, line_dash="dash", annotation_text="Mean", annotation_position="top left")
            if sigma:
                fig.add_hline(y=UCL, line_dash="dot", annotation_text="UCL")
                fig.add_hline(y=LCL, line_dash="dot", annotation_text="LCL")
            fig.update_layout(title="Individuals Chart — NOx", xaxis_title="Time", yaxis_title="g/kWh")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 10 sequential points for a stable control chart.")
    else:
        st.info("Map Timestamp and NOx to build control charts.")

# =============== Optimization & Calibration ===============
with tabs[5]:
    st.subheader("Optimization & Calibration Feedback")
    tips = []

    # Data-driven signs from regression on NOx
    target_col = nox_col
    feature_cols = [c for c in [afr_col, egr_col, inj_col, egt_col, rpm_col, trq_col] if c]
    if target_col and len(feature_cols) >= 2:
        data = df[feature_cols + [target_col]].dropna()
        if len(data) > 50:
            X = data[feature_cols].astype(float).values
            y = data[target_col].astype(float).values
            Xs = StandardScaler().fit_transform(X)
            lr = LinearRegression().fit(Xs, y)
            coefs = pd.Series(lr.coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)
            st.write("**NOx Driver Coefficients (standardized)**")
            st.dataframe(coefs.to_frame("coef").round(3))

            # Heuristic suggestions from signs
            sgn = np.sign(coefs)
            def has(name): return name in coefs.index

            # AFR
            if has(afr_col):
                if coefs[afr_col] > 0:
                    tips.append("NOx rises with AFR (lean). Consider slightly richer AFR near high‑NOx zones.")
                else:
                    tips.append("Lean mixtures don’t appear to raise NOx strongly — AFR headroom exists.")

            # EGR
            if has(egr_col):
                if coefs[egr_col] < 0:
                    tips.append("EGR reduces NOx — try small EGR increases in high‑NOx cells.")
                else:
                    tips.append("EGR increase may not help NOx here — check combustion stability/soot trade‑offs.")

            # Injection timing
            if has(inj_col):
                if coefs[inj_col] > 0:
                    tips.append("Retarding injection (more +° ATDC) seems to increase NOx. Consider slight advance.")
                else:
                    tips.append("Slight retard may help NOx; watch CO/PM.")

            # EGT
            if has(egt_col) and coefs[egt_col] > 0:
                tips.append("Higher EGT correlates with higher NOx — reduce load in hot zones or improve charge cooling.")

            # CO/PM guardrails if mapped
            guards = []
            if co_col:
                guards.append("monitor CO; avoid overly rich AFR and excessive retard")
            if pm_col:
                guards.append("watch PM; avoid rich λ and ensure injection timing stays within smoke limit")
            if guards:
                tips.append("While reducing NOx, " + " and ".join(guards) + ".")

        else:
            st.info("Need >50 valid rows to generate stronger driver-based suggestions.")
    else:
        st.info("Map NOx and at least two of AFR/EGR/Injection timing/EGT/RPM/Torque for optimization tips.")

    # After-treatment efficiency
    st.markdown("**After‑treatment Efficiency (SCR/DPF/TWC proxies)**")
    eff_rows = []
    def eff(in_col, out_col, name):
        if in_col and out_col and in_col in df.columns and out_col in df.columns:
            d = df[[in_col, out_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) >= 10:
                e = (d[in_col] - d[out_col]) / d[in_col]
                return name, float(np.nanmedian(e.clip(lower=0, upper=1)))
        return None

    for tup in [
        eff(nox_in_col, nox_out_col, "SCR NOx"),
        eff(pm_in_col,  pm_out_col,  "DPF PM"),
        eff(co_in_col,  co_out_col,  "TWC CO"),
    ]:
        if tup: eff_rows.append({"System": tup[0], "Median Efficiency": percent(tup[1])})

    if eff_rows:
        st.dataframe(pd.DataFrame(eff_rows), use_container_width=True)
    else:
        st.info("Map *In* and *Out* pollutant columns to compute after‑treatment efficiency (e.g., NOx_in/NOx_out).")

    if tips:
        st.markdown("### Suggested Calibration Actions")
        for t in tips:
            st.markdown(f"- {t}")

# =============== Reporting ===============
with tabs[6]:
    st.subheader("Reporting & Visualization")
    # Box plots for pollutant distribution
    pol_map = { "NOx": nox_col, "CO": co_col, "HC": hc_col, "PM": pm_col }
    sel_pols = [k for k,v in pol_map.items() if v]
    if sel_pols:
        melt = []
        for k in sel_pols:
            melt.append(pd.DataFrame({"Pollutant": k, "Value": df[pol_map[k]].astype(float)}))
        melt = pd.concat(melt, ignore_index=True).dropna()
        fig = px.box(melt, x="Pollutant", y="Value", title="Pollutant Distribution (g/kWh)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Map at least one pollutant to show box plots.")

    # Scatter relationships
    st.markdown("**Scatter: Parameter Relationships**")
    left, right = st.columns(2)
    with left:
        xcol = st.selectbox("X", [c for c in [afr_col, egr_col, inj_col, egt_col, rpm_col, trq_col, pwr_col] if c])
    with right:
        ycol = st.selectbox("Y", [c for c in [nox_col, co_col, pm_col, hc_col] if c])
    if xcol and ycol:
        fig = px.scatter(df, x=xcol, y=ycol, trendline="ols", title=f"{ycol} vs {xcol}")
        st.plotly_chart(fig, use_container_width=True)

    # Combo chart: Compliance vs Performance
    st.markdown("**Combo: Compliance vs Performance**")
    if rpm_col and (trq_col or pwr_col) and nox_col:
        perf = df[[rpm_col, trq_col] if trq_col else [rpm_col]].copy()
        if trq_col:
            perf["Power_kW"] = (df[trq_col].astype(float) * df[rpm_col].astype(float)) / 9550.0 if not pwr_col else df[pwr_col].astype(float)
        else:
            perf["Power_kW"] = df[pwr_col].astype(float)
        perf["NOx_gpkWh"] = df[nox_col].astype(float)
        perf = perf.dropna()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=perf[rpm_col], y=perf["Power_kW"], mode="markers", name="Power (kW)", yaxis="y1"))
        fig.add_trace(go.Scatter(x=perf[rpm_col], y=perf["NOx_gpkWh"], mode="markers", name="NOx (g/kWh)", yaxis="y2"))
        fig.update_layout(
            title="Power vs RPM (y1) & NOx vs RPM (y2)",
            xaxis=dict(title="RPM"),
            yaxis=dict(title="Power (kW)", side="left"),
            yaxis2=dict(title="NOx (g/kWh)", overlaying="y", side="right")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Map RPM, Power/Torque, and NOx to see the combo chart.")

# Footer
st.caption("Note: Regulatory limits here are illustrative. Replace values in REG_LIMITS with your exact class/size-cycle standards.")
