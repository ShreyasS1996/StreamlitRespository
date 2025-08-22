import io
import numpy as np
import pandas as pd
import streamlit as st

# Try Plotly (preferred); if not present, use Altair fallback
PLOTLY_OK = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    PLOTLY_OK = False
    import altair as alt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

st.set_page_config(page_title="Powertrain Emissions & Performance Analytics", layout="wide")

@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def percent(x): return f"{100.0*x:.1f}%"

# Example limits (edit to your program)
REG_LIMITS = {
    "BS VI": {"NOx": 0.46, "CO": 1.5, "HC": 0.13, "PM": 0.01},
    "Euro VI": {"NOx": 0.4, "CO": 1.5, "HC": 0.13, "PM": 0.01},
    "EPA 2010": {"NOx": 0.2/0.7457, "CO": 15.5/0.7457, "HC": 0.14/0.7457, "PM": 0.01/0.7457},
}

# ---------------- Sidebar: data ----------------
st.sidebar.title("Data & Setup")
uploaded = st.sidebar.file_uploader("Upload raw CSV", type=["csv"])
if uploaded:
    df = read_csv(uploaded)
else:
    # tiny synthetic demo if no CSV
    rng = np.random.default_rng(42); n = 1200
    rpm = rng.integers(800, 4200, n); trq = rng.uniform(20, 320, n)
    pwr = (rpm * trq) / 9550.0
    afr = rng.uniform(12, 20, n); egr = rng.uniform(0, 20, n)
    inj = rng.uniform(-5, 10, n); egt = rng.uniform(180, 550, n)
    fuel = rng.uniform(0.2, 3.5, n)
    nox = 0.0008*(egt**1.1) + 0.02*(afr-14.7) - 0.01*egr + 0.002*inj + rng.normal(0,0.05,n)
    co  = 0.25 + 0.02*(14.7-afr) + 0.002*np.maximum(0,-inj) + rng.normal(0,0.03,n)
    pm  = 0.002 + 0.00002*trq + 0.00001*np.maximum(0,13.5-afr) + rng.normal(0,0.001,n)
    ts = pd.date_range("2025-01-01", periods=n, freq="min")
    batch = rng.choice(list("ABCDEF"), size=n)
    df = pd.DataFrame({
        "timestamp": ts, "batch": batch, "test_id": rng.integers(1,9,n),
        "rpm": rpm, "torque_Nm": trq, "power_kW": pwr,
        "AFR": afr, "EGR_pct": egr, "inj_timing_deg": inj, "EGT_C": egt,
        "fuel_flow_gps": fuel,
        "NOx_gpkWh": np.clip(nox,0,None), "CO_gpkWh": np.clip(co,0,None), "PM_gpkWh": np.clip(pm,0,None),
        "NOx_in_gpkWh": np.clip(nox*1.8,0,None), "NOx_out_gpkWh": np.clip(nox*0.7,0,None),
        "PM_in_gpkWh": np.clip(pm*1.7,0,None), "PM_out_gpkWh": np.clip(pm*0.6,0,None),
        "CO_in_gpkWh": np.clip(co*1.2,0,None), "CO_out_gpkWh": np.clip(co*0.8,0,None),
    })

cols = list(df.columns)

def _pick(label, default, key):
    options = ["<none>"] + cols
    if default in cols:
        return st.sidebar.selectbox(label, options, index=cols.index(default)+1, key=key)
    return st.sidebar.selectbox(label, options, key=key)
def m(x): return None if x == "<none>" else x

st.sidebar.caption("Map your CSV columns to analysis fields")
c_rpm = _pick("RPM", "rpm", "rpm"); c_trq = _pick("Torque [Nm]", "torque_Nm", "trq")
c_pwr = _pick("Power [kW]", "power_kW", "pwr"); c_fuel = _pick("Fuel flow [g/s]", "fuel_flow_gps", "fuel")
c_nox = _pick("NOx (g/kWh)", "NOx_gpkWh", "nox"); c_co = _pick("CO (g/kWh)", "CO_gpkWh", "co")
c_hc  = _pick("HC (g/kWh)", None, "hc"); c_pm = _pick("PM (g/kWh)", "PM_gpkWh", "pm")
c_afr = _pick("AFR", "AFR", "afr"); c_egr = _pick("EGR [%]", "EGR_pct", "egr")
c_inj = _pick("Injection timing [°]", "inj_timing_deg", "inj"); c_egt = _pick("EGT [°C]", "EGT_C", "egt")
c_time = _pick("Timestamp", "timestamp", "time"); c_batch = _pick("Batch", "batch", "batch"); c_test = _pick("Test ID", "test_id", "test")
c_nox_in = _pick("NOx In", "NOx_in_gpkWh", "nxi"); c_nox_out = _pick("NOx Out", "NOx_out_gpkWh", "nxo")
c_pm_in = _pick("PM In", "PM_in_gpkWh", "pmi"); c_pm_out = _pick("PM Out", "PM_out_gpkWh", "pmo")
c_co_in = _pick("CO In", "CO_in_gpkWh", "coi"); c_co_out = _pick("CO Out", "CO_out_gpkWh", "coo")

rpm_col, trq_col, pwr_col, fuel_col = map(m, [c_rpm, c_trq, c_pwr, c_fuel])
nox_col, co_col, hc_col, pm_col = map(m, [c_nox, c_co, c_hc, c_pm])
afr_col, egr_col, inj_col, egt_col = map(m, [c_afr, c_egr, c_inj, c_egt])
time_col, batch_col, test_col = map(m, [c_time, c_batch, c_test])
nox_in_col, nox_out_col = map(m, [c_nox_in, c_nox_out])
pm_in_col, pm_out_col = map(m, [c_pm_in, c_pm_out])
co_in_col, co_out_col = map(m, [c_co_in, c_co_out])

std_choice = st.sidebar.selectbox("Regulatory Standard", list(REG_LIMITS.keys()))
limits = REG_LIMITS[std_choice]

st.title("Powertrain Emissions, Performance & Calibration Analytics")
tabs = st.tabs(["🔎 Data Preview","✅ Compliance","⚙️ Performance","🔗 Correlation & RCA","📈 Trend & Stability","🧠 Optimization & Calibration","📊 Reporting"])

# ------------- helper plotting wrappers -------------
def scatter(df, x, y, title):
    if PLOTLY_OK:
        return st.plotly_chart(px.scatter(df, x=x, y=y, title=title), use_container_width=True)
    chart = alt.Chart(df).mark_circle(opacity=0.6).encode(x=x, y=y).properties(title=title).interactive()
    st.altair_chart(chart, use_container_width=True)

def line(df, x, y, title):
    if PLOTLY_OK:
        return st.plotly_chart(px.line(df, x=x, y=y, title=title), use_container_width=True)
    chart = alt.Chart(df).mark_line().encode(x=x, y=y).properties(title=title)
    st.altair_chart(chart, use_container_width=True)

def box(df_long, x, y, title):
    if PLOTLY_OK:
        return st.plotly_chart(px.box(df_long, x=x, y=y, title=title), use_container_width=True)
    chart = alt.Chart(df_long).mark_boxplot().encode(x=x, y=y).properties(title=title)
    st.altair_chart(chart, use_container_width=True)

def heatmap_from_pivot(pivot, xname, yname, title):
    if PLOTLY_OK:
        fig = go.Figure(data=go.Heatmap(x=pivot.columns, y=pivot.index, z=pivot.values, colorbar_title="Value"))
        fig.update_layout(xaxis_title=xname, yaxis_title=yname, title=title)
        return st.plotly_chart(fig, use_container_width=True)
    dfp = pivot.reset_index().melt(id_vars=pivot.index.name, var_name=xname, value_name="val")
    chart = alt.Chart(dfp).mark_rect().encode(x=xname, y=pivot.index.name, tooltip=["val"]).properties(title=title)
    st.altair_chart(chart, use_container_width=True)

# ---------------- Tab 1: Data ----------------
with tabs[0]:
    st.subheader("Uploaded Data")
    st.dataframe(df.head(200), use_container_width=True)
    st.caption(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.write("**Column Mapping**")
    st.json({
        "RPM": rpm_col, "Torque": trq_col, "Power_kW": pwr_col, "Fuel_gps": fuel_col,
        "NOx": nox_col, "CO": co_col, "HC": hc_col, "PM": pm_col,
        "AFR": afr_col, "EGR_%": egr_col, "Inj_timing_deg": inj_col, "EGT_C": egt_col,
        "Timestamp": time_col, "Batch": batch_col, "Test_ID": test_col
    })

# ---------------- Tab 2: Compliance ----------------
with tabs[1]:
    st.subheader("Compliance Analysis")
    chosen = [(p, c) for p, c in [("NOx", nox_col), ("CO", co_col), ("HC", hc_col), ("PM", pm_col)] if c]
    if not chosen:
        st.warning("Map at least one pollutant column.")
    else:
        rows = []
        for pname, pcol in chosen:
            limit = limits.get(pname)
            if limit is None:
                st.info(f"No limit set for {pname} under {std_choice}. Edit REG_LIMITS in app.")
                continue
            x = pd.to_numeric(df[pcol], errors="coerce")
            valid = x.dropna()
            passes = (valid <= limit).sum()
            fails = (valid > limit).sum()
            rows.append({
                "Pollutant": pname,
                "Limit (g/kWh)": round(limit,4),
                "Pass count": int(passes),
                "Fail count": int(fails),
                "Pass rate": f"{(passes/max(1,passes+fails))*100:.1f}%",
                "Median headroom (g/kWh)": round((limit - valid).median(),4)
            })
        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)
        # distributions
        box(pd.concat([pd.DataFrame({"Pollutant": p, "Value": pd.to_numeric(df[c], errors="coerce")}) for p,c in chosen], ignore_index=True),
            "Pollutant","Value","Pollutant Distributions (g/kWh)")

# ---------------- Tab 3: Performance ----------------
with tabs[2]:
    st.subheader("Performance Analysis")
    if not (rpm_col and (trq_col or pwr_col)):
        st.warning("Map RPM and either Torque or Power.")
    else:
        power_kW = pd.to_numeric(df[pwr_col], errors="coerce") if pwr_col else (pd.to_numeric(df[trq_col], errors="coerce")*pd.to_numeric(df[rpm_col], errors="coerce"))/9550.0
        scatter(pd.DataFrame({rpm_col: df[rpm_col], "Power_kW": power_kW}), rpm_col, "Power_kW", "Power vs RPM")
        if trq_col:
            scatter(df, rpm_col, trq_col, "Torque vs RPM")

        if fuel_col:
            bsfc = (pd.to_numeric(df[fuel_col], errors="coerce")*3600.0) / np.where(power_kW<=0, np.nan, power_kW)
            if trq_col:
                Rbins = st.slider("RPM bins", 8, 60, 24); Tbins = st.slider("Torque bins", 8, 60, 24)
                d = pd.DataFrame({rpm_col: df[rpm_col], trq_col: df[trq_col], "BSFC": bsfc}).dropna()
                d["rpm_bin"] = pd.cut(pd.to_numeric(d[rpm_col], errors="coerce"), bins=Rbins)
                d["trq_bin"] = pd.cut(pd.to_numeric(d[trq_col], errors="coerce"), bins=Tbins)
                piv = d.groupby(["rpm_bin","trq_bin"])["BSFC"].mean().unstack()
                # relabel to bin midpoints
                piv.index.name = "RPM"; piv.columns.name = "Torque (Nm)"
                piv.index = [0.5*(b.left+b.right) for b in piv.index]
                piv.columns = [0.5*(b.left+b.right) for b in piv.columns]
                heatmap_from_pivot(piv, "Torque (Nm)", "RPM", "BSFC Heatmap (g/kWh)")
            else:
                scatter(pd.DataFrame({rpm_col: df[rpm_col], "BSFC_gpkWh": bsfc}), rpm_col, "BSFC_gpkWh", "BSFC vs RPM")
        else:
            st.info("Map Fuel flow to enable BSFC.")

# ---------------- Tab 4: Correlation & RCA ----------------
with tabs[3]:
    st.subheader("Correlation & Root Cause")
    if nox_col and egt_col: scatter(df, egt_col, nox_col, "NOx vs EGT")
    if co_col and afr_col:  scatter(df, afr_col, co_col, "CO vs AFR")
    if pm_col and trq_col:  scatter(df, trq_col, pm_col, "PM vs Load (Torque)")

    num = df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True).round(2)
        if PLOTLY_OK:
            st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix"), use_container_width=True)
        else:
            corr_reset = corr.reset_index().melt("index")
            heat = alt.Chart(corr_reset).mark_rect().encode(x="index:O", y="variable:O", color="value:Q", tooltip=["index","variable","value"])
            st.altair_chart(heat.properties(title="Correlation Matrix"), use_container_width=True)

    st.markdown("**Key Drivers via Standardized Linear Regression**")
    targets = [p for p,c in [("NOx", nox_col), ("CO", co_col), ("PM", pm_col)] if c]
    if targets:
        drivers_target = st.selectbox("Target pollutant", targets)
        tcol = {"NOx": nox_col, "CO": co_col, "PM": pm_col}[drivers_target]
        feats = [c for c in [afr_col, egr_col, inj_col, egt_col, rpm_col, trq_col, pwr_col] if c]
        if len(feats) >= 2:
            data = df[feats+[tcol]].dropna()
            if len(data) > 20:
                X = data[feats].astype(float).values; y = data[tcol].astype(float).values
                Xs = StandardScaler().fit_transform(X); lr = LinearRegression().fit(Xs, y)
                coefs = pd.Series(lr.coef_, index=feats).sort_values(key=np.abs, ascending=False)
                st.dataframe(coefs.to_frame("coef").round(3))
                try:
                    mi = mutual_info_regression(Xs, y, random_state=0)
                    st.dataframe(pd.Series(mi, index=feats).sort_values(ascending=False).to_frame("MI").round(3))
                except Exception:
                    pass
        else:
            st.info("Map at least two features among AFR/EGR/Injection timing/EGT/RPM/Torque/Power plus a target pollutant.")

# ---------------- Tab 5: Trend & Stability ----------------
with tabs[4]:
    st.subheader("Trend & Stability")
    if time_col and nox_col:
        d = df[[time_col, nox_col]].dropna().sort_values(time_col)
        line(d, time_col, nox_col, "NOx over Time")
    if batch_col and nox_col:
        if PLOTLY_OK:
            st.plotly_chart(px.box(df.dropna(subset=[batch_col, nox_col]), x=batch_col, y=nox_col, title="Batch-to-Batch Variation — NOx"), use_container_width=True)
        else:
            box(df.dropna(subset=[batch_col, nox_col])[[batch_col, nox_col]], batch_col, nox_col, "Batch-to-Batch Variation — NOx")

    st.markdown("**Individuals & Moving Range Control Chart (NOx)**")
    if time_col and nox_col:
        d = df[[time_col, nox_col]].dropna().sort_values(time_col)
        y = d[nox_col].astype(float).values
        if len(y) >= 10:
            mean = np.nanmean(y); mr = np.abs(np.diff(y))
            mrbar = np.nanmean(mr) if len(mr) else np.nan; d2 = 1.128
            sigma = mrbar/d2 if (mrbar and not np.isnan(mrbar)) else np.nan
            UCL = mean + 3*sigma if sigma else np.nan; LCL = mean - 3*sigma if sigma else np.nan
            d = d.assign(mean=mean, UCL=UCL, LCL=LCL)
            if PLOTLY_OK:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=d[time_col], y=d[nox_col], mode="lines+markers", name="NOx"))
                fig.add_hline(y=mean, line_dash="dash", annotation_text="Mean")
                if sigma: fig.add_hline(y=UCL, line_dash="dot", annotation_text="UCL"); fig.add_hline(y=LCL, line_dash="dot", annotation_text="LCL")
                fig.update_layout(title="Individuals Chart — NOx", xaxis_title="Time", yaxis_title="g/kWh")
                st.plotly_chart(fig, use_container_width=True)
            else:
                base = alt.Chart(d).encode(x=time_col)
                st.altair_chart((base.mark_line().encode(y=nox_col) +
                                 base.mark_rule(strokeDash=[4,4]).encode(y="mean:Q") +
                                 (base.mark_rule(strokeDash=[2,2]).encode(y="UCL:Q") if sigma else alt.Chart()) +
                                 (base.mark_rule(strokeDash=[2,2]).encode(y="LCL:Q") if sigma else alt.Chart())
                                 ).properties(title="Individuals Chart — NOx"),
                                use_container_width=True)
        else:
            st.info("Need at least 10 sequential points for a control chart.")

# ---------------- Tab 6: Optimization & Calibration ----------------
with tabs[5]:
    st.subheader("Optimization & Calibration Feedback")
    tips = []
    target_col = nox_col
    feats = [c for c in [afr_col, egr_col, inj_col, egt_col, rpm_col, trq_col] if c]
    if target_col and len(feats) >= 2:
        data = df[feats + [target_col]].dropna()
        if len(data) > 50:
            X = data[feats].astype(float).values; y = data[target_col].astype(float).values
            Xs = StandardScaler().fit_transform(X); lr = LinearRegression().fit(Xs, y)
            coefs = pd.Series(lr.coef_, index=feats).sort_values(key=np.abs, ascending=False)
            st.write("**NOx Driver Coefficients (standardized)**")
            st.dataframe(coefs.to_frame("coef").round(3))
            def has(name): return name in coefs.index
            if has(afr_col) and coefs[afr_col] > 0: tips.append("NOx rises with leaner AFR — enrich slightly in high‑NOx cells.")
            if has(egr_col) and coefs[egr_col] < 0: tips.append("EGR appears effective — try modest EGR increases in hot zones.")
            if has(inj_col) and coefs[inj_col] > 0: tips.append("Retard seems to raise NOx — consider slight advance within limits.")
            if has(egt_col) and coefs[egt_col] > 0: tips.append("High EGT drives NOx — reduce load or improve charge cooling.")
            guards = []
            if co_col: guards.append("watch CO; avoid too‑rich λ or excessive retard")
            if pm_col: guards.append("watch PM; avoid smoke limit and maintain adequate rail/boost")
            if guards: tips.append("While reducing NOx, " + " and ".join(guards) + ".")
        else:
            st.info("Need >50 valid rows for stronger driver suggestions.")
    else:
        st.info("Map NOx and ≥2 of AFR/EGR/Injection timing/EGT/RPM/Torque for tips.")

    st.markdown("**After‑treatment Efficiency (median)**")
    def median_eff(in_col, out_col, name):
        if in_col and out_col and in_col in df.columns and out_col in df.columns:
            d = df[[in_col, out_col]].replace([np.inf,-np.inf], np.nan).dropna()
            if len(d) >= 10:
                e = (d[in_col]-d[out_col])/d[in_col]
                return {"System": name, "Median Efficiency": percent(float(np.nanmedian(np.clip(e,0,1))))}
    eff_rows = list(filter(None, [
        median_eff(nox_in_col, nox_out_col, "SCR NOx"),
        median_eff(pm_in_col, pm_out_col, "DPF PM"),
        median_eff(co_in_col, co_out_col, "TWC CO"),
    ]))
    if eff_rows: st.dataframe(pd.DataFrame(eff_rows), use_container_width=True)
    else: st.info("Map *_in and *_out columns to compute SCR/DPF/TWC efficiencies.")

# ---------------- Tab 7: Reporting ----------------
with tabs[6]:
    st.subheader("Reporting & Visualization")
    pol_map = {"NOx": nox_col, "CO": co_col, "HC": hc_col, "PM": pm_col}
    sel = [k for k,v in pol_map.items() if v]
    if sel:
        melted = pd.concat([pd.DataFrame({"Pollutant": k, "Value": pd.to_numeric(df[v], errors="coerce")}) for k,v in pol_map.items() if v], ignore_index=True).dropna()
        box(melted, "Pollutant", "Value", "Pollutant Distribution (g/kWh)")
    else:
        st.info("Map at least one pollutant to show box plots.")

    st.markdown("**Scatter: Parameter Relationships**")
    left, right = st.columns(2)
    Xopts = [c for c in [afr_col, egr_col, inj_col, egt_col, rpm_col, trq_col, pwr_col] if c]
    Yopts = [c for c in [nox_col, co_col, pm_col, hc_col] if c]
    xcol = left.selectbox("X", Xopts) if Xopts else None
    ycol = right.selectbox("Y", Yopts) if Yopts else None
    if xcol and ycol:
        scatter(df, xcol, ycol, f"{ycol} vs {xcol}")

    if rpm_col and (trq_col or pwr_col) and nox_col:
        perf = pd.DataFrame({rpm_col: df[rpm_col]})
        perf["Power_kW"] = (pd.to_numeric(df[trq_col], errors="coerce")*pd.to_numeric(df[rpm_col], errors="coerce"))/9550.0 if trq_col and not pwr_col else pd.to_numeric(df[pwr_col], errors="coerce")
        perf["NOx_gpkWh"] = pd.to_numeric(df[nox_col], errors="coerce")
        perf = perf.dropna()
        if PLOTLY_OK:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=perf[rpm_col], y=perf["Power_kW"], mode="markers", name="Power (kW)", yaxis="y1"))
            fig.add_trace(go.Scatter(x=perf[rpm_col], y=perf["NOx_gpkWh"], mode="markers", name="NOx (g/kWh)", yaxis="y2"))
            fig.update_layout(title="Power vs RPM (y1) & NOx vs RPM (y2)", xaxis=dict(title="RPM"),
                              yaxis=dict(title="Power (kW)", side="left"),
                              yaxis2=dict(title="NOx (g/kWh)", overlaying="y", side="right"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Two-layer Altair: scale independent via transform
            p1 = alt.Chart(perf).mark_point().encode(x=rpm_col, y="Power_kW")
            p2 = alt.Chart(perf).mark_point().encode(x=rpm_col, y=alt.Y("NOx_gpkWh", axis=alt.Axis(titleColor="gray")))
            st.altair_chart(alt.layer(p1, p2).resolve_scale(y='independent').properties(title="Power vs RPM & NOx vs RPM"), use_container_width=True)

st.caption("If Plotly is unavailable, charts auto‑fallback to Altair. For full features (LOWESS/OLS trendlines), keep Plotly + statsmodels installed.")
