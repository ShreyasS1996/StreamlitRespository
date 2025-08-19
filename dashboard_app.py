import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime

st.set_page_config(page_title="Local Analytics Dashboard", layout="wide")

st.title("📊 Local Analytics Dashboard with Forecasting")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # Filters
    st.sidebar.header("🛠 Filters")

    # Dropdown filters for categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        unique_vals = df[col].dropna().unique().tolist()
        selected_vals = st.sidebar.multiselect(f"Filter by {col}", unique_vals, default=unique_vals)
        df = df[df[col].isin(selected_vals)]

    # Date filter
    date_cols = df.select_dtypes(include='datetime64').columns.tolist()
    if not date_cols:
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                date_cols.append(col)
                break
            except:
                continue

    if date_cols:
        date_col = date_cols[0]
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        start_date, end_date = st.sidebar.date_input("Select date range", [min_date, max_date])
        df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

    st.subheader("📈 Visualizations")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    chart_type = st.selectbox("Choose chart type", ["Line Chart", "Bar Chart", "Pie Chart"])

    if chart_type == "Line Chart":
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", numeric_cols)
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar Chart":
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", numeric_cols)
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart":
        pie_col = st.selectbox("Category column", cat_cols)
        pie_val = st.selectbox("Value column", numeric_cols)
        pie_data = df.groupby(pie_col)[pie_val].sum().reset_index()
        fig = px.pie(pie_data, names=pie_col, values=pie_val, title=f"{pie_val} distribution by {pie_col}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Forecasting with Prophet")

    if date_cols and numeric_cols:
        forecast_col = st.selectbox("Select column to forecast", numeric_cols)
        forecast_df = df[[date_col, forecast_col]].dropna()
        forecast_df.columns = ['ds', 'y']

        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.info("Please upload a CSV or Excel file to begin.")

