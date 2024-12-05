import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# App Title
st.title("Time Series Analysis and Forecasting App")

# Sidebar Inputs
st.sidebar.header("Upload and Configure")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
date_column = st.sidebar.text_input("Enter Date Column Name (e.g., 'Date')")
value_column = st.sidebar.text_input("Enter Time Series Column Name (e.g., 'Value')")

if uploaded_file:
    # Load the data
    data = pd.read_csv(uploaded_file, parse_dates=[date_column], index_col=date_column)
    time_series = data[value_column]

    st.write("### Uploaded Data")
    st.write(data.head())

    # Decompose Time Series
    st.write("### Time Series Decomposition")
    st.write("Analyzing trend, seasonality, and residuals...")
    decomposition = seasonal_decompose(time_series, model="additive", period=12)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    decomposition.trend.plot(ax=ax[0], title="Trend")
    decomposition.seasonal.plot(ax=ax[1], title="Seasonality")
    decomposition.resid.plot(ax=ax[2], title="Residuals")
    plt.tight_layout()
    st.pyplot(fig)

    # ARIMA Model Suggestion using Statsmodels
    st.write("### ARIMA Model Suggestion")
    st.write("Fitting ARIMA model with different parameters...")
    
    # Use statsmodels ARIMA instead of pmdarima
    arima_order = (1, 1, 1)  # Example order, you can modify this as needed
    arima_model = ARIMA(time_series, order=arima_order)
    arima_model_fit = arima_model.fit()

    st.write("Suggested ARIMA Order (p, d, q):", arima_order)

    # Prophet Forecasting
    st.write("### Prophet Forecasting")
    # Prepare data for Prophet
    prophet_data = data.reset_index()
    prophet_data.columns = ['ds', 'y']  # Prophet expects columns named 'ds' (datetime) and 'y' (values)

    forecast_steps = st.sidebar.number_input("Number of steps to forecast", min_value=1, value=12)
    model_choice = st.sidebar.radio("Choose a Model for Forecasting", ["ARIMA", "Prophet"])

    if st.sidebar.button("Generate Forecast"):
        if model_choice == "ARIMA":
            st.write("### ARIMA Forecast")
            with st.spinner("Generating forecast using ARIMA..."):
                # Generate forecast using ARIMA model
                forecast = arima_model_fit.forecast(steps=forecast_steps)
                forecast_index = pd.date_range(time_series.index[-1], periods=forecast_steps + 1, freq="M")[1:]

                # Create a DataFrame for ARIMA forecast
                forecast_df = pd.DataFrame({
                    "Forecast": forecast
                }, index=forecast_index)

                st.write(forecast_df)

                # Plot ARIMA Forecast
                plt.figure(figsize=(10, 6))
                plt.plot(time_series, label="Original Time Series")
                plt.plot(forecast_df.index, forecast_df["Forecast"], label="ARIMA Forecast", color="red")
                plt.legend()
                plt.title("ARIMA Forecast")
                st.pyplot(plt)

        elif model_choice == "Prophet":
            st.write("### Prophet Forecast")
            with st.spinner("Generating forecast using Prophet..."):
                # Fit Prophet model
                prophet_model = Prophet()
                prophet_model.fit(prophet_data)

                # Create a DataFrame for future dates
                future = prophet_model.make_future_dataframe(periods=forecast_steps, freq='M')
                forecast = prophet_model.predict(future)

                # Display forecast
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                # Plot Prophet Forecast
                fig = prophet_model.plot(forecast)
                st.pyplot(fig)
