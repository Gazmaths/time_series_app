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

if uploaded_file:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Display the available columns in the CSV file
    st.write("### Available Columns in the Uploaded CSV")
    st.write(data.columns)

    # Allow users to select the date column and value column dynamically
    date_column = st.sidebar.selectbox("Select Date Column", data.columns)
    value_column = st.sidebar.selectbox("Select Time Series Column", data.columns)

    # Ensure the selected columns exist in the data
    if date_column not in data.columns:
        st.error(f"Column '{date_column}' not found in the uploaded file.")
    elif value_column not in data.columns:
        st.error(f"Column '{value_column}' not found in the uploaded file.")
    else:
        # Parse the selected date column as datetime and set it as index
        data[date_column] = pd.to_datetime(data[date_column])
        data.set_index(date_column, inplace=True)
        time_series = data[value_column]

        # Show uploaded data preview
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

        # ARIMA Model
        st.write("### ARIMA Forecast")
        arima_order = st.sidebar.text_input("Enter ARIMA order (p,d,q) as a tuple", "(1,1,1)")

        try:
            arima_order = eval(arima_order)  # Convert string input to tuple (p,d,q)
            if isinstance(arima_order, tuple) and len(arima_order) == 3:
                with st.spinner("Fitting ARIMA model..."):
                    arima_model = ARIMA(time_series, order=arima_order)
                    arima_fit = arima_model.fit()

                    # Forecasting
                    forecast_steps = st.sidebar.number_input("Number of steps to forecast", min_value=1, value=12)
                    forecast = arima_fit.forecast(steps=forecast_steps)
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

            else:
                st.error("Invalid ARIMA order. Please enter a tuple of the form (p,d,q).")

        except Exception as e:
            st.error(f"Error fitting ARIMA model: {str(e)}")

        # Prophet Forecasting
        st.write("### Prophet Forecast")
        # Prepare data for Prophet
        prophet_data = data.reset_index()
        prophet_data.columns = ['ds', 'y']  # Prophet expects columns named 'ds' (datetime) and 'y' (values)

        forecast_steps = st.sidebar.number_input("Number of steps to forecast (Prophet)", min_value=1, value=12)
        model_choice = st.sidebar.radio("Choose a Model for Forecasting", ["Prophet"])

        if st.sidebar.button("Generate Prophet Forecast"):
            if model_choice == "Prophet":
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
