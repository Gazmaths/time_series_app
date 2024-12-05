**Time Series Analysis and Forecasting App**
This Streamlit application allows users to upload time series data, analyze its components (seasonality, trend, residual), and forecast future values using two popular models: ARIMA (via pmdarima) and Prophet. The app dynamically suggests the best ARIMA parameters and enables intuitive interaction with the data for better decision-making.

**Features**
Upload and Analyze Data: Upload a CSV file containing your time series data.
Decomposition: Visualize the trend, seasonality, and residuals of your time series.
ARIMA Model Suggestion:
Automatically selects the best ARIMA (p, d, q) and seasonal order using pmdarima.
Prophet Forecasting:
Use Prophet for accurate time series forecasting with confidence intervals.
Dynamic Plotting:
Interactive plots for both ARIMA and Prophet forecasts.
Custom Forecast Horizon:
Choose the number of future steps to forecast.
