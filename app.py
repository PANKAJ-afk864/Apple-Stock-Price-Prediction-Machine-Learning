import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pickle import dump, load
import os

# Page title
st.title("Apple Stock Price Forecasting using SARIMA")

# Upload CSV
uploaded_file = st.file_uploader("Upload your stock CSV file with 'Adj Close' column", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)

    st.write("### Data Preview")
    st.write(df.tail())

    # Forecast horizon slider
    forecast_days = st.slider("Select number of days to forecast", min_value=7, max_value=60, value=30)

    # Train the model
    st.write("### Training SARIMA Model...")
    model3 = SARIMAX(df['Adj Close'], order=(1,1,1), seasonal_order=(1,1,1,30))
    model3_fit = model3.fit(disp=False)

    # Forecast next N days
    forecast = model3_fit.forecast(forecast_days)

    st.write(f"### Forecasted Prices (Next {forecast_days} Days)")
    st.write(forecast)

    # Plot the data
    st.write("### Forecast Plot")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['Adj Close'], label='Adj Close', color='purple')
    ax.plot(pd.date_range(df.index[-1], periods=forecast_days+1, freq='D')[1:], forecast, label='Forecasted', color='red')
    ax.legend(loc='upper left')
    plt.title(f'AAPL Stock Price Forecast - Next {forecast_days} Days')
    st.pyplot(fig)

    # Save the model
    if st.button("Save Model to Pickle"):
        dump(model3_fit, open('pickle_file_Apple.sav', 'wb'))
        st.success("Model saved successfully as pickle_file_Apple.sav")

    # Load the model and show prediction
    if os.path.exists('pickle_file_Apple.sav'):
        if st.button("Load Pickle Model and Forecast Again"):
            loaded_model = load(open('pickle_file_Apple.sav', 'rb'))
            fct = loaded_model.forecast(forecast_days)
            st.write(f"Forecast from Pickle Model ({forecast_days} Days):")
            st.write(fct)
