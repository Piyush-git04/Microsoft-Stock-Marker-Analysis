import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd

# STEP 1: Load the stock data
def load_data():
    df = pd.read_csv(r'C:\Users\DELL\Desktop\MSFT\MSFT_stock.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated()].sort_index()
    return df

df = load_data()
st.write('Index dtype:', df.index.dtype)

# STEP 2: Load the ARIMA model
def load_model():
    return joblib.load(r'C:\Users\DELL\Desktop\MSFT\arima_final_model.pkl')

model = load_model()

# STEP 3: User input for forecasting horizon
n_steps = st.number_input("How many future business days to forecast?:", min_value=1, max_value=365, value=30)

# STEP 4: Forecast and plot
if st.button('Forecast'):
    # Forecast future 'diff' values
    forecast_diff = model.forecast(steps=n_steps)

    # Convert to actual 'Close' values
    last_close = df['Close'].iloc[-1]
    forecast_close = last_close + pd.Series(forecast_diff).cumsum()

    # Generate future business dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps, freq='B')

    # Create DataFrame for forecast
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast_Close': forecast_close.values
    }).set_index('Date')
    forecast_df.index = pd.to_datetime(forecast_df.index)
    forecast_df = forecast_df[~forecast_df.index.duplicated()].sort_index()

    # Optional: display table
    st.subheader('Forecasted Close Prices')
    forecast_df_display = forecast_df.copy()
    forecast_df_display.index = forecast_df_display.index.strftime('%Y-%m-%d')
    st.dataframe(forecast_df_display)

    # Line chart
    st.line_chart(forecast_df)

    # STEP 5: Matplotlib plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Historical Close')
    ax.plot(forecast_df.index, forecast_df['Forecast_Close'], label='Forecast (Next Days)', linestyle='--', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('ARIMA Forecast: Next Business Days')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    # STEP 6: Download option
    csv = forecast_df.to_csv().encode('utf-8')
    st.download_button('Download Forecast CSV', data=csv, file_name='Forecast.csv', mime='text/csv')
