import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib 
import pandas as pd

# Importing the data-'df' for last 'close' price and last 'date'
def load_data():
    df= pd.read_csv(r'\MSFT\df.csv', parse_dates= ['Date'], dayfirst= False) 
    df.set_index('Date', inplace= True)
    return df

df= load_data()
st.write('Index dtype:', df.index.dtype)


# Load ARIMA model
def load_model():
    return joblib.load(r'\MSFT\arima_final_model.pkl')

model= load_model()


# Input: Number of days to forecast
n_steps= st.number_input("How many future business days to forecast 'Microsoft Stock price' from 18-06-25?:", min_value= 1, max_value= 365, value= 30)

# Forecast when butto is clicked
if st.button('Forecast'):
    forecast_diff= model.forecast(steps= n_steps)

    # Recover last known close
    last_close= df['Close'].iloc[-1]
    forecast_close= last_close + forecast_diff.cumsum()

    # Create future date index
    last_date = pd.to_datetime(df.index[-1])
    future_date= pd.date_range(start= last_date + pd.Timedelta(days= 1), periods= n_steps, freq= 'B')

    forecast_df= pd.DataFrame({
        'Date': future_date,
        'Forecast_Close': forecast_close.values}).set_index('Date')


    # Format date format (no time)
    forecast_df_display= forecast_df.copy()
    forecast_df_display.index= forecast_df_display.index.strftime('%d-%m-%Y')
    
    # Show tables
    st.subheader('Forecasted Close Prices')
    st.dataframe(forecast_df_display)

    st.line_chart(forecast_df)

    # 6. Plot
    # ensure datetime index for plotting
    df.index= pd.to_datetime(df.index, dayfirst= True).normalize()
    forecast_df.index= pd.to_datetime(forecast_df.index).normalize()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'].values, label='Historical Close')
    plt.plot(forecast_df.index, forecast_df['Forecast_Close'].values, label='Forecast (Next Days)', linestyle='--', color= 'red')

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('ARIMA Forecast: Next Business Days')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Optional: Save forecast as CSV
    csv= forecast_df.to_csv().encode('utf-8')
    st.download_button('Download Forecast CSV', data= csv, file_name= 'Forecast.csv', mime= 'text/csv')
