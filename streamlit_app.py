import streamlit as st

st.title("Microsoft Stock Price Forecasting App")

st.write("If you can see this message, Streamlit is working correctly.")

# Try to import matplotlib and show status
try:
    import matplotlib.pyplot as plt
    st.success("matplotlib imported successfully!")
except ImportError as e:
    st.error(f"Failed to import matplotlib: {e}")

# Try to import other required libraries
try:
    import numpy as np
    st.success("numpy imported successfully!")
except ImportError as e:
    st.error(f"Failed to import numpy: {e}")

try:
    import joblib
    st.success("joblib imported successfully!")
except ImportError as e:
    st.error(f"Failed to import joblib: {e}")

try:
    import pandas as pd
    st.success("pandas imported successfully!")
except ImportError as e:
    st.error(f"Failed to import pandas: {e}")