import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd

# Black-Scholes-Merton model function
def black_scholes_merton(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Streamlit UI
st.title("Optimized Black-Scholes-Merton Option Pricing Heatmap")

# Add a button linking to the derivation page
st.markdown(
    """
    <a href="https://sites.google.com/view/rahulsunilkumar/the-black-scholes-pricing-model?authuser=0" target="_blank">
    <button style="background-color:#4CAF50; color:white; padding:10px 20px; text-align:center; border:none; border-radius:4px;">
    Learn about the derivation here
    </button></a>
    """, unsafe_allow_html=True
)

# User Inputs for the Option Parameters
current_price = st.sidebar.number_input('Current Asset Price (S)', min_value=50.0, max_value=150.0, value=100.0)
strike_price = st.sidebar.number_input('Strike Price (K)', min_value=50.0, max_value=150.0, value=100.0)
time_to_maturity = st.sidebar.number_input('Time to Maturity (T, in years)', min_value=0.1, max_value=5.0, value=1.0)
risk_free_rate = st.sidebar.number_input('Risk-free Interest Rate (r)', min_value=0.0, max_value=0.2, value=0.05)
volatility = st.sidebar.number_input('Volatility (σ)', min_value=0.1, max_value=1.0, value=0.2)

# Heatmap Parameters
min_spot_price = st.sidebar.number_input('Min Spot Price for Heatmap', min_value=50.0, max_value=150.0, value=60.0)
max_spot_price = st.sidebar.number_input('Max Spot Price for Heatmap', min_value=50.0, max_value=150.0, value=140.0)

min_volatility = st.sidebar.number_input('Min Volatility for Heatmap', min_value=0.1, max_value=1.0, value=0.1)
max_volatility = st.sidebar.number_input('Max Volatility for Heatmap', min_value=0.1, max_value=1.0, value=1.0)

# Display the summary table at the top
st.subheader("Input Parameters Summary")
summary_data = {
    'Parameter': ['Current Asset Price (S)', 'Strike Price (K)', 'Time to Maturity (T)', 'Volatility (σ)', 'Risk-free Rate (r)'],
    'Value': [current_price, strike_price, time_to_maturity, volatility, risk_free_rate]
}
summary_df = pd.DataFrame(summary_data)
st.table(summary_df)

# Create a grid of spot prices and volatilities
spot_prices = np.linspace(min_spot_price, max_spot_price, 20)  # 20 discrete levels for spot price
volatilities = np.linspace(min_vol
