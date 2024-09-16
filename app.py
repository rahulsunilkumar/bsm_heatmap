import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm


# Black-Scholes-Merton model function
def black_scholes_merton(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


# Streamlit UI
st.title("Black-Scholes-Merton Option Pricing Heatmap")

# User Inputs
S = st.sidebar.slider('Spot Price (S)', min_value=50, max_value=150, value=100)
K = st.sidebar.slider('Strike Price (K)', min_value=50, max_value=150, value=100)
T = st.sidebar.slider('Time to Maturity (T, in years)', min_value=0.1, max_value=5.0, value=1.0)
r = st.sidebar.slider('Risk-free Rate (r)', min_value=0.0, max_value=0.2, value=0.05)
sigma = st.sidebar.slider('Volatility (σ)', min_value=0.1, max_value=1.0, value=0.2)

# Create a grid of spot prices and strike prices
S_values = np.linspace(50, 150, 100)
K_values = np.linspace(50, 150, 100)

# Calculate call and put option prices
call_prices = np.array([[black_scholes_merton(S, K, T, r, sigma, 'call') for K in K_values] for S in S_values])
put_prices = np.array([[black_scholes_merton(S, K, T, r, sigma, 'put') for K in K_values] for S in S_values])

# Plot heatmaps
st.subheader("Call Option Price Heatmap")
fig, ax = plt.subplots()
sns.heatmap(call_prices, xticklabels=np.round(K_values, 2), yticklabels=np.round(S_values, 2), ax=ax)
ax.set_xlabel('Strike Price (K)')
ax.set_ylabel('Spot Price (S)')
st.pyplot(fig)

st.subheader("Put Option Price Heatmap")
fig, ax = plt.subplots()
sns.heatmap(put_prices, xticklabels=np.round(K_values, 2), yticklabels=np.round(S_values, 2), ax=ax)
ax.set_xlabel('Strike Price (K)')
ax.set_ylabel('Spot Price (S)')
st.pyplot(fig)
