import pandas as pd
import datetime
from pandas.tseries.offsets import BDay
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st
from functools import lru_cache

# Load the CSV file with an alternate encoding
try:
    ticker_data = pd.read_csv('tickers.csv', encoding='ISO-8859-1')  # or encoding='latin1'
    st.write("Loaded CSV file successfully.")
    
    # Drop rows where the 'Name' column (or the stock name column) has missing values
    ticker_data.dropna(subset=['Name'], inplace=True)
    
except Exception as e:
    st.write(f"Error loading tickers: {e}")
    ticker_data = pd.DataFrame()

@lru_cache(maxsize=32)
def fetch_data_yahoo(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for the ticker symbol '{ticker}'.")
        return data
    except Exception as e:
        st.write(f"Error fetching data for ticker '{ticker}': {e}")
        return None
        
def preprocess_data(data):
    """Preprocess data by creating lag features."""
    data = data.copy()
    target = data['Close']
    
    features = pd.DataFrame()
    for i in range(1, 6):  # Create 5 lag features
        features[f'Lag_{i}'] = target.shift(i)

    features = features.dropna()
    target = target[5:]  # Align target with features

    return features, target

def train_model(X, y):
    """Train the Linear Regression model using cross-validation."""
    model = LinearRegression()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    mean_score = -scores.mean()  # Convert back to positive MSE
    st.write(f"Mean Cross-Validation MSE: {mean_score:.4f}")

    # Fit the model on the entire dataset after evaluation
    model.fit(X, y)
    return model

def predict_future(model, last_known_features, lookahead_days):
    """Predict future stock prices using the trained model."""
    predictions = []
    current_features = last_known_features.copy()

    for _ in range(lookahead_days):
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        
        new_features = current_features.values.flatten().tolist()[1:] + [pred]
        current_features = pd.DataFrame([new_features], columns=current_features.columns)
    
    return predictions

def calculate_profit_loss(predicted_prices, current_price):
    """Calculate profit or loss based on predictions and provide advice."""
    profit_loss = []
    trend = "Stable"

    # Determine trend based on the first and last predicted prices
    if predicted_prices[0] < predicted_prices[-1]:
        trend = "Uptrend"
    elif predicted_prices[0] > predicted_prices[-1]:
        trend = "Downtrend"

    # Assess profit or loss
    for price in predicted_prices:
        if price > current_price:
            profit_loss.append('Profit')
        elif price < current_price:
            profit_loss.append('Loss')
        else:
            profit_loss.append('No Change')

    # Provide more nuanced advice
    if trend == "Uptrend":
        advice = 'Strong Buy - Prices are expected to rise.'
    elif trend == "Downtrend":
        advice = 'Sell - Prices are expected to fall.'
    else:
        if all(p < current_price for p in predicted_prices):
            advice = 'Sell - All future prices are lower than the current price.'
        else:
            advice = 'Hold - Mixed signals; consider your investment strategy.'

    return profit_loss, advice

def stock_price_prediction(ticker, lookahead_days, lookback_days):
    """Predict stock prices based on historical data."""
    end_date = datetime.date.today().strftime('%Y-%m-%d')  # End date is today's date
    start_date = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime('%Y-%m-%d')  # Start date adjusted for lookback
    historical_data = fetch_data_yahoo(ticker, start_date, end_date)  # Use dynamic range

    if historical_data is None or historical_data.empty:
        st.write(f"No data available for {ticker}.")
        return None, None

    current_price = historical_data['Close'].iloc[-1]

    if len(historical_data) < 200:  # Check for sufficient data
        st.write(f"Insufficient data for {ticker}. Consider using a more liquid stock.")
        return None, None

    features, target = preprocess_data(historical_data)

    model = train_model(features, target)
    last_known_features = features.iloc[-1:]

    predictions = predict_future(model, last_known_features, lookahead_days)

    future_dates = pd.bdate_range(historical_data.index[-1] + BDay(1), periods=lookahead_days)

    profit_loss, advice = calculate_profit_loss(predictions, current_price)

    result = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': predictions,
        'Profit/Loss': profit_loss
    })

    return result, advice

def plot_results(results, historical_data, ticker):
    """Visualize historical and predicted stock prices using Plotly."""
    fig = go.Figure()

    # Plot historical prices
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='royalblue')
    ))

    # Plot predicted prices
    fig.add_trace(go.Scatter(
        x=results['Date'],
        y=results['Predicted Price'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='orange', dash='dash')
    ))

    # Automatically adjust y-axis to zoom in on the range of prices
    min_price = min(historical_data['Close'].min(), results['Predicted Price'].min())
    max_price = max(historical_data['Close'].max(), results['Predicted Price'].max())
    price_range_padding = (max_price - min_price) * 0.05  # 5% padding on y-axis

    # Update layout for interactivity and auto-adjust y-axis
    fig.update_layout(
        title=f'Stock Price Prediction for {ticker}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(x=0, y=1),
        hovermode='x unified',
        dragmode='zoom',
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis=dict(range=[min_price - price_range_padding, max_price + price_range_padding]),
        template='plotly_white'
    )

    # Show Plotly figure in Streamlit
    st.plotly_chart(fig)

# Streamlit widgets for interactive stock prediction
def interactive_stock_prediction():
    """Interactive widget with stock ticker search functionality."""
    tickers = ticker_data['Ticker'].tolist() if 'Ticker' in ticker_data.columns else []

    ticker = st.selectbox("Choose a stock ticker:", tickers)  # Replace combobox with selectbox

    lookahead_days = st.slider("Lookahead Days:", min_value=1, max_value=100, value=10)
    lookback_days = st.slider("Lookback Days:", min_value=1, max_value=365, value=200)

    run_button = st.button("Run Prediction")

    if run_button:
        results, advice = stock_price_prediction(ticker, lookahead_days, lookback_days)
        if results is not None:
            st.write("### Predicted Stock Prices:")
            st.write(results)
            st.write("### Investment Advice:")
            st.write(advice)
            plot_results(results, fetch_data_yahoo(ticker, '2000-01-01', datetime.date.today().strftime('%Y-%m-%d')), ticker)

# Run the Streamlit app
interactive_stock_prediction()
