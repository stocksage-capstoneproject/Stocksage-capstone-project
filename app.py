import pandas as pd
import datetime
from pandas.tseries.offsets import BDay
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import streamlit as st
from functools import lru_cache

# Load the CSV file with an alternate encoding
try:
    ticker_data = pd.read_csv('tickers.csv', encoding='ISO-8859-1')  # or encoding='latin1'
    print("Loaded CSV file successfully.")
    
    # Drop rows where the 'Name' column (or the stock name column) has missing values
    ticker_data.dropna(subset=['Name'], inplace=True)
    
except Exception as e:
    print(f"Error loading tickers: {e}")
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
        print(f"Error fetching data for ticker '{ticker}': {e}")
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
    print(f"Mean Cross-Validation MSE: {mean_score:.4f}")

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

    # Ensure predicted_prices is a list of values for comparison
    if isinstance(predicted_prices, pd.Series):
        predicted_prices = predicted_prices.tolist()

    # Make sure current_price is a scalar value (not a series)
    if isinstance(current_price, pd.Series):
        current_price = current_price.iloc[0]

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

def stock_price_prediction(ticker, lookahead_days):
    """Predict stock prices based on historical data."""
    end_date = datetime.date.today().strftime('%Y-%m-%d')  # End date is today's date
    historical_data = fetch_data_yahoo(ticker, '2000-01-01', end_date)

    if historical_data is None or historical_data.empty:
        print(f"No data available for {ticker}.")
        return None, None

    print(f"Fetched data for {ticker}: {historical_data.shape[0]} rows.")

    current_price = historical_data['Close'].iloc[-1]

    # Adjust the data threshold using the shape attribute
    if historical_data.shape[0] < 100:
        print(f"Insufficient data for {ticker}. Consider using a more liquid stock.")
        return None, None

    features, target = preprocess_data(historical_data)

    model = train_model(features, target)
    last_known_features = features.iloc[-1:]

    predictions = predict_future(model, last_known_features, lookahead_days)

    future_dates = pd.bdate_range(historical_data.index[-1] + BDay(1), periods=lookahead_days)

    profit_loss, advice = calculate_profit_loss(predictions, current_price)

    # Create the results DataFrame with a column for predicted prices
    result = pd.DataFrame({
        'Date': future_dates,
        'Predicted': predictions,  # Ensure the predictions are in the 'Predicted' column
        'Profit/Loss': profit_loss
    })

    return result, advice, historical_data

def plot_results_matplotlib(results, historical_data, lookahead_days):
    """Plot the predicted and historical stock prices using Matplotlib with zooming capabilities."""
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot historical data (lookback period)
    ax.plot(historical_data.index[-lookahead_days-365:], historical_data['Close'][-lookahead_days-365:], label='Historical Data (Lookback)', color='blue')

    # Plot predicted data (lookahead period)
    ax.plot(results['Date'], results['Predicted'], label='Predicted Data (Lookahead)', color='red', linestyle='--')

    # Add labels and title
    ax.set_title('Historical and Predicted Stock Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    # Enabling zoom and pan by default
    ax.set_autoscale_on(True)
    fig.canvas.draw()

    # Display the plot in Streamlit
    st.pyplot(fig)

def interactive_stock_prediction():
    """Interactive widget with search functionality."""
    stock_names = ticker_data['Name'].tolist() if 'Name' in ticker_data.columns else []

    stock_ticker = st.text_input('Enter Stock Ticker:', '')
    lookback_days = st.slider('Lookback Days', 1, 365, 365)
    lookahead_days = st.slider('Lookahead Days', 1, 100, 10)

    if st.button('Run Prediction'):
        if stock_ticker.strip() == '':
            st.error('Please enter a stock ticker.')
            return

        # Find the corresponding ticker based on the stock ticker
        ticker_row = ticker_data[ticker_data['Ticker'].str.upper() == stock_ticker.upper()]

        if ticker_row.empty:
            st.error(f"No stock found for '{stock_ticker}'.")
            return

        ticker_symbol = ticker_row.iloc[0]['Ticker']

        results, advice, historical_data = stock_price_prediction(ticker_symbol, lookahead_days)
        if results is not None:
            st.write("### Predicted Stock Prices:")
            st.write(results)
            st.write("### Investment Advice:")
            st.write(advice)

            try:
                plot_results_matplotlib(results, historical_data, lookahead_days)
            except Exception as e:
                st.error(f"Error plotting the graph: {e}")

# Run the interactive prediction tool in Streamlit
interactive_stock_prediction()
