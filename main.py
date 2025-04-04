from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import pickle

app = Flask(__name__)

# Load pre-trained model (Ensure you have 'crypto_model.pkl')
try:
    with open("crypto_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None  # Handle case where model is missing


# Fetch real-time crypto price
def get_crypto_price(symbol="BTC-USD"):
    crypto = yf.Ticker(symbol)
    data = crypto.history(period="1d")
    if data.empty:
        return None
    return round(data["Close"].iloc[-1], 2)  # Get latest closing price


@app.route("/")
def home():
    return "Crypto Price Predictor API is Running!"


@app.route("/price", methods=["GET"])
def get_price():
    symbol = request.args.get("symbol", "BTC-USD")  # Default to Bitcoin
    price = get_crypto_price(symbol)
    if price:
        return jsonify({"symbol": symbol, "current_price": price})
    return jsonify({"error": "Could not fetch price"}), 400


@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol", "BTC-USD")
    price = get_crypto_price(symbol)

    if price is None:
        return jsonify({"error": "Could not fetch crypto price"}), 400

    if model:
        future_price = model.predict(np.array(price).reshape(-1, 1))[0]
        return jsonify({"symbol": symbol, "current_price": price, "predicted_price": round(future_price, 2)})
    else:
        return jsonify({"symbol": symbol, "current_price": price, "predicted_price": "Model not available"}), 200


if __name__ == "__main__":
    app.run(debug=True)
