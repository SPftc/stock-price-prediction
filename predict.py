import json
from pathlib import Path

import joblib

from data_loader import download_stock_data
from features import add_technical_features, encode_ticker
from sentiment import get_news_sentiment


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "saved_models"

xgb_model = joblib.load(MODEL_DIR / "xgb_model.pkl")

with open(MODEL_DIR / "feature_columns.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)


def predict_ticker(ticker: str):
    ticker = ticker.upper().strip()

    df = download_stock_data(ticker, start="2020-01-01")
    df["Ticker"] = ticker
    df = add_technical_features(df)
    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Not enough data to build features for {ticker}")

    latest_row = df.iloc[-1:].copy()
    latest_row = encode_ticker(latest_row)

    X = latest_row.reindex(columns=FEATURE_COLUMNS, fill_value=0).astype(float)

    proba_up = float(xgb_model.predict_proba(X)[0, 1])
    proba_down = 1.0 - proba_up
    prediction = int(proba_up >= 0.5)

    sentiment_score, headlines = get_news_sentiment(ticker)

    sentiment_bullish = (sentiment_score + 1) / 2
    combined_score = 0.85 * proba_up + 0.15 * sentiment_bullish

    latest_date = str(df.iloc[-1]["Date"]) if "Date" in df.columns else None

    return {
        "ticker": ticker,
        "prediction": prediction,
        "proba_up": proba_up,
        "proba_down": proba_down,
        "sentiment_score": sentiment_score,
        "combined_score": combined_score,
        "latest_date": latest_date,
        "headlines": headlines[:5],
    }
