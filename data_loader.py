from datetime import datetime
import pandas as pd
import yfinance as yf


def download_stock_data(ticker: str, start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    df = df.reset_index()
    df.columns = [c.replace(" ", "_") for c in df.columns]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    return df
