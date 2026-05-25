from typing import Tuple, List
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyzer = SentimentIntensityAnalyzer()


def get_news_sentiment(ticker: str, max_items: int = 10) -> Tuple[float, List[str]]:
    headlines = []

    try:
        news = yf.Ticker(ticker).news or []
    except Exception:
        return 0.0, headlines

    scores = []
    for item in news[:max_items]:
        title = item.get("title", "")
        if title:
            headlines.append(title)
            scores.append(analyzer.polarity_scores(title)["compound"])

    if not scores:
        return 0.0, headlines

    return float(sum(scores) / len(scores)), headlines
