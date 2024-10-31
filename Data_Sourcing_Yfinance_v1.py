import pandas as pd


def download_data_yfinance(ticker, start_date, end_date):

    sp500_data = yf.download(ticker, start=start_date, end=end_date)
    closing_prices = sp500_data['Close']
    monthly_closing_prices = closing_prices.resample('M').mean()

    return closing_prices, monthly_closing_prices


