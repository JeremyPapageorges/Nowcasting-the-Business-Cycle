import os
import sys
import pickle
import importlib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
import streamlit as st

import Data_Sourcing_v1
importlib.reload(Data_Sourcing_v1)
from Data_Sourcing_v1 import download_data

import Data_Sourcing_DBnomics_v1
importlib.reload(Data_Sourcing_DBnomics_v1)
from Data_Sourcing_DBnomics_v1 import dbnomics_download

import Data_Sourcing_Yfinance_v1
importlib.reload(Data_Sourcing_Yfinance_v1)
from Data_Sourcing_Yfinance_v1 import download_data_yfinance

kkr_purple = '#590e5b'
marker_color = '#FF69B4'

# set up the subfolder path for saving pickled data when updates are not necessary

subfolder_path = "Pickled_data"

# ensure the user can determine if the data should be updated or not

data_update = st.text_input("Should the program retrieve the latest available data (True for yes, False for no)?")

def data_storage(file_name,data):

    file_path = os.path.join(subfolder_path, file_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Pickle the data to the specified path
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

    print(f"Data has been pickled to {file_path}")

def data_extraction(file_name):
    # PICKLE THE DATA 
    file_path = os.path.join(subfolder_path, file_name)

    # Load (unpickle) the data
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    print("Loaded data:", data)
    return data


def download_data_yfinance(ticker, start_date, end_date):

    sp500_data = yf.download(ticker, start=start_date, end=end_date)
    closing_prices = sp500_data['Close']
    monthly_closing_prices = closing_prices.resample('M').mean()
    return closing_prices, monthly_closing_prices

def graph(series,title_1, label_1, y_label_1, title_2, label_2, y_label_2):

    fig, axs = plt.subplots(2,1, figsize = (10,6))

    kkr_purple = '#590e5b'  # Approximate hex color from the KKR logo
    marker_color = '#FF69B4'  # Another shade for the markers (pink)

    # First subplot: Full year-over-year percentage change
    axs[0].plot(series, color=kkr_purple, label= label_1, linewidth = 0.7)
    axs[0].set_title(title_1)
    axs[0].set_ylabel(y_label_1)
    axs[0].legend(loc="upper left")  # Add legend here with label
    axs[0].grid(True)
    # Second subplot: Last 12 months with mean line and markers
    axs[1].plot(series.iloc[-12:], color=kkr_purple, label= label_2, linewidth = 0.7)
    axs[1].scatter(series.iloc[-12:].index, series.iloc[-12:], color=marker_color, label="Monthly Data")
    axs[1].axhline(series.mean(), color='red', linestyle='--', linewidth=1, label="Mean")
    axs[1].set_title(title_2)
    axs[1].set_ylabel(y_label_2)
    axs[1].set_xlabel("Date")
    axs[1].legend(loc="upper left")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show

# download ISM data from dbnomics

file_name = "ISM_data.pkl"

if data_update == True:

    df1, df2 = dbnomics_download()
    ISM = pd.DataFrame()
    ISM['manufacturing'] = df1['value']
    ISM['non manufacturing'] =df2['value']
    ISM = ISM.set_index(df1['period'])

    ISM['Manufacturing delta neutral'] = (ISM['manufacturing'] - 50)/ 50
    ISM['Non manufacturing delta neutral'] = (ISM['non manufacturing'] - 50)/50

    #adjust the dates in the index to reflect eom
    ISM.index = ISM.index + pd.offsets.MonthEnd(1)
    new_row_date = ISM.index.max() + pd.DateOffset(months=1)
    ISM.loc[new_row_date] = [np.nan] * ISM.shape[1]

    data_storage(file_name, ISM)

ISM = data_extraction(file_name)



# set up start date and end date deployment
end_date = date.today() - timedelta(days = 1)
start_date = end_date - relativedelta(years = 50)
print(f'data begins at {start_date} and ends at date {end_date}')

#######################################################
# STOCK MARKET DATA
#######################################################

file_name = 'stock_market_data.pkl'

if data_update == True:

    # download SP500 data using yfinance
    stock_market = "^GSPC"
    (stock_market_data,
    stock_market_data_averages) = download_data_yfinance(stock_market, start_date, end_date)

    year_delta_stock_market = stock_market_data_averages.pct_change(12)
    year_delta_stock_market.index = year_delta_stock_market.index.tz_localize(None)

    data_storage(file_name, year_delta_stock_market)

year_delta_stock_market = data_extraction(file_name)

#######################################################
# YIELD SPREAD ANALYSIS USING T10Y3M PROVIDED BY THR FED DATABASE
#######################################################
file_name = 'yield_curve_data.pkl'

if data_update == True:

    yield_curve = ['T10Y3M','GS10','TB3MS']

    (yield_curve_data,
    yield_curve_data_averages) = download_data(yield_curve, start_date, end_date)

    data_storage(file_name, yield_curve_data_averages)

yield_curve_data_averages = data_extraction(file_name)

#######################################################
# Credit Spreads
#######################################################
file_name = 'credit_spread_data.pkl'

if data_update == True:

    credit_spreads = ['DBAA', 'DAAA','BAMLH0A0HYM2EY']
    (credit_spread_data,
    credit_spread_data_averages) = download_data(credit_spreads, start_date, end_date)

    # LOOKING AT CREDIT SPREAD DATA USING AAA-BAA SPREAD
    Baa_Aaa_spread = credit_spread_data['DBAA'] - credit_spread_data['DAAA']
    avg_Baa_Aaa_spread = Baa_Aaa_spread.resample('M').mean()
    year_delta_credit_spread = avg_Baa_Aaa_spread.pct_change(12) # we multiply by -1 since an increase in credit spread is negative
    year_delta_credit_spread.dropna(inplace = True)
    year_delta_credit_spread.name = 'Credit spread'

    # LOOKING AT CREDIT SPREAD DATA USING BOFA BOND SPREAD DATA
    HY_Aaa_spread = credit_spread_data['BAMLH0A0HYM2EY'] - credit_spread_data['DAAA']
    HY_Aaa_spread.dropna(inplace = True)
    avg_HY_spread = HY_Aaa_spread.resample('M').mean()
    year_delta_HY_spread = avg_HY_spread.pct_change(12)

    data_storage(file_name, year_delta_credit_spread)

year_delta_credit_spread = data_extraction(file_name)


#######################################################
# LABOUR MARKET
#######################################################
file_name = 'labour_market_data.pkl'

if data_update == True:

    labour_market = ['USPRIV','ICSA','UNEMPLOY', 'CE16OV', 'AWHI']

    (labour_market_data,
    labour_market_data_averages) = download_data(labour_market, start_date, end_date)

    # build the labour market index
    labour_market_data['employment_ratio'] = labour_market_data['CE16OV'] / labour_market_data['UNEMPLOY']
    labour_market_data_averages['employment_ratio'] = labour_market_data['employment_ratio'].resample('M').mean() 

    #Need to tadjust for the FRED's way of indexing time series data
    labour_market_data_averages[['USPRIV','UNEMPLOY','CE16OV','AWHI', 'employment_ratio']] = labour_market_data_averages[['USPRIV','UNEMPLOY','CE16OV','AWHI', 'employment_ratio']].shift(1)

    year_delta_labour_market = labour_market_data_averages.pct_change(12)
    year_delta_labour_market['ICSA'] = year_delta_labour_market['ICSA'] * -1 # need to invert the labour market index

    labour_market_index = year_delta_labour_market[['ICSA', 'employment_ratio', 'USPRIV', 'AWHI']].mean(axis = 1)
    labour_market_index.name = 'labour_market_index'

    data_storage(file_name, labour_market_index)


labour_market_index = data_extraction(file_name)

#######################################################
# CONSUMER SPENDING
#######################################################

# Lets build the consumer spendig index

file_name = 'consumer_spending.pkl'

if data_update == True:

    consumer_spending = ['PCEC96','RRSFS','UMCSENT', 'W875RX1', 'PCE', 'PCEPI', 'RSXFS']

    (consumer_spending_data,
    consumer_spending_data_averages) = download_data(consumer_spending, start_date, end_date)

    # shift down due to the way the FRED publishes data
    # add an additional row sinceall time series are lagged and need to be adjusted accordingly

    new_row_date = consumer_spending_data.index.max() + pd.DateOffset(months=1)
    consumer_spending_data.loc[new_row_date] = [np.nan] * consumer_spending_data.shape[1]
    consumer_spending_data[['PCEC96','RRSFS','UMCSENT', 'W875RX1', 'PCE', 'PCEPI', 'RSXFS']] =  consumer_spending_data[['PCEC96','RRSFS','UMCSENT', 'W875RX1', 'PCE', 'PCEPI', 'RSXFS']].shift(1)

    consumer_spending_data['personal_consumption_expenditure'] = consumer_spending_data['PCE'] / consumer_spending_data['PCEPI']
    year_delta_consumer_spending = consumer_spending_data.pct_change(12).shift(1)
    year_delta_consumer_spending['consumer_spending_index'] = year_delta_consumer_spending[['personal_consumption_expenditure', 'RSXFS', 'RRSFS']].mean(axis = 1)

    data_storage(file_name, year_delta_consumer_spending)

year_delta_consumer_spending = data_extraction(file_name)

#######################################################
#BUSINESS SPENDING
#######################################################
file_name = "business_spending_data"

if data_update == True:

    business_activity = ['INDPRO', 'CMRMTSPL']
    (business_activity_data,
    business_activity_data_averages) = download_data(business_activity, start_date, end_date)

    new_row_date = business_activity_data.index.max() + pd.DateOffset(months=1)
    business_activity_data.loc[new_row_date] = [np.nan] * business_activity_data.shape[1]
    business_activity_data = business_activity_data.shift()

    year_delta_business_activity = business_activity_data.pct_change(12)

    data_storage(file_name, year_delta_business_activity)

year_delta_business_activity = data_extraction(file_name)

#######################################################
#OIL
#######################################################
file_name = "oil_data"

if data_update == True:

    oil = ['DCOILWTICO']
    (oil_data,
    oil_data_averages) = download_data(oil, start_date, end_date)
    year_delta_oil = oil_data_averages.pct_change(12)
    data_storage(file_name, year_delta_oil)

year_delta_oil = data_extraction(file_name)

#######################################################
# MONEY SUPPLY
#######################################################
file_name = "money_supply_data"

if data_update == True:

    monetary_base = ['BOGMBASE','CPIAUCSL']
    (monetary_base_data,
    monetary_base_data_averages) = download_data(monetary_base, start_date, end_date)

    monetary_base_data_averages['real monetary base'] = monetary_base_data_averages['BOGMBASE'] / monetary_base_data_averages['CPIAUCSL']
    year_delta_monetary_base = monetary_base_data_averages.pct_change(12)

    data_storage(file_name, year_delta_monetary_base)

year_delta_monetary_base = data_extraction(file_name)

#######################################################
# HOUSING MARKET
#######################################################
file_name = "housing_market_data"

if data_update == True:

    housing = ['PERMIT']
    (housing_data,
    housing_data_averages) = download_data(housing, start_date, end_date)
    new_row_date = housing_data_averages.index.max() + pd.DateOffset(months=1)
    housing_data_averages.loc[new_row_date] = [np.nan] * housing_data_averages.shape[1]
    housing_data_averages = housing_data_averages.shift()

    year_delta_housing = housing_data_averages.pct_change(12)
    
    data_storage(file_name, year_delta_housing)

year_delta_housing = data_extraction(file_name)

# Combine all time series that were going to be using for the 

indicators = pd.concat([year_delta_stock_market, # Stock market YoY change , 421
                        yield_curve_data_averages['T10Y3M'][1:]/10, # Yield curve % spread (multiplied by 10), 422
                        year_delta_credit_spread * -1, # Credit Spread YoY change
                        labour_market_index, # Labour Market Index
                        year_delta_consumer_spending['consumer_spending_index'], # Personal COnsumption Expenditure, Advanced retail sales: real trade, dvance Real Retail and Food Services Sales
                        year_delta_consumer_spending['UMCSENT'], # Reuters, UoM  Consumer Sentiment 
                        year_delta_business_activity['INDPRO'],# Industrial production
                        year_delta_business_activity['CMRMTSPL'], # Real Manufacturing and Trade Industries Sales
                        ISM['Manufacturing delta neutral'], # ISM Manufacturing
                        ISM['Non manufacturing delta neutral'], # ISM non-manufacturing
                        year_delta_oil['DCOILWTICO'] * -1 , # Crude Oil, inverted (rise in oil leads to drop in confidence)
                        year_delta_housing['PERMIT'], # housing starts
                        ], axis = 1)

st.write("Latest recession risk signal data:")
st.write(indicators.tail(24))



