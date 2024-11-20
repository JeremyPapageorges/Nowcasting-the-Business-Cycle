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

data_update = True


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
    new_row_date = ISM.index.max() + pd.offsets.MonthEnd(1)
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
yield_curve_data_averages.rename(columns = {'T10Y3M' : '10Y - 3M yield spread'})

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


    # Force user to provide latest UMCSENT data reading
    umcsent_input = st.number_input("Provide latest UMCSENT available reading:", step = 0.1)

    # Define a flag to control the script execution
    input_valid = False

    # Check for input validity
    while not input_valid:
        if umcsent_input:
            input_valid = True
        else:
            st.warning("Please provide a value to proceed.")
            st.stop()  # Halts Streamlit execution temporarily
        
    consumer_spending_data['UMCSENT'][-1] = umcsent_input
    st.write('Consumer spending data')
    st.dataframe(consumer_spending_data)


    # shift down due to the way the FRED publishes data
    # add an additional row since all time series are lagged and need to be adjusted accordingly

    new_row_date = consumer_spending_data.index.max() + pd.offsets.MonthEnd(1)
    consumer_spending_data.loc[new_row_date] = [np.nan] * consumer_spending_data.shape[1]

    consumer_spending_data[['PCEC96','RRSFS','UMCSENT', 'W875RX1', 'PCE', 'PCEPI', 'RSXFS']] =  consumer_spending_data[['PCEC96','RRSFS','UMCSENT', 'W875RX1', 'PCE', 'PCEPI', 'RSXFS']].shift(1)

    consumer_spending_data['personal_consumption_expenditure'] = consumer_spending_data['PCE'] / consumer_spending_data['PCEPI']
    # forward fill the PCE index as to ensure the cosumer spendign index reflects an appropriate mean value
    consumer_spending_data['personal_consumption_expenditure'].fillna(method='ffill', inplace=True)

    year_delta_consumer_spending = consumer_spending_data.pct_change(12)
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

    new_row_date = business_activity_data.index.max() + pd.offsets.MonthEnd(1)
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
    new_row_date = housing_data_averages.index.max() + pd.offsets.MonthEnd(1)
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


indicators = indicators.rename(
    columns = {
       '^GSPC' : 'YoY S&P 500 % change',
       'T10Y3M' : '10Y-3M yield spread',
       'Credit Spread' : 'YoY BAA-AAA spread % change (inverted)',
       'labour_market_index' : 'YoY labour market index % change',
       'consumer_spending_index' : 'YoY consumer market index % change',
       'UMCSENT' : 'YoY UMich Sentiment Survey % change',
       'INDPRO' : 'YoY Industrial production % change',
       'CMRMTSPL' : 'YoY real manufacturing & trade industry sales % change',
       'Manufacturing delta neutral' : 'ISM Manufacturing less neutral reading % above/below',
       'Non manufacturing delta neutral' : 'ISM Non-manufacturing less neutral reading % above/below',
       'DCOILWTICO' : 'YoY WTI oil price % change (inverted)',
       'PERMIT' : 'YoY Housing start permits % change'

    }
)
st.write("Latest recession risk signal data:")
st.write(indicators.tail(24))


EMI_base = indicators.median(axis = 1)
EMI_base.name = 'EMI'
EMI = EMI_base.rolling(3).mean()

rising_indicators = indicators> 0
ETI_base = rising_indicators.mean(axis = 1)
ETI_base.name = 'ETI'
ETI = ETI_base.rolling(3).mean()


probit_data = pd.concat([EMI, ETI], axis = 1).dropna()
probit_data['recession'] = ((ETI < 0.50) & (EMI<0)).astype(int) # returns 1 if recesssion, returns 0 otherwise


# fit the probit model
X = probit_data['ETI']
X_2 = probit_data['EMI']
X_3 = probit_data[['ETI', 'EMI']]
X = sm.add_constant(X)
X_2 = sm.add_constant(X_2)
X_3 = sm.add_constant(X_3)
y = probit_data['recession']


probit_model_1 = sm.Probit(y, X)
probit_model_2 = sm.Probit(y, X_2)
probit_model_3 = sm.Probit(y, X_3)
probit_results = probit_model_3.fit()

# Alt formulations of the probit models
probit_results_1 = probit_model_1.fit()
probit_results_2 = probit_model_2.fit()


print(probit_results.summary())


probit_data['recession_probability'] = probit_results.predict(X_3)

probit_data['recession_probability_alt_1'] = probit_results_1.predict(X)
probit_data['recession_probability_alt_2'] = probit_results_2.predict(X_2)

fig, axs = plt.subplots(3,1, figsize = (10,12))

# First subplot: Full year-over-year percentage change
axs[0].plot(EMI, color=kkr_purple, label="Economic Tredn Index", linewidth = 0.7)
axs[0].axhline(0.0, color = 'red', linestyle = '--', linewidth = 0.7)   
axs[0].scatter(EMI.index[-1], EMI[-1], color =  'red', label = "Latest EMI reading", marker = 'D')
axs[0].set_title("Economic Momentum Index (Median Monthly % change for 12 indicators)")
axs[0].set_ylabel("%")
axs[0].legend(loc="upper left")  # Add legend here with label
axs[0].grid(True)

axs[1].plot(ETI, color=kkr_purple, label="Economic Trend Index", linewidth = 0.7)
axs[1].axhline(0.5, color = 'red', linestyle = '--', linewidth = 0.7)
axs[1].scatter(ETI.index[-1], ETI[-1], color =  'red', label = "Latest ETI reading", marker = 'D')
axs[1].set_title("Economic Trend Index (Monthly diffusion index for 12 indicators)")
axs[1].set_ylabel("%")
axs[1].legend(loc="upper left")  # Add legend here with label
axs[1].grid(True)

axs[2].plot(probit_data['recession_probability'], color=kkr_purple, label="Probit recession risk probability", linewidth = 0.5)
axs[2].set_title("Recession risk probability implpied by the ETI and EMI indicators, based on estimates using a Probit model")
axs[2].scatter(probit_data['recession_probability'].index[-1], probit_data['recession_probability'][-1], color =  'red', label = "Latest probability reading", marker = 'D')
axs[2].set_ylabel("Probability %")
axs[2].legend(loc="upper left")  # Add legend here with label
axs[2].grid(True)

plt.show()

#display = st.selectbox("Display options", ('Full time series', 'Last Three Months', 'Last three months rolling 3-month average'))

# Define colors
kkr_purple = "#4B0082"  # Replace with the exact color you want

# EMI Plot
st.write("### Economic Momentum Index (Median Monthly % change for 12 indicators)")
st.line_chart(EMI, x_label = 'Date', y_label = 'EMI (12-month pct change)', color = kkr_purple)  # Simple line chart for EMI
st.markdown(f"Latest EMI reading: **{EMI[-1]:.2f}%** at {date.today().strftime('%Y-%m-%d')}")

# Add horizontal line (as text for simplicity) and legend
st.markdown("<hr style='border-top: 1px dashed red;'>", unsafe_allow_html=True)

# ETI Plot
st.write("### Economic Trend Index (Monthly diffusion index for 12 indicators)")
st.line_chart(ETI,x_label = 'Date', y_label = 'ETI (Monthly diffusion)', color = kkr_purple)  # Simple line chart for ETI
st.markdown(f"Latest ETI reading: **{ETI[-1]:.2f}%** at {date.today().strftime('%Y-%m-%d')}")
st.markdown("<hr style='border-top: 1px dashed red;'>", unsafe_allow_html=True)

# Recession Probability Plot
st.write("### Recession Risk Probability implied by ETI and EMI indicators")
st.line_chart(probit_data['recession_probability'],x_label = 'Date', y_label = 'Recession risk probability', color = kkr_purple)
st.markdown(f"Latest probability reading: **{probit_data['recession_probability'][-1]:.2f}%** at {date.today().strftime('%Y-%m-%d')}")


# Show Last Twelve Months evolution of the time series
# Add horizontal line (as text for simplicity) and legend
st.markdown("<hr style='border-top: 1px dashed red;'>", unsafe_allow_html=True)
st.write("### Last Twelve Months ETI & EMI readings w/ Probit model output")


fig, axs = plt.subplots(3,1, figsize = (10,12))

axs[0].plot(EMI.iloc[-12:], color=kkr_purple, label="LTM Economic Momentum Index (rolling 3-month average)", linewidth = 1.2)
axs[0].plot(EMI_base.iloc[-12:], color='#FF69B4', label="LTM Economic Momentum Index", linewidth = 0.7)
axs[0].axhline(0.0, color = 'red', linestyle = '--', linewidth = 0.7)
axs[0].scatter(EMI.iloc[-12:].index, EMI.iloc[-12:], color='#590e5b', marker = 'D')
axs[0].scatter(EMI_base.iloc[-12:].index, EMI_base.iloc[-12:], color='#FF69B4', marker = 'o', s = 10)
axs[0].scatter(EMI.index[-1], EMI[-1], color =  'red', label = "Latest EMI reading", marker = 'D')
axs[0].set_title("LTM Economic Momentum Index (Median Monthly % change for 12 indicators)")
axs[0].set_ylabel("%")
axs[0].legend(loc="upper left")  # Add legend here with label
axs[0].grid(True)

axs[1].plot(ETI.iloc[-12:], color=kkr_purple, label="LTM Economic Trend Index (rolling 3-month average)", linewidth = 1.2)
axs[1].plot(ETI_base.iloc[-12:], color='#FF69B4', label="LTM Economic Trend Index", linewidth = 0.7)
axs[1].axhline(0.5, color = 'red', linestyle = '--', linewidth = 0.7)
axs[1].scatter(ETI.iloc[-12:].index, ETI.iloc[-12:], color='#590e5b', marker = 'D')
axs[1].scatter(ETI_base.iloc[-12:].index, ETI_base.iloc[-12:], color='#FF69B4', marker = 'o', s = 10)
axs[1].scatter(ETI.index[-1], ETI[-1], color =  'red', label = "Latest ETI reading", marker = 'D')
axs[1].set_title("LTM Economic Trend Index (Monthly diffusion index for 12 indicators)")
axs[1].set_ylabel("%")
axs[1].legend(loc="lower left")  # Add legend here with label
axs[1].grid(True)

axs[2].plot(probit_data['recession_probability'].iloc[-12:], color=kkr_purple, label="LTM Probit recession risk probability", linewidth = 0.5)
#axs[2].plot(probit_data['recession_probability_alt_1'].iloc[-12:], color=kkr_purple, label="LTM Probit recession risk probability (ETI only)", linewidth = 0.5)
#axs[2].plot(probit_data['recession_probability_alt_2'].iloc[-12:], color=kkr_purple, label="LTM Probit recession risk probability(EMI only)", linewidth = 0.5)
axs[2].set_title("LTM Recession risk probability implpied by the ETI and EMI indicators, based on estimates using a Probit model")
axs[2].scatter(probit_data['recession_probability'].iloc[-12:].index, probit_data['recession_probability'].iloc[-12:], color='#FF69B4', label="Monthly Readings")
axs[2].scatter(probit_data['recession_probability'].index[-1], probit_data['recession_probability'][-1], color =  'red', label = "Latest probability reading", marker = 'D')
axs[2].set_ylabel("Probability %")
axs[2].legend(loc="upper left")  # Add legend here with label
axs[2].grid(True)

plt.show()
st.pyplot(fig)