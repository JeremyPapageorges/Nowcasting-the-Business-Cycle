from fredapi import Fred
import pandas as pd
def download_data(tickers,start,end):
    key = '3d5c0f448ca8947e438c3bd88905e4d2'
    fred = Fred(api_key = key)
    data_dict = {}
    monthly_avg_dict = {}
    frequency_dict = {}

    for tick in tickers:
        try:
            data = fred.get_series(tick, observation_start = start, observation_end = end)
            data_dict[tick] = pd.DataFrame(data, columns = [tick])

            # we also download fred metadata to get the frequency
            series_info = fred.get_series_info(tick)
            frequency_dict[tick] = series_info.get('frequency', 'Unknown')

            #show the frequency of the time series downloaded:
            print("FREQUENCY OF TICKER:")
            print(f'{tick} : {frequency_dict[tick]}')

        except Exception as e:
            print(f"Error fetching data for {tick}: {e}")

        # Apply resampling where necesaary depending to 
        # Handle daily frequency
        if frequency_dict[tick] == 'Daily, Close' or frequency_dict[tick] == 'Daily' :
            monthly_avg_dict[tick] = data_dict[tick].resample('M').mean()
        # Handle weekly frequency
        if frequency_dict[tick] == 'Weekly, Ending Saturday':
            monthly_avg_dict[tick] = data_dict[tick].resample('M').mean()
        # Handling monthly frequency
        if frequency_dict[tick] == 'Monthly':
            monthly_avg_dict[tick] = data_dict[tick]
            #adjust the dates in the index to reflect eom
            monthly_avg_dict[tick].index = monthly_avg_dict[tick].index - pd.offsets.MonthEnd(1)

    combined_data = pd.concat(data_dict.values(), axis = 1)
    combined_data = combined_data.rename_axis("Date", axis="index")
    combined_monthly_data = pd.concat(monthly_avg_dict.values(), axis = 1)
    combined_monthly_data = combined_monthly_data.rename_axis("Date", axis = 'index')

    return combined_data, combined_monthly_data