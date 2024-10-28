from fredapi import Fred
import pandas as pd
def download_data(tickers,start,end):
    key = '3d5c0f448ca8947e438c3bd88905e4d2'
    fred = Fred(api_key = key)
    data_dict = {}
    frequency_dict = {}

    for tick in tickers:
        try:
            data = fred.get_series(tick, observation_start = start, observation_end = end)
            data_dict[tick] = pd.DataFrame(data, columns = [tick])

            # we also download fred metadata to get the frequency
            series_info = fred.get_series_info(tick)
            frequency_dict[tick] = series_info.get('frequency', 'Unknown')

        except Exception as e:
            print(f"Error fetching data for {tick}: {e}")
    combined_data = pd.concat(data_dict.values(), axis = 1)
    combined_data = combined_data.rename_axis("Date", axis="index")

    # display the frequency information:
    print("FREQUENCY OF EACH TICKER:")
    for ticker, freq in frequency_dict.items():
        print(f"{tick} : {freq}")
    
    # create monthly averaged data:
    monthly_averages = combined_data.resample('M').mean()
    pct_change_monthly_averages = monthly_averages.pct_change(12)

    return combined_data, monthly_averages, pct_change_monthly_averages