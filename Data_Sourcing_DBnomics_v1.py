from dbnomics import fetch_series, fetch_series_by_api_link
import pandas as pd

def dbnomics_download():
    df1 = fetch_series('ISM/pmi/pm')
    df2 = fetch_series('ISM/nm-pmi/pm')
    # Display the first few row
    print(df1.head())
    return df1, df2

    