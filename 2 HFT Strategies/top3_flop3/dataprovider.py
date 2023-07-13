import pandas as pd
import numpy as np

class CryptoMinuteDataProvider:
    '''
    This class provides functionality for loading and processing minute-level
    cryptocurrency data from multiple Parquet files. It fills missing data points
    with forward-filled values and keeps only the relevant columns.

    Attributes:
    ----------
    dataframes : dict
        A dictionary containing pandas DataFrames, where each key-value pair represents
        a cryptocurrency symbol and its corresponding data in the form of a DataFrame.

    Methods:
    -------
    get_full_raw_data():
        Returns the processed minute-level cryptocurrency data as a pandas DataFrame,
        combining data from all the provided Parquet files. Loads the raw data and
        processes it if it hasn't been done before.
    '''
    def __init__(self, dataframes):
        self._raw_data = None
        self.dataframes = dataframes

    def _load_data(self):
        print('[INFO] Rebuilding raw data from Parquet files')
        first_date = '2022-05-01 00:00:00'
        last_date = '2023-05-01 00:00:00'
        datetime_range = pd.date_range(first_date, last_date, freq='min')
        datetime_range = pd.DataFrame(datetime_range)
        datetime_range.columns = ['Date']
        datetime_range.set_index('Date', inplace=True)
        datetime_range['filled'] = 'yes'
        raw_data = []

        for coin_name, a_df in self.dataframes.items():
            try:
                a_df.index = pd.to_datetime(a_df.index)
                a_df.index.names = ['Date']
                a_df = datetime_range.join(a_df)
                a_df.close = a_df.close.ffill()
                a_df.loc[a_df.open.isnull(), 'open'] = a_df.loc[a_df.open.isnull()].close
                a_df.loc[a_df.high.isnull(), 'high'] = a_df.loc[a_df.high.isnull()].close
                a_df.loc[a_df.low.isnull(), 'low'] = a_df.loc[a_df.low.isnull()].close
                a_df.taker_buy_quote_asset_volume = a_df.taker_buy_quote_asset_volume.replace(np.NAN, 0)
                a_df.taker_buy_base_asset_volume = a_df.taker_buy_base_asset_volume.replace(np.NAN, 0)
                a_df['Coin'] = coin_name
                a_df.set_index('Coin', append=True, inplace=True)
                del(a_df['filled'])
                print('[INFO] Processing', coin_name)
                raw_data.append(a_df)
            except Exception as e:
                print(f'[ERROR] An error occurred while processing {coin_name}: {e}')
                pass

        if not raw_data:
            print('[ERROR] No dataframes processed. Check if the files are correctly read and processed.')
            return

        raw_data = pd.concat(raw_data, axis=0)
        raw_data = raw_data.sort_index(axis=0, level='Date')
        print('[INFO] Raw data successfully processed')
        print(raw_data.columns)
        raw_data.columns = [
            'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
            'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume'
        ]
        del(raw_data['quote_asset_volume'])
        del(raw_data['number_of_trades'])
        del(raw_data['volume'])
        self._raw_data = raw_data

    def get_full_raw_data(self):
        if self._raw_data is None:
            self._load_data()
        return self._raw_data

# Usage
# data_path = r'Your path here'
# all_files = os.listdir(data_path)
# parquet_files = [file for file in all_files if file.endswith('.parquet')]

# dataframes = {}

# for file in parquet_files:
#     df = pd.read_parquet(os.path.join(data_path, file))
#     dataframes[file[:-8]] = df

# provider = CryptoMinuteDataProvider(dataframes)
# raw_data = provider.get_full_raw_data()
