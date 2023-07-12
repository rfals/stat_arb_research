import aiohttp
import asyncio
from asyncio import Semaphore
import numpy as np
import pandas as pd
from datetime import datetime
import time
import logging
import nest_asyncio
import sys
import os

# necessary for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# necessary for Jupyter Notebook as it uses its own event loop
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)


class BinanceDataPuller:
    def __init__(self, symbols=None, quote_currency='USDT', output_path=''):
        self.symbols = symbols
        self.quote_currency = quote_currency
        self.endpoint = 'https://api.binance.com/api/v3/klines'
        self.output_path = output_path
        self.session = aiohttp.ClientSession()
        self.interval_seconds = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '8h': 28800,
            '12h': 43200,
            '1d': 86400,
            '3d': 259200,
            '1w': 604800,
            '1M': 2592000
        }
        self.rate_limiter = Semaphore(10)  # Adjust the number of concurrent requests

    async def get_binance_candles(self, symbol, interval, start_time_str, end_time_str):
        async with self.rate_limiter:

            start_time = int(datetime.strptime(start_time_str, '%d %b %Y %H:%M:%S').timestamp() * 1000)
            end_time = int(datetime.strptime(end_time_str, '%d %b %Y %H:%M:%S').timestamp() * 1000)

            candle_data_list = []
            backoff = 0.05  # 50ms backoff between requests
            while start_time < end_time:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': 1000,
                    'startTime': start_time,
                    'endTime': end_time
                }
                try:
                    async with self.session.get(self.endpoint, params=params, timeout=10) as response:
                        used_weight = int(response.headers.get('X-MBX-USED-WEIGHT-1M', '0'))

                        if response.status == 429:  # HTTP 429: Too Many Requests
                            sleep_time = int(response.headers.get('Retry-After', '60')) + backoff
                            logging.warning(f"Rate limit exceeded. Waiting for {sleep_time} seconds.")
                            await asyncio.sleep(sleep_time)
                            backoff *= 2
                        elif response.status != 200:
                            logging.warning(f"Error fetching data for {symbol}: {response_data}")
                            return np.array([])
                        else:
                            backoff = max(0.1, backoff / 2)
                            response_data = await response.json(content_type=None)
                except aiohttp.ContentTypeError as e:
                    logging.warning(f"Error fetching data for {symbol}: {e}")
                    start_time += self.interval_seconds[interval] * 1000
                    return np.array([])

                await asyncio.sleep(backoff) 

                candle_data_list.extend([sublist[:6] for sublist in response_data])
                start_time += self.interval_seconds[interval] * 1000

        candle_data = np.array(candle_data_list, dtype=object).reshape(-1, 6)
        return candle_data

    def save_to_parquet(self, data, symbol, interval):
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.DataFrame(data, columns=columns)
        file_name = f'{symbol}_USDT_2021_2023_{interval}.parquet'
        file_path = os.path.join(self.output_path, file_name)
        df.to_parquet(file_path, index=False)

    async def fetch_and_save_data(self, symbol, interval, start_time_str, end_time_str):
        start_time = time.time()
        logging.info(f"Fetching data for {symbol}...")

        candle_data = await self.get_binance_candles(symbol, interval, start_time_str, end_time_str)

        if candle_data.size == 0:
            logging.warning(f"No data found for {symbol}")
        else:
            self.save_to_parquet(candle_data, symbol, interval)
            elapsed_time = time.time() - start_time
            logging.info(f"Data saved to {symbol}_USDT_2021_2023_{interval}.parquet in {elapsed_time:.2f} seconds")

    async def fetch_all(self, interval, start_time_str, end_time_str):
        tasks = []

        for symbol in self.symbols:
            task = asyncio.create_task(self.fetch_and_save_data(symbol, interval, start_time_str, end_time_str))
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def close(self):
        await self.session.close()

async def main():
    start_time_str = '01 Jan 2023 00:00:00'
    end_time_str = '25 Jan 2023 23:59:00'

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'SOLUSDT', 'DOTUSDT', 'LTCUSDT']
    output_path = r'C:\Users\ReinisFals\OneDrive - Peero, SIA\Desktop\stat_arb_research\stat_arb_research\data\Binance API Data\test_1m'

    fetcher = BinanceDataPuller(symbols=symbols)
    await fetcher.fetch_all('1m', start_time_str, end_time_str)
    await fetcher.close()

if __name__ == '__main__':
    asyncio.run(main())


# date_format = "%d %b %Y %H:%M:%S"
# start_date = datetime.strptime(start_date_str, date_format)
# end_date = datetime.strptime(end_date_str, date_format)
# difference = end_date - start_date
# minutes_difference = difference.total_seconds() / 60
# print(f"The number of minutes between {start_date_str} and {end_date_str} is {minutes_difference:.0f} minutes.")