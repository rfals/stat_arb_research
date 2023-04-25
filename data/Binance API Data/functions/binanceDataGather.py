import aiohttp
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
import time
import logging
import nest_asyncio
import sys

# necessary for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# necessary for Jupyter Notebook as it uses its own event loop
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)


class BinanceDataPuller:
    def __init__(self, symbols=None, quote_currency='USDT'):
        self.symbols = symbols
        self.quote_currency = quote_currency
        self.endpoint = 'https://api.binance.com/api/v3/klines'
        self.session = aiohttp.ClientSession()

    async def get_binance_candles(self, symbol, interval, start_time_str, end_time_str):
        start_time = int(datetime.strptime(start_time_str, '%d %b %Y %H:%M:%S').timestamp() * 1000)
        end_time = int(datetime.strptime(end_time_str, '%d %b %Y %H:%M:%S').timestamp() * 1000)

        candle_data = np.array([])

        while start_time < end_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': 1000,
                'startTime': start_time,
                'endTime': end_time
            }
            try:
                async with self.session.get(self.endpoint, params=params) as response:
                    used_weight = int(response.headers.get('X-MBX-USED-WEIGHT-1M', '0'))
                    if used_weight >= 240:
                        sleep_time = 60  # Wait for 60 seconds to reset the rate limit
                        logging.info(f"Rate limit exceeded. Waiting for {sleep_time} seconds.")
                        await asyncio.sleep(sleep_time)

                    response_data = await response.json(content_type=None)
                    if response.status != 200:
                        logging.warning(f"Error fetching data for {symbol}: {response_data}")
                        return np.array([])
            except aiohttp.ContentTypeError as e:
                logging.warning(f"Error fetching data for {symbol}: {e}")
                return np.array([])

            await asyncio.sleep(0.1)  # 100ms delay between requests

            candle_data = np.append(candle_data, np.array(response_data))
            start_time += 1000 * 60  # Increment start_time by 1 minute (60,000 milliseconds)

        candle_data = candle_data.reshape(-1, 6)
        return candle_data


    def save_to_csv(self, data, symbol):
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(f'{symbol}_USDT_2021_2022_1hr.csv', index=False)

    async def fetch_and_save_data(self, symbol, interval, start_time_str, end_time_str):
        start_time = time.time()
        logging.info(f"Fetching data for {symbol}...")

        candle_data = await self.get_binance_candles(symbol, interval, start_time_str, end_time_str)

        if candle_data.size == 0:
            logging.warning(f"No data found for {symbol}")
        else:
            self.save_to_csv(candle_data, symbol)
            elapsed_time = time.time() - start_time
            logging.info(f"Data saved to {symbol}_USDT_2021_2022_1hr.csv in {elapsed_time:.2f} seconds")

    async def fetch_all(self, interval, start_time_str, end_time_str):
        tasks = []

        for symbol in self.symbols:
            task = asyncio.create_task(self.fetch_and_save_data(symbol, interval, start_time_str, end_time_str))
            tasks.append(task)

        await asyncio.gather(*tasks)

        async def close(self):
            await self.session.close()


async def main():
    start_time_str = '01 Jan 2021 00:00:00'
    end_time_str = '31 Dec 2023 23:59:00'

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'SOLUSDT', 'DOTUSDT', 'LTCUSDT']

    fetcher = BinanceDataPuller(symbols=symbols)
    await fetcher.fetch_all('1m', start_time_str, end_time_str)
    await fetcher.close()


if __name__ == '__main__':
    nest_asyncio.apply()
    asyncio.run(main())

