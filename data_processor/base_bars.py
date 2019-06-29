import pandas as pd 
import numpy as np 

class BaseBars:
    def __init__(self, file_path, output_path, method, threshold, batch_size=20000000):
        self.file_path = file_path
        self.output_path = output_path
        self.method = method
        self.threshold = threshold
        self.batch_size = batch_size
        self.cache = []

    def batch_run(self, verbose=True):
        header = True
        if verbose:
            print(f'Reading data in batches of {self.batch_size}')
        
        count = 0
        cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

        #list_bars = []

        for batch in pd.read_csv(self.file_path, chunksize=self.batch_size, index_col=0):
            if verbose:
                print(f'Sampling batch {count}')
            datetime, list_bars = self._sample(batch)
            full_bars = pd.concat([pd.DataFrame(datetime), pd.DataFrame(list_bars)], axis=1)
            full_bars.columns = cols
            #print(type(list_bars[2][3]))
            #list_bars.columns = cols
            full_bars.to_csv(self.output_path, header=header, index=False, mode='a')
            header = False

    
    def _sample(self, data):
        high_price, low_price, cum_volume, cum_dollar, tick = -np.inf, np.inf, 0, 0, 0
        cache = []
        #cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        datetime = []
        list_bars = []
        #list_bars = pd.DataFrame(columns=cols)
        for row in data.values:
            if high_price < row[2]:
                high_price = row[2]
            if low_price > row[2]:
                low_price = row[2]
            tick += 1
            cum_volume += row[3]
            cum_dollar += row[2]*row[3]
            cache.append(row[2])

            if self.method == "tick":
                if tick == self.threshold:
                    date = row[0]
                    time = row[1]
                    timestamp, bar = self._create_bar(cache, date, time, high_price, low_price, cum_volume, cum_dollar)
                    list_bars.append(bar)
                    datetime.append(timestamp)
                    high_price, low_price, cum_volume, cum_dollar, tick = -np.inf, np.inf, 0, 0, 0
            if self.method == "volume":
                if cum_volume >= self.threshold:
                    date = row[0]
                    time = row[1]
                    timestamp, bar = self._create_bar(cache, date, time, high_price, low_price, cum_volume, cum_dollar)
                    list_bars.append(bar)
                    datetime.append(timestamp)
                    high_price, low_price, cum_volume, cum_dollar, tick = -np.inf, np.inf, 0, 0, 0
            if self.method == "dollar":
                if cum_dollar >= self.threshold:
                    date = row[0]
                    time = row[1]
                    timestamp, bar = self._create_bar(cache, date, time, high_price, low_price, cum_volume, cum_dollar)
                    list_bars.append(bar)
                    datetime.append(timestamp)
                    high_price, low_price, cum_volume, cum_dollar, tick = -np.inf, np.inf, 0, 0, 0
            #print(row[1])
        return datetime, list_bars


    def _create_bar(self, price_list, date, time, high, low, cum_volume, cum_dollar):
        open_price = price_list[0]
        close_price = price_list[-1]
        return [date, time], [float(open_price), float(high), float(low), float(close_price), float(cum_volume)]
