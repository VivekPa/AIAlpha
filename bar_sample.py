import pandas as pd 
import numpy as np 
from data_processor.base_bars import BaseBars

print('Creating tick bars...')
base = BaseBars("data/raw_data/price_vol.csv", "data/processed_data/price_bars/tick_bars.csv", "tick", 10)
base.batch_run()

print('Creating dollar bars...')
base = BaseBars("data/raw_data/price_vol.csv", "data/processed_data/price_bars/dollar_bars.csv", "dollar", 20000)
base.batch_run()

print('Creating volume bars...')
base = BaseBars("data/raw_data/price_vol.csv", "data/processed_data/price_bars/volume_bars.csv", "volume", 50)
base.batch_run()