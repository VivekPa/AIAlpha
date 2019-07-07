import pandas as pd 
import numpy as np 

df = pd.read_csv('data/raw_data/price_vol.csv', index_col=0)
print(df.shape)

sample_data = df.iloc[:1000000, :]
sample_data.to_csv('sample_data/raw_data/price_vol.csv')