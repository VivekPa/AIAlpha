import pandas as pd
import numpy as np
import bokeh
#import vader

pd.options.mode.chained_assignment = None


df = pd.read_csv('export.csv')
df2=df[['Title','Date','Title Sentiment','Body Sentiment']]
df2['Weighted Sentiment'] = pd.Series(np.random.randn(len(df2['Title'])), index=df2.index)


for i in range(len(df2['Title'])):
    x = df2['Title Sentiment'][int(i)]
    y = df2['Body Sentiment'][int(i)]
    weight = (0.8*x) + (0.2*y)
    df2['Weighted Sentiment'][i] = weight

print(df2)    
