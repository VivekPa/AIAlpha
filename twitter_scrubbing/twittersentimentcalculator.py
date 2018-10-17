import pandas as pd
import numpy as np
from textblob import TextBlob

pd.options.mode.chained_assignment = None

df = pd.read_csv('twitter.csv')
n = 0
sentiment = 0
df['New sentiment'] = pd.Series(np.random.randn(len(df['sentiment'])), index=df.index)
print(df)
for i in range(len(df)):
	text = TextBlob(df.iloc[i][2])
	newsentiment = text.sentiment.polarity
	sentiment += df.iloc[i][3]
	n += 1
	df['New sentiment'][i] = newsentiment
df.to_csv("twitter.csv")
print(sentiment/n)
