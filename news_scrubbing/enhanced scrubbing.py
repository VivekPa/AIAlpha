from random import randint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
import pandas as pd
import nltk
import datetime as dt

indentifiers = ['IN', 'CC', 'CD']

pd.options.mode.chained_assignment = None

analyser = SentimentIntensityAnalyzer()

PUNC_LIST = [".", "!", "?", ",", ";", ":", "-", "'", "\"","!!", "!!!", "??", "???", "?!?", "!?!", "?!?!", "!?!?"]

def print_sentiment_scores(sentence):
    snt = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(snt)))

def attention_check(text):
    templist = text.split()
    tokenlist = nltk.pos_tag(nltk.word_tokenize(text))
    start = len(templist)
    for item in templist:
        if (len(item) < 4) and (len(item) > 1) and (item not in PUNC_LIST):
            if randint(1,4) != 1:
                i = 0
                while i<len(templist):
                    if templist[i] == item:
                        del templist[i]
                        del tokenlist[i]
                    else:
                        i+=1
    for item in templist:
        j = 0
        while j <len(templist) and (len(templist) > (start/5)) and (len(templist)>56):
            if tokenlist[j][1] in indentifiers:
                if randint(0,20) in [i for i in range(7)]:
                    del templist[j]
                    del tokenlist[j]
            else:
                if randint(0,20) in [i for i in range(3)]:
                    del templist[j]
                    del tokenlist[j]
            j+=1
            
    text = ' '.join(templist)
    return text

df = pd.read_csv('news.csv')
df2=df[['Title','Body','Date']]
df2['Overall Sentiment'] = pd.Series(np.random.randn(len(df2['Title'])), index=df2.index)

for i in range(len(df2['Title'])):
    boody = df2['Body'][i]
    weight = TextBlob(attention_check(boody)).sentiment.polarity
    head = TextBlob(df2['Title'][i]).sentiment.polarity
    weight = weight*0.2+head*0.8
    df2['Overall Sentiment'][i] = weight



df3 = pd.read_csv('competitornews.csv')
df4=df3[['Title','Body','Date']]
df4['Overall Sentiment'] = pd.Series(np.random.randn(len(df4['Title'])), index=df4.index)

for i in range(len(df4['Title'])):
    boody = df4['Body'][i]
    weight = TextBlob(attention_check(boody)).sentiment.polarity
    head = TextBlob(df4['Title'][i]).sentiment.polarity
    weight = weight*0.2+head*0.8
    df4['Overall Sentiment'][i] = weight


df5 = pd.concat([df2,df4])
df5.index = range(1,len(df5['Title'])+1)
df5.to_csv("FinalisedNewsData.csv")

weightedavg = (df5['Overall Sentiment'].ewm(halflife=0.3289666).mean())
finalavg = weightedavg.iloc[len(weightedavg)-1]
print(weightedavg)
print(finalavg)
