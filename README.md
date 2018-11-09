# AlphaAI: Multilayer neural network architecture for stock return prediction
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![GitHub license](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/VivekPa/AlphaAI/blob/master/LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This project is meant to be an **advanced** implementation of **stacked neural networks** to predict the return of stocks. My goal for the viewer is to understand the core principles that go behind the development of such a multilayer model and the nuances of training the individual components for optimal predictive ability. Once the core principles are understood, the various components of the model can be replaced with the state of the art models available at time of usage. 

In essence, we will be downloading stock price data from Yahoo Finance, and use `pandas` and `numpy` to preprocesss the data, making them into dataframes that we can input into the neural network. However, compared to [NeuralNetworkStocks](https://github.com/VivekPa/NeuralNetworkStocks), where the preprocessing is simple and only involves making four datasets (training x and y, test x and y), the preprocessing for this neural network architecture is much more complicated as it involves making multiple datasets for the various components of the neural network architecture. Once we have preprocessed the data, we use **wavelet transform** to *denoise the data*, subsequently using **stacked autoencoders** to *extract features* from the data and finally inputting the features into a **LSTM model** to predict the stock return. 

This model is not meant to be used to live trade without modifications. However, an extended version of this model can very well be profitable with the right strategies. 

I truly hope you find this project informative and useful in developing your own trading strategies or machine learning models. If you found this project interesting, do **leave a star**.

*Disclaimer, this is purely an educational project. Any backtesting performance do not guarentee live trading results. Trade at your own risk.*
*This is only a guide on the usage of the model. If you want to delve into the reasoning behind the model and the theory, please check out my blog: [Engineer Quant](https://medium.com/engineer-quant)*

## Contents
- [Contents](#contents)
- [Overview](#overview)
- [Quickstart](#quickstart)
- [Wavelet Transform](#wavelet-transform)
- [Stacked Autoencoder](#stacked-autoencoder)
- [Neural Network Model](#neural-network-model)
- [Results](#results)
- [Online Learning](#online-learning)
- [What next?](#what-next?)
  - [Getting Data](#getting-data)
  - [Neural Network Model](#neural-network-model)
  - [Supporting Trade](#supporting-trade)
- [Contributing](#contributing)

## Overview

Those who have done some form of machine learning would know that the workflow follows this format: acquire data, preprocess, train, test, monitor model. However, given the complexity of this model, the workflow has been modified to the following:

1. Acquire the stock price data - this is the primary data for our model.
2. Preprocess the data - denoise data and make the train, test datasets for the stacked autoencoders.
3. Train the stacked autoencoder - this will give us our feature extractor.
4. Process the data - this will give us the *features* of our model, along with train, test datasets.
5. Use the neural network to learn from the training data.
6. Test the model with the testing set - this gives us a gauge of how good our model is.
7. Make useful stock price predictions 

As you can see, this pipeline is much longer, and even a bit confusing at first glance. Perhaps this flowchart will make things simpler:

![alt text][flowchart]

[flowchart]: https://engfinance.files.wordpress.com/2018/10/alphaai_diagram_final.png "Pipeline Flowchart"

Now let me elaborate the various parts of the pipeline.

## Quickstart

For those who just want to see the model work, run the following code (make sure you are on Python 3 to prevent any bugs or errors):

```bash
pip install -r requirements.txt
python run.py
```
## Wavelet Transform

The objective at this stage is to denoise the stock price data, which, due to complex market dynamics, are usually unncessarily noisy. This noisy can cloud up the machine learning algorithm and produce poorer results. Hence, I have chosen to use the wavelet transform to denoise the data. This can be done using the library `pywt` which is excellent for those interested in using wavelet transforms. 

## Stacked Autoencoder

Feature extraction is a crucial part of building an effective machine learning model, but often involves expert domain knowledge. However, we can attempt to overcome this by using neural networks that are able to decompress the data given into smaller number of neurons than the input number. When we train such a neural network, it becomes able to extract the 'important sections' of the data so to speak. Hence, this compressed version of the data can be considered as *features*. Although this method is useful, the downside is that we do not know what the various compressed data points mean and hence cannot extract methods to achieve them in differnt datasets. 

## Neural Network Model

Using neural networks for the prediction of time series has become widespread and the power of neural networks is well known. I have also chosen to use multilayer perceptron neural networks to predict the stock return given the features extracted from our data. I have found that MLP has a greater predictive power compared to LSTM due to the autoencoding, which results in the loss of the time series nature of the data. 

## Results

Using this stacked neural network model, I was able to achieve pretty good results. The following are graphs of my predictions vs the actual market prices for various securities.

EURUSD

![alt text][EURUSD]

[EURUSD]: https://engfinance.files.wordpress.com/2018/11/figure_1-4.png "Prediction 1"

EURUSD prices - R^2: 0.90

![alt text][EURUSD2]

[EURUSD2]: https://engfinance.files.wordpress.com/2018/11/figure_1-5.png "Prediction 2"


## Online Learning

The training normally stops after the model has trained on historic data and merely predicts future data. However, I believe that it might be a waste of data if the model does not also learn from the predictions. This is done by training the model on the new (prediction, actual) pairs to continually improve the model. 

## What's next?

The beauty of this model is the once the construction is understood, the individual models can be swapped out for the best model there is. So over time the actual models used here will be different but the core framework will still be the same. 

### Getting Data

One issue with Yahoo Finance is that the API has been removed and although `fix_yahoo_finance` does a fantastic job making it easier to get the data, I believe that it would be helpful for long. Hence, I have started working on making a financial database to store the stock prices and if possible, fundamental data prices. 

### Neural Network Model

As mentioned before, the subcompartments can be modified to make the model better. I am constantly looking for neural network models better that LSTM that to do the job. 

### Supporting Trade

If the predictive power of the neural network gets to a point where it can support trading strategies, then it might be lucrative to use it to build a strategy. I am currently working on using **Reinforcement Learning** to make a trading agent that will learn the trading strategy to maximise the portfolio. 

## Contributing

I am always grateful for feedback and modifications that would help! 

Hope you have enjoyed that! To see more content like this, please visit: [Engineer Quant](https://medium.com/engineer-quant)
