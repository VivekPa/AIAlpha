# AlphaAI: Multilayer neural network architecture for stock return prediction
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![GitHub license](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/VivekPa/AlphaAI/blob/master/LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This project is meant to be an **advanced** implementation of **stacked neural networks** to predict the return of stocks. My goal for the viewer is to understand the core principles that go behind the development of such a multilayer model and the nuances of training the individual components for optimal predictive ability. Once the core principles are understood, the various components of the model can be replaced with the state of the art models available at time of usage. 

In essence, we will be downloading stock price data from Yahoo Finance, and use `pandas` and `numpy` to preprocesss the data, making them into dataframes that we can input into the neural network. However, compared to [NeuralNetworkStocks](https://github.com/VivekPa/NeuralNetworkStocks), where the preprocessing is simple and only involves making four datasets (training x and y, test x and y), the preprocessing for this neural network architecture is much more complicated as it involves making multiple datasets for the various components of the neural network architecture. Once we have preprocessed the data, we use wavelet transform to denoise the data, subsequently using stacked autoencoders to extract features from the data and finally inputting the features into a LSTM model to predict the stock return. 

This model is not meant to be used to live trade without modifications. However, an extended version of this model can very well be profitable with the right strategies. 

I truly hope you find this project informative and useful in developing your own trading strategies or machine learning models. If you found this project interesting, do **leave a star**.

*Disclaimer, this is purely an educational project. Any backtesting performance do not guarentee live trading results. Trade at your own risk.*
*This is only a guide on the usage of the model. If you want to delve into the reasoning behind the model and the theory, please check out my blog: [Engineer Quant](https://medium.com/engineer-quant)*

## Contents
- [Contents](#contents)
- [Overview](#overview)
- [Quickstart](#quickstart)
- [Stock Price Data](#stock-price-data)
- [Preprocessing](#preprocessing)
  - [Preparing Autoencoder Train Dataset](#preparing-train-dataset)
  - [Preparing Autoencoder Test Dataset](#preparing-test-dataset)
- [Stacked Autoencoder](#stacked-autoencoder)
- [Data Processing](#data-processing)
  - [Preparing Train Dataset](#preparing-train-dataset)
  - [Preparing Test Dataset](#preparing-test-dataset)
- [Neural Network Model](#neural-network-model)
  - [LSTM Model](#lstm-model)
- [Backtesting](#backtesting)
- [Stock Predictions](#stock-predictions)
- [Online Learning](#online-learning)
- [What next?](#what-next?)
  - [Getting Data](#getting-data)
  - [Neural Network Model](#neuron-network-model)
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
8. Supplement your trading strategies with the predictions

As you can see, this pipeline is much longer, and even a bit confusing at first glance. Perhaps this flowchart will make things simpler:

![alt text][flowchart]

[flowchart]: https://engfinance.files.wordpress.com/2018/10/alphaai-flowchart.png "Pipeline Flowchart"

Now let me elaborate the various parts of the pipeline.

## Quickstart

For those who just want to see the model work, run the following code (make sure you are on Python 3 to prevent any bugs or errors):

```bash
pip install -r requirements.txt
python full_run.py
```
## Stock Price Data

We are going to use Yahoo Finance API to get our stock data. However, since the API is being shut down, it might be a good idea to develop your own stock database. 


