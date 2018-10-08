# AlphaAI: Multilayer neural network architecture for stock return prediction
[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![GitHub license](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/VivekPa/AlphaAI/blob/master/LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

This project is meant to be an **advanced** implementation of **stacked neural networks** to predict the return of stocks. My goal for the viewer is to understand the core principles that go behind the development of such a multilayer model and the nuances of training the individual components for optimal predictive ability. Once the core principles are understood, the various components of the model can be replaced with the state of the art models available at time of usage. 

In essence, we will be downloading stock price data from Yahoo Finance, and use `pandas` and `numpy` to preprocesss the data, making them into dataframes that we can input into the neural network. However, compared to NeuralNetworkStocks, where the preprocessing is simple and only involves making four datasets (training x and y, test x and y), the preprocessing for this neural network is much more complicated as it involves making multiple datasets for the various components of the neural network architecture.
