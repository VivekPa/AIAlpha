from get_data import GetData
from preprocessing import PreProcessing
from autoencoder import AutoEncoder
from data_processing import DataProcessing
from model import NeuralNetwork
from model_20_encoded import nnmodel

dataset, average, std = nnmodel(500, 0.01, 0.01)
print(f"Price Accuracy Average = {average} \nPrice Accuracy Standard Deviation = {std}")
