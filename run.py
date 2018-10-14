from get_data import GetData
from preprocessing import PreProcessing
from autoencoder import AutoEncoder
from data_processing import DataProcessing
from model import NeuralNetwork

data = GetData("AAPL", "2000-01-01", "2018-10-01")
data.get_stock_data()
preprocess = PreProcessing(0.8, 0.25)
preprocess.make_wavelet_train()
preprocess.make_test_data()
autoencoder = AutoEncoder(20)
autoencoder.build_train_model(55, 40, 30, 30, 40)
process = DataProcessing(0.8, 0.25)
process.make_test_data()
process.make_train_data()
process.make_train_y()
process.make_test_y()
model = NeuralNetwork(20, True)
model.make_train_model()
