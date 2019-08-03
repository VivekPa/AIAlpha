from models.autoencoder import AutoEncoder
from models.nnmodel import NNModel
from models.rfmodel import RFModel
from data_processor.data_processing import DataProcessing
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from data_processor.base_bars import BaseBars


print('Creating tick bars...')
base = BaseBars("sample_data/raw_data/price_vol.csv", "sample_data/processed_data/price_bars/tick_bars.csv", "tick", 10)
base.batch_run()

print('Creating dollar bars...')
base = BaseBars("sample_data/raw_data/price_vol.csv", "sample_data/processed_data/price_bars/dollar_bars.csv", "dollar", 20000)
base.batch_run()

print('Creating volume bars...')
base = BaseBars("sample_data/raw_data/price_vol.csv", "sample_data/processed_data/price_bars/volume_bars.csv", "volume", 50)
base.batch_run()

print('Processing data...')
preprocess = DataProcessing(0.8)
df = preprocess.make_features(file_path=f"sample_data/processed_data/price_bars/dollar_bars.csv", window=20,  
    csv_path="sample_data/processed_data/autoencoder_data", save_csv=True)
fulldata, y_values, train_x, train_y, test_x, test_y =  preprocess.make_train_test(df_x=df, df_y=None, window=1, 
csv_path="sample_data/processed_data/autoencoder_data", save_csv=True)

print('Loading data...')
a_train_x = pd.read_csv('sample_data/processed_data/autoencoder_data/train_x.csv', index_col=0)
a_train_y = pd.read_csv('sample_data/processed_data/autoencoder_data/train_y.csv', index_col=0)
a_test_x = pd.read_csv('sample_data/processed_data/autoencoder_data/test_x.csv', index_col=0)
a_test_y = pd.read_csv('sample_data/processed_data/autoencoder_data/test_y.csv', index_col=0)
print(a_train_x.head())
print(a_train_x.shape)

print('Scaling data...')
scaler = MinMaxScaler(feature_range=(-1, 1))
x_train_a = scaler.fit_transform(a_train_x.iloc[:, 1:])
x_test_a = scaler.transform(a_test_x.iloc[:, 1:])

autoencoder = AutoEncoder(20, x_train_a.shape[1])
autoencoder.build_model(100, 50, 50, 100)

print('Training model...')
autoencoder.train_model(autoencoder.autoencoder, x_train_a, epochs=20, model_name='autoencoder')

print('Testing model...')
autoencoder.test_model(autoencoder.autoencoder, x_test_a)

print('Encoding data...')
a_full_data = pd.read_csv('sample_data/processed_data/autoencoder_data/full_x.csv', index_col=0)
a_scaled_full = pd.DataFrame(scaler.transform(a_full_data.iloc[:, 1:]))
autoencoder.encode_data(a_scaled_full, csv_path='sample_data/processed_data/nn_data/full_x.csv')

print('Processing data...')
preprocess = DataProcessing(0.8)
df1 = pd.read_csv("sample_data/processed_data/nn_data/full_x.csv", index_col=0) 
df2 = pd.read_csv('sample_data/processed_data/autoencoder_data/full_y.csv', index_col=0)
fulldata, y_values, train_x, train_y, test_x, test_y =  preprocess.make_train_test(df_x=df1, df_y=df2, window=1, 
csv_path="rf_data", has_y=True, binary_y=True, save_csv=True)
y = pd.read_csv('sample_data/processed_data/rf_data/full_y.csv', index_col=0)
preprocess.check_labels(y)

print('Loading data...')
train_x = pd.read_csv('sample_data/processed_data/rf_data/train_x.csv', index_col=0)
train_y = pd.read_csv('sample_data/processed_data/rf_data/train_y.csv', index_col=0)
test_x = pd.read_csv('sample_data/processed_data/rf_data/test_x.csv', index_col=0)
test_y = pd.read_csv('sample_data/processed_data/rf_data/test_y.csv', index_col=0)
print(train_x.head())
print(train_y.shape)

print('Scaling data...')
scaler = MinMaxScaler(feature_range=(-1, 1))
x_train = scaler.fit_transform(train_x)
x_test = scaler.transform(test_x)

# nnmodel = NNModel(x_train.shape[1])
# nnmodel.make_model()

# print('Training model...')
# nnmodel.train_model(train_x, train_y, model_name='nnmodel', epochs=2)

# print("Testing model...")
# nnmodel.test_model(x_test, test_y)

# print('Making predictions...')
# pred_ret = nnmodel.predict_ret(x_test, y=None)

# plt.plot(pred_ret)
# plt.plot(np.array(test_y))
# plt.show()

# print(x_train.shape)
# print(train_y.shape)
# print(x_test.shape)
# print(test_y.shape)

rfmodel = RFModel(x_train.shape[1])
rfmodel.make_model(300, -1, verbose=1)
rfmodel.train_model(x_train, train_y)
rfmodel.test_model(x_test, test_y)

rfmodel = RFModel(x_train_a.shape[1])
rfmodel.make_model(300, -1, verbose=1)
rfmodel.train_model(x_train_a, train_y)
rfmodel.test_model(x_test_a, test_y)
