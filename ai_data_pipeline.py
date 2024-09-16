from models.autoencoder import AutoEncoder
from models.nnmodel import NNModel
from models.rfmodel import RFModel
from data_processor.data_processing import DataProcessing
from data_processor.feature_engineering import AdvancedFeatureEngineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt
from data_processor.base_bars import BaseBars
import joblib

# Define constants
RAW_DATA_PATH = "sample_data/raw_data/price_vol.csv"
PROCESSED_DATA_PATH = "sample_data/processed_data/"
TICK_BARS_PATH = PROCESSED_DATA_PATH + "price_bars/tick_bars.csv"
DOLLAR_BARS_PATH = PROCESSED_DATA_PATH + "price_bars/dollar_bars.csv"
VOLUME_BARS_PATH = PROCESSED_DATA_PATH + "price_bars/volume_bars.csv"
AUTOENCODER_DATA_PATH = PROCESSED_DATA_PATH + "autoencoder_data/"
NN_DATA_PATH = PROCESSED_DATA_PATH + "nn_data/"
RF_DATA_PATH = PROCESSED_DATA_PATH + "rf_data/"
FULL_X_PATH = AUTOENCODER_DATA_PATH + "full_x.csv"
FULL_Y_PATH = AUTOENCODER_DATA_PATH + "full_y.csv"
TRAIN_X_PATH = RF_DATA_PATH + "train_x.csv"
TRAIN_Y_PATH = RF_DATA_PATH + "train_y.csv"
TEST_X_PATH = RF_DATA_PATH + "test_x.csv"
TEST_Y_PATH = RF_DATA_PATH + "test_y.csv"

print('Creating tick bars...')
base = BaseBars(RAW_DATA_PATH, TICK_BARS_PATH, "tick", 10)
base.batch_run()

print('Creating dollar bars...')
base = BaseBars(RAW_DATA_PATH, DOLLAR_BARS_PATH, "dollar", 20000)
base.batch_run()

print('Creating volume bars...')
base = BaseBars(RAW_DATA_PATH, VOLUME_BARS_PATH, "volume", 50)
base.batch_run()

print('Processing data...')
preprocess = DataProcessing(0.8)
df = preprocess.make_features(file_path=DOLLAR_BARS_PATH, window=20,  
    csv_path=AUTOENCODER_DATA_PATH, save_csv=True)
fulldata, y_values, train_x, train_y, test_x, test_y = preprocess.make_train_test(df_x=df, df_y=None, window=1, 
csv_path=AUTOENCODER_DATA_PATH, save_csv=True)

print('Loading data...')
a_train_x = pd.read_csv(TRAIN_X_PATH, index_col=0)
a_train_y = pd.read_csv(TRAIN_Y_PATH, index_col=0)
a_test_x = pd.read_csv(TEST_X_PATH, index_col=0)
a_test_y = pd.read_csv(TEST_Y_PATH, index_col=0)
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
a_full_data = pd.read_csv(FULL_X_PATH, index_col=0)
a_scaled_full = pd.DataFrame(scaler.transform(a_full_data.iloc[:, 1:]))
autoencoder.encode_data(a_scaled_full, csv_path=NN_DATA_PATH + 'full_x.csv')

print('Processing data...')
preprocess = DataProcessing(0.8)
df1 = pd.read_csv(NN_DATA_PATH + "full_x.csv", index_col=0) 
df2 = pd.read_csv(FULL_Y_PATH, index_col=0)
fulldata, y_values, train_x, train_y, test_x, test_y = preprocess.make_train_test(df_x=df1, df_y=df2, window=1, 
csv_path=RF_DATA_PATH, has_y=True, binary_y=True, save_csv=True)
y = pd.read_csv(RF_DATA_PATH + 'full_y.csv', index_col=0)
preprocess.check_labels(y)

print('Loading data...')
train_x = pd.read_csv(TRAIN_X_PATH, index_col=0)
train_y = pd.read_csv(TRAIN_Y_PATH, index_col=0)
test_x = pd.read_csv(TEST_X_PATH, index_col=0)
test_y = pd.read_csv(TEST_Y_PATH, index_col=0)
print(train_x.head())
print(train_y.shape)

print('Scaling data...')
scaler = MinMaxScaler(feature_range=(-1, 1))
x_train = scaler.fit_transform(train_x)
x_test = scaler.transform(test_x)

print('Training Random Forest model...')
rfmodel = RFModel(x_train.shape[1])
rfmodel.make_model(300, -1, verbose=1)
rfmodel.train_model(x_train, train_y)
rfmodel.test_model(x_test, test_y)

print('Training AutoEncoder-based Random Forest model...')
rfmodel = RFModel(x_train_a.shape[1])
rfmodel.make_model(300, -1, verbose=1)
rfmodel.train_model(x_train_a, train_y)
rfmodel.test_model(x_test_a, test_y)

# Save models
joblib.dump(rfmodel, 'rfmodel.pkl')
joblib.dump(autoencoder, 'autoencoder.pkl')

print('Models saved.')

