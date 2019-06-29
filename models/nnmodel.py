import keras.layers as kl
from keras.models import Model
from keras import regularizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class NNModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def make_model(self):
        input_data = kl.Input(shape=(1, self.input_shape))
        lstm = kl.LSTM(5, input_shape=(1, self.input_shape), return_sequences=True, activity_regularizer=regularizers.l2(0.003),
                       recurrent_regularizer=regularizers.l2(0), dropout=0.2, recurrent_dropout=0.2)(input_data)
        perc = kl.Dense(5, activation="sigmoid", activity_regularizer=regularizers.l2(0.005))(lstm)
        lstm2 = kl.LSTM(2, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
                        dropout=0.2, recurrent_dropout=0.2)(perc)
        out = kl.Dense(1, activation="sigmoid", activity_regularizer=regularizers.l2(0.001))(lstm2)

        model = Model(input_data, out)

        self.model = model

    def train_model(self, x, y, epochs, model_name, save_model=True):
        self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse", "acc"])

        # load data

        train_x = np.reshape(np.array(x), (len(x), 1, self.input_shape))
        train_y = np.array(y)
        # train_stock = np.array(pd.read_csv("train_stock.csv"))

        # train model

        self.model.fit(train_x, train_y, epochs=epochs)
        
        if save_model:
            self.model.save(f"models/saved_models/{model_name}.h5", overwrite=True, include_optimizer=True)

    def test_model(self, x, y):
        test_x = np.reshape(np.array(x), (len(x), 1, self.input_shape))
        test_y = np.array(y)

        print(self.model.evaluate(test_x, test_y))

    def predict_ret(self, x, y):
        test_x = x
        predicted_data = []
        for i in test_x:
            prediction = (self.model.predict(np.reshape(i, (1, 1, self.input_shape))))
            predicted_data.append(np.reshape(prediction, (1,)))
        return pd.DataFrame(predicted_data)

    def predict_prices(self, x, y, prices):
        test_x = x
        test_y = y
        prices = prices
        prediction_data = []
        stock_data = []
        for i in range(len(test_y)):
            prediction = (self.model.predict(np.reshape(test_x[i, :], (1, 1, self.input_shape))))
            prediction_data.append(np.reshape(prediction, (1,)))
            pred_price = np.exp(np.reshape(prediction, (1,)))*prices[i]
            stock_data.append(pred_price)
        return prediction_data, stock_data

