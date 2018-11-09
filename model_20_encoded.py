import tensorflow as tf
from keras.models import Model
import keras.layers as kl
import keras as kr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def nnmodel(epochs, regularizer1, regularizer2):

    train_data = np.array(pd.read_csv("60_return_forex/encoded_return_train_data.csv", index_col=0))
    # length = len(train_data)
    train_data = np.reshape(train_data, (len(train_data), 20))
    print(np.shape(train_data))
    test_data = np.array(pd.read_csv("60_return_forex/encoded_return_test_data.csv", index_col=0))
    test_data = np.reshape(test_data, (len(test_data), 20))
    train_y = np.array(pd.read_csv("forex_y/log_train_y.csv", index_col=0))
    test_y = np.array(pd.read_csv("forex_y/log_test_y.csv", index_col=0))
    price = np.array(pd.read_csv("forex_y/test_price.csv", index_col=0))

    model = kr.models.Sequential()
    # model.add(kl.Dense(50, activation="sigmoid", activity_regularizer=kr.regularizers.l2(0)))
    model.add(kl.Dense(20, input_dim=20, activation="tanh", activity_regularizer=kr.regularizers.l2(regularizer1)))
    model.add(kl.Dense(20, activation="tanh", activity_regularizer=kr.regularizers.l2(regularizer2)))
    # model.add(kl.Dense(100))
    model.add(kl.Dense(1))

    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(train_data, train_y, epochs=epochs)
    model.save("models/final_model.h5")
    predicted_data = []
    predicted_price = []
    for i in range(len(test_data)):
        prediction = model.predict(np.reshape(test_data[i], (1, 20)))
        predicted_data.append(prediction)
        price_pred = np.exp(prediction)*price[i]
        predicted_price.append(price_pred)
        # print(test_data[i])

    # print(model.evaluate(test_data, test_y))
    pd.DataFrame(np.reshape(predicted_price, (len(predicted_price, )))).to_csv("60_return_forex/predicted_price.csv")
    pd.DataFrame(price).to_csv("60_return_forex/price.csv")

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(predicted_data)), np.reshape(test_y, (len(test_y))),
             np.reshape(predicted_data, (len(predicted_data))))
    plt.title("Prediction vs Actual")
    plt.ylabel("Log Return")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(predicted_price)), np.reshape(price, (len(price))),
             np.reshape(predicted_price, (len(predicted_price))))
    plt.xlabel("Time stamp")
    plt.ylabel("Market Price")
    plt.show()

    price_r_score = r2_score(np.reshape(predicted_price, (len(predicted_price))), price)
    return_r_score = r2_score(np.reshape(predicted_data, (len(predicted_data))), test_y)
    price_mse = mean_squared_error(np.reshape(predicted_price, (len(predicted_price))), price)
    return_mse = mean_squared_error(np.reshape(predicted_data, (len(predicted_data))), test_y)

    print(f"Regularizer for 1: {regularizer1} \nRegularizer for 2: {regularizer2} \nEpochs: {epochs}")
    print(f"Predicted Price r^2 value: {price_r_score} \nPredicted return r^2 value: {return_r_score}"
          f"\nPredict Price MSE: {price_mse} \nPredicted Return MSE: {return_mse}")
    dataset = []
    values = np.array([regularizer1, regularizer2, epochs, price_r_score, return_r_score, price_mse, return_mse])
    dataset.append(values)
    dataset = pd.DataFrame(dataset, columns=["regularizer1", "regularizer2", "epochs", "price_r_score", "return_r_score", "price_mse", "return_mse"])
    # print(dataset)
    accuracy = []
    for i in range(len(price)-1):
        acc = 100 - (np.abs(predicted_price[i] - price[i+1]))/price[i+1] * 100
        accuracy.append(acc)
    average = np.mean(accuracy)
    std = np.std(accuracy)
    ret_acc = []
    for i in range(len(test_y)-1):
        if test_y[i] != 0:
            acc = 100 - (np.abs(predicted_data[i] - test_y[i]))/test_y[i] * 100
            ret_acc.append(acc)
    ret_avg = np.mean(ret_acc)
    ret_std = np.std(ret_acc)
    pd.DataFrame(np.reshape(ret_acc, (len(ret_acc, )))).to_csv("60_return_forex/ret_acc.csv")
    prediction = np.exp(model.predict(np.reshape(test_data[-2], (1, 20))))*price[-2]
    print(prediction)

    return dataset, average, std


if __name__ == "__main__":
    dataset, average, std = nnmodel(500, 0.05, 0.01)
    print(f"Price Accuracy Average = {average} \nPrice Accuracy Standard Deviation = {std}")

