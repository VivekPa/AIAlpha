import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataProcessing:
    def __init__(self, split):
        self.split = split
    
    def make_features(self, file_path, window, csv_path, make_y=True, verbose=True, save_csv=False):
        df = pd.read_csv(f"{file_path}", index_col=0)
        print(df.shape)
        cols = df.columns
        #print(type(df[cols[1]].iloc[1]))
        #print(df[cols[2]].head())
        for i in cols[1:]:
            print(i)
            df[f'{i}_ret1'] = np.log(df[i]/df[i].shift(1))
            #df[f'{i}_autocorr1'] = df[f'{i}_ret1'].corr(df[f'{i}_ret1'].shift(1))
            for j in range(2, window, 2):
                #print((df[i]/df[i].shift(1)).head())
                df[f'{i}_ret{j}'] = np.log(df[i]/df[i].shift(j))
                df[f'{i}_mavg{j}'] = df[f'{i}_ret1'].rolling(j).mean()
                df[f'{i}_ewm{j}'] = df[f'{i}_ret1'].ewm(span=j).mean()
                #df[f'{i}_autocorr{j}'] = df[f'{i}_ret1'].corr(df[f'{i}_ret1'].shift(j))
            if window > 10:
                for j in range(10, window, 5):
                    df[f'{i}_vol{j}'] = df[f'{i}_ret1'].rolling(j).std()
                    df[f'{i}_ewmvol{j}'] = df[f'{i}_ret1'].ewm(span=j).std()
                    break
        df['liq'] = df['close']*df['volume']
        df['liq_ret1'] = df['liq']/df['liq'].shift(1)
        #df['liq_autocorr1'] = df['liq'].corr(df['liq'].shift(1))
        for j in range(2, window, 2):
            df[f'liq_ret{j}'] = df['liq']/df['liq'].shift(j)
            df[f'liq_mavg{j}'] = df['liq'].rolling(j).mean()
            df[f'liq_ewm{j}'] = df['liq'].ewm(span=j).mean()
            #df[f'liq_autocorr{j}'] = df['liq'].corr(df['liq'].shift(j))
        if window > 10:
            for j in range(10, window, 5):
                #df[f'liq_vol{j}'] = df['liq'].rolling(j).std()
                #df[f'liq_ewmvol{j}'] = df['liq'].ewm(span=j).std()
                break
        if verbose:
            df = df.dropna()
            print(df.shape)
            print(df.tail())
            print(df.head())
        if save_csv:
            df.to_csv(f'{csv_path}/full_features.csv')
        return df

    def make_train_test(self, df_x, df_y, window, csv_path, has_y=False, binary_y=False, save_csv=False):
        """
        Splits the dataset into train and test
        :param df_x: dataframe of x variables
        :type df_x: pd.DataFrame
        :param df_y: dataframe of y values
        :type df_y: pd.DataFrame
        :param window: the prediction window
        :type window: int
        :param has_y: whether df_y exists separately or is a column in df_x (must be 'target' column)
        :type has_y: boolean
        :return: train_x, train_y, test_x, test_y
        :rtype: pd.DataFrames
        """
        if has_y:
            y_values = df_y.copy()
            y_values.columns = ['y_values']
            fulldata = df_x.copy()
        else:
            if window == 0:
                y_values = df_x['close'].copy()
                y_values.columns = ['y_values']
                fulldata = df_x.copy()
            else:
                y_values = np.log(df_x['close'].copy()/df_x['close'].copy().shift(-window)).dropna()
                y_values.columns = ['y_values']
                fulldata = df_x.iloc[:-window, :].copy()           
        if binary_y:
            y_values.loc[y_values['y_values']<0] = -1
            y_values.loc[y_values['y_values']>0] = 1
            y_values.loc[y_values['y_values']==0] = 0
        print(y_values.shape)
        print(fulldata.shape)
        train_y = y_values.iloc[:int(len(y_values)*self.split)]
        test_y = y_values.iloc[int(len(y_values)*self.split)+1:]

        train_x = fulldata.iloc[:int(len(y_values)*self.split), :]
        test_x = fulldata.iloc[int(len(y_values)*self.split)+1:len(y_values), :]

        print(train_y.shape)
        print(train_x.shape)

        if save_csv:
            train_x.to_csv(f'data/processed_data/{csv_path}/train_x.csv')
            train_y.to_csv(f'data/processed_data/{csv_path}/train_y.csv', header=['y_values'])
            test_x.to_csv(f'data/processed_data/{csv_path}/test_x.csv')
            test_y.to_csv(f'data/processed_data/{csv_path}/test_y.csv', header=['y_values'])
            fulldata.to_csv(f'data/processed_data/{csv_path}/full_x.csv')
            y_values.to_csv(f'data/processed_data/{csv_path}/full_y.csv', header=['y_values'])
        return fulldata, y_values, train_x, train_y, test_x, test_y
    
    def check_labels(self, y_values):
        print(y_values['y_values'].value_counts())


if __name__ == "__main__":
    # preprocess = DataProcessing(0.8)
    # df = preprocess.make_features(file_path="price_bars/tick_bars.csv", window=20,  
    #     csv_path="autoencoder_data", save_csv=True)
    # fulldata, y_values, train_x, train_y, test_x, test_y =  preprocess.make_train_test(df_x=df, df_y=None, window=1, 
    # csv_path="autoencoder_data", save_csv=True)

    preprocess = DataProcessing(0.8)
    df1 = pd.read_csv("../data/processed_data/nn_data/full_x.csv", index_col=0) 
    df2 = pd.read_csv('../data/processed_data/autoencoder_data/full_y.csv', index_col=0)
    fulldata, y_values, train_x, train_y, test_x, test_y =  preprocess.make_train_test(df_x=df1, df_y=df2, window=1, 
    csv_path="train_test_data", has_y=True, save_csv=True)
