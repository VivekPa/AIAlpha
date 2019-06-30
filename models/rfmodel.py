import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier as RF 
from sklearn.ensemble import BaggingClassifier as BC 
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

class RFModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def make_model(self, n_estimators, n_jobs, verbose=1):
        model1 = RF(n_estimators=1, criterion='entropy', bootstrap=False, class_weight='balanced_subsample')
        self.model = BC(base_estimator=model1, n_estimators=n_estimators, max_features=1., verbose=verbose)

    def train_model(self, x, y, sample_weights=None):
        self.model.fit(x, y, sample_weight=sample_weights)

    def test_model(self, x, y, sample_weights=None):
        # model_acc = self.model.score(x, y, sample_weight=sample_weights)

        # zeros_count = y['y_values'].value_counts().loc[0]
        # null_acc = zeros_count/len(y)
        
        y_true = pd.DataFrame(index=y.index)
        y_true.loc[y['y_values'] == 1, 'up'] = 1
        y_true.loc[y['y_values'] == -1, 'down'] = 1
        y_true.loc[y['y_values'] == 0, 'no_ch'] = 1
        y_true = y_true.fillna(0)    

        y_pred = self.model.predict_proba(x)
        model_loss = log_loss(y_true, y_pred, sample_weight=sample_weights)

        base_case = pd.DataFrame(index=y.index)
        base_case['up'] = np.zeros(len(y))
        base_case['down'] = np.zeros(len(y))
        base_case['no_ch'] = np.ones(len(y))

        base_loss = log_loss(y_true, base_case)

        # print(f'Model accuracy: {model_acc}')
        # print(f'Null accuracy: {null_acc}')
        print(f'Model log loss: {model_loss}')
        print(f'Base log loss: {base_loss}')