import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

class TrendPredictor:

    def __init__(self, history_data):
        self.history_data = history_data

        self.predictors = None
        self.response = None
        self.n = None
        self.m = None

        self.predicted_value = None
        self.predicted_trend = None
        self.today = None
        pass

    def prepare_data(self):
        self.history_data = self.history_data.dropna()
        self.today = self.history_data.index[-1]
        self.n = self.history_data.shape[0]
        self.m = self.history_data.shape[1]
        X = self.history_data.iloc[0: self.n - 1, 0: self.m - 1]
        X = np.array(X)
        self.predictors = X.reshape(self.n - 1, self.m - 1)
        self.response = np.array(self.history_data.iloc[0: self.n - 1, self.m - 1]).reshape(self.n - 1, 1)


    def linear_regression(self):

        self.prepare_data()

        X = self.predictors
        y = self.response

        model = LinearRegression()
        model.fit(X, y)

        X_pred = np.array(self.history_data.iloc[self.n-1, 0: self.m-1])
        X_pred = X_pred.reshape(1, self.m-1)
        self.predicted_value = model.predict(X_pred)[0][0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend =  1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0

        pass

    def lasso_regression(self):
        self.prepare_data()
        X = self.predictors
        y = self.response

        model = Lasso(alpha = 0.1)
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0

        pass

    def ridge_regression(self):
        self.prepare_data()
        X = self.predictors
        y = self.response

        model = Ridge(alpha=2, solver='cholesky')
        model.fit(X, y)
        X_pred = np.array(self.history_data.iloc[self.n - 1, 0: self.m - 1])
        X_pred = X_pred.reshape(1, self.m - 1)
        self.predicted_value = model.predict(X_pred)[0][0]

        Close_today = y[-1]
        if self.predicted_value > Close_today:
            self.predicted_trend = 1
        elif self.predicted_value < Close_today:
            self.predicted_trend = -1
        else:
            self.predicted_trend = 0

        pass
