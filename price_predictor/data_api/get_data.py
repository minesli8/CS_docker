from pandas_datareader import data


class Stock:

    def __init__(self, stock_name, source, date_begin, date_end):
        self.stock_name = stock_name
        self.source = source
        self.date_begin = date_begin
        self.date_end = date_end

        self.stock_data = None
        self.stock_data_lagged = None
        pass

    def get_data(self):
        self.stock_data = data.DataReader(self.stock_name, self.source, self.date_begin, self.date_end)

    def lag_data(self, lag):
        data_name = 'Close_lag' + str(lag)
        self.stock_data[data_name] = self.stock_data['Close'].shift(lag)