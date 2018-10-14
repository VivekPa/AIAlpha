import pandas_datareader.data as pdr
import fix_yahoo_finance as fix
fix.pdr_override()


class GetData:
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end

    # get stock data
    def get_stock_data(self):
        stock_data = pdr.get_data_yahoo(self.ticker, self.start, self.end)
        stock_data.to_csv("stock_data.csv")

    # get twitter data
    # do your code here!

    # get news data
    # do your code here!


if __name__ == "__main__":
    data = GetData("AAPL", "2000-01-01", "2018-10-01")
    data.get_stock_data()
