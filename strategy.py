import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

# note that annual needs to convert the interval freq to 1y
class Strategy():
    def __init__(self, symbol, benchmark, start, end, interval, annual = 252, risk_free = 0, position = None):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.benchmark = benchmark
        self.risk_free = risk_free
        self.annual = annual
        self._position_name = None
        self._full = False
        self.data = None
        self.benchmark_data = None
        self.stock_returns_base = None
        self.stock_returns = None # annual
        self.bench_returns_base = None
        self.bench_returns = None # annual
        self.returns_base = None
        self.returns = None # annual
        self.get_data()
        if position:
            self.add_position(position)
        self._beta = None
        self._self_beta = None
        self._alpha = None
        self._self_alpha = None
        self._sharpe = None
        self._drawdown = None
        self._drawdown_ts = None
        self._calmar = None
        self._sortino = None
        self._treynor = None
        self._ir = None

    def get_data(self):
        stock_data = yf.Ticker(self.symbol)
        hist_stock = stock_data.history(start = self.start, end = self.end, interval = self.interval)
        self.data, self.stock_returns_base = self.calculate_close_returns(hist_stock)
        self.stock_returns = self.stock_returns_base * self.annual

        bench_data = yf.Ticker(self.benchmark)
        bench_stock = bench_data.history(start = self.start, end = self.end, interval = self.interval)
        self.bechmark_data, self.bench_returns_base = self.calculate_close_returns(bench_stock)
        self.bench_returns = self.bench_returns_base * self.annual


    # takes in a dataframe and outputs a new dataframe with a log returns named 'Returns'
    # uses the Close price
    # returns df with returns col and returns in the base interval specifed at creation of Strategy
    def calculate_close_returns(self, in_df):
        df = in_df.copy()
        df['Returns'] = np.log(df["Close"] / df["Close"].shift(1))
        base_norm_ret = np.exp(self.df["Returns"].mean()) -1 
        return df, base_norm_ret

    # takes in df and outputs a new df with log returns named "target + _returns"
    # target arg is the column with the position in the stock
    # returns df with returns and returns in the base interval specified at creation of Strategy
    def calculate_returns(self, target):
        df = self.data.copy()
        df[target + '_returns'] = df["Returns"] * df[target].shift(1)
        base_norm_ret = np.exp(df[target + '_returns'].mean()) -1
        return df, base_norm_ret

    # takes a df with only one position col
    # one Strategy object can only have one position column
    # calculates returns as well
    def add_position(self,position_df):
        assert [pos_name not in self.data.columns for pos_name in position_df.columns], 'position name conflict' # no conflicting names
        assert len(position_df) == 1, 'more than one position column given'
        assert self._full == False, 'strategy already has a position, please create another strategy'
        self._full = True
        df = self.data.copy()
        df = df.merge(position_df, how = 'inner', left_index = True)
        self.data = df
        self._position_name = position_df.columns[0]
        self.data, self.returns_base = self.calculate_returns(self, self._position_name)
        self.returns = self.returns_base * self.annual
    #return self.data

    # calculate beta for strategy
    # wrt to benchmark and the stock itself
    def calculate_beta(self):
        assert self._position_name is not None, 'no position for strategy, please provide one via add_position()'
        self._self_beta = self.data[self.position_name + "_returns"].cov(self.data['Returns']) / self.data.var()
        self._beta = self.data[self.position_name + "_returns"].cov(self.benchmark_data['Returns']) / self.benchmark_data.var()

    # getter for beta
    def beta(self):
        if not self._beta:
            self.calculate_beta()
        return self._beta

    # calculate the alpha for strategy
    # wrt to benchmark and the stock itself
    def calculate_alpha(self):
        assert self._position_name is not None, 'no position for strategy, please provide one via add_position()'
        self._alpha = self.returns - self.risk_free - self._beta * (self.bench_returns - self.risk_free)
        self._self_alpha = self.returns - self.risk_free - self._self_beta * (self.stock_returns - self.risk_free)

    # getter for alpha
    def alpha(self):
        if not self._alpha:
            self.calculate_alpha()
        return self._alpha

    # calculate Sharpe Ratio as (returns - risk free) / stdev
    def calculate_sharpe(self):
        assert self._position_name is not None, 'no position for strategy, please provide one via add_position()'
        self._sharpe = (self.returns - self.risk_free) / self.data[self.position_name + "_returns"].stdev()

    # getter for sharpe ratio
    def sharpe(self):
        if not self._sharpe:
            self.calculate_sharpe()
        return self._sharpe


    # calculate drawdown from log returns column
    def calculate_drawdown(self):
        returns = self.data[self.position_name + "_returns"].cumsum().apply(np.exp)
        index = 1 + returns
        previous_peaks = index.cummax()
        drawdowns = (index-previous_peaks)/previous_peaks
        self._drawdown = max(drawdowns)
        self._drawdown_ts = drawdowns
    
    # getter for drawdown
    def drawdown(self):
        if not self._drawdown:
            self.calculate_drawdown()
        return self._drawdown
    
    # calmar ratio = R/drawdown
    def calculate_calmar(self):
        drawdown = self.drawdow()
        self._calmar = (self.stock_returns - self.risk_free) / drawdown
    
    # getter for calmar
    def calmar(self):
        if not self._calmar:
            self.calculate_calmar()
        return self._calmar
    
    # sortino = (r - risk_free) / neg_std_dev
    def calculate_sortino(self):
        self._sortino = (self.returns - self.risk_free) / self.data[self.position_name + "_returns"][self.data[self.position_name + "_returns"] < 0].stdev()
    
    # getter for sortino
    def sortino(self):
        if not self._sortino:
            self.calculate_sortino()
        return self._sortino

    # treynor ratio = R/Beta
    def calculate_treynor(self):
        self._treynor = self.returns/self._beta
    
    # getter for treynor
    def treynor(self):
        if not self._treynor:
            self.calculate_treynor()
        return self._treynor
    
    # information ratio = (returns - benchmark_returns) / std(returns - benchmark_returns)
    def calculate_information_ratio(self):
        rp = self.data[self.position_name + "_returns"] / self.data[self.position_name + "_returns"].shift(1)
        rb = self.benchmark_data["Close"] / self.benchmark_data["Close"].shift(1)
        diff = (rp -rb).sum()
        stdev = (rp-rb).stdev()
        self._ir = diff/stdev
    
    # getter for information_ratio
    def information_ratio(self):
        if not self._ir:
            self.calculate_information_ratio()
        return self._ir
        

# line_a and line_b must be time series (can use ta package to create)
# line_a crosses line_b from below to create a long signal
# line_a crosses line_b from above to create a short signal
def Cross_Strategy(Strategy):
    def __init__(self, symbol, benchmark, start, end, interval, line_a, line_b,  annual = 252, risk_free = 0):
        super(Cross_Strategy, self).__init__(symbol, benchmark, start, end, interval, annual, risk_free)
        self.line_a = line_a
        self.line_b = line_b
        self.add_position()
    

    # create crossover positions / signals here 
    def create_crossover_position(self):
        df = pd.DataFrame()
        df['Position'] = np.where(self.line_a > self.line_b, 1, -1)
        df['Signal'] = df['Position'].diff()
        self.add_position(df['Position'])






  


  
    
  
