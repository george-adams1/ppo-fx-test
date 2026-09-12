import pandas as pd
import ta
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

class Data:
    def get(self):
        eurusd, gbpusd, usdjpy = self.load()
        eurusd, gbpusd, usdjpy = self.set_time_to_datetime(eurusd), self.set_time_to_datetime(gbpusd), self.set_time_to_datetime(usdjpy)
        eurusd, gbpusd, usdjpy = self.add_technical_indicators(eurusd), self.add_technical_indicators(gbpusd), self.add_technical_indicators(usdjpy)
        eurusd, gbpusd, usdjpy = self.common_rows(eurusd, gbpusd, usdjpy)
        df = self.merge_dataframes(eurusd, gbpusd, usdjpy)
        df = self.replace_nan_values(df, ["timedate","pair"])

        df = df[['pair'] + [col for col in df.columns if col != 'pair']]
        df = df.sort_values(by=['datetime','pair']).reset_index(drop=True)
        return df


    def load(self):
        eurusd = pd.read_csv("fx_data/EURUSD_Candlestick_1_M_BID_20.02.2023-20.05.2023.csv")
        gbpusd = pd.read_csv("fx_data/GBPUSD_Candlestick_1_M_BID_20.02.2023-20.05.2023.csv")
        usdjpy = pd.read_csv("fx_data/USDJPY_Candlestick_1_M_BID_20.02.2023-20.05.2023.csv")
        return eurusd, gbpusd, usdjpy
    
    def set_time_to_datetime(self, df):
        df['datetime'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
        df = df.drop('Gmt time', axis=1)
        df = df[['datetime'] + [col for col in df.columns if col != 'datetime']]
        return df

    def add_technical_indicators(self, df):
        # Add Standard Deviation (STDEV)
        df['stdev'] = ta.volatility.bollinger_mavg(df['Close']).fillna(0)

        # Add Bollinger Bands (BB)
        indicator_bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_upperband'] = indicator_bb.bollinger_hband().fillna(0)
        df['bb_lowerband'] = indicator_bb.bollinger_lband().fillna(0)

        # Add Average Directional Index (ADX)
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close']).fillna(0)

        # Add Moving Average Convergence Divergence (MACD)
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd().fillna(0)
        df['macd_signal'] = macd.macd_signal().fillna(0)
        df['macd_hist'] = macd.macd_diff().fillna(0)

        # Add Relative Strength Index (RSI)
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi().fillna(0)

        # Add Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_oscillator'] = stoch.stoch().fillna(0)
        df['stoch_signal'] = stoch.stoch_signal().fillna(0)

        # Add Relative Strength Index (RSI)
        rsi_indicator = ta.momentum.RSIIndicator(df['Close'])
        df['rsi'] = rsi_indicator.rsi().fillna(0)

        # Add RSI signals
        df['rsi_signal'] = 0  # Initialize all signals as 0
        df.loc[df['rsi'] > 55, 'rsi_signal'] = -1  # Sell signal when RSI is above 55
        df.loc[df['rsi'] < 35, 'rsi_signal'] = 1  # Buy signal when RSI is below 35

        return df

    def common_rows(self, a, b, c):
        # leave only rows with timedate that all 3 dfs have
        common_items = list(set(a.datetime.tolist()) & set(b.datetime.tolist()) & set(c.datetime.tolist()))
        a = a[a.datetime.isin(common_items)]
        b = b[b.datetime.isin(common_items)]
        c = c[c.datetime.isin(common_items)]
        return a, b, c
    
    def merge_dataframes(self, eurusd, gbpusd, usdjpy):
        # Add a new column 'pair' to each data frame
        eurusd['pair'] = 'eurusd'
        gbpusd['pair'] = 'gbpusd'
        usdjpy['pair'] = 'usdjpy'

        # Merge the data frames based on the 'timedate' column
        merged_df = pd.concat([eurusd, gbpusd, usdjpy], ignore_index=True)
        merged_df.sort_values('datetime', inplace=True)

        return merged_df

    def replace_nan_values(self, df: pd.DataFrame, columns_to_skip: list):
        '''
        1) fills empty values by their previouse values
        2) fills all left empty values by 0 and return
        '''
        only_selected_columns = [c for c in df.columns.tolist() if c not in columns_to_skip]

        #1
        r_df = pd.DataFrame()
        for pair in df.pair.unique().tolist():
            pair_df = df[df.pair == pair]
            pair_df[only_selected_columns] = pair_df[only_selected_columns].fillna(method='ffill')
            r_df = r_df.append([pair_df])
        df = r_df

        #2
        df = df.fillna(value=0)
        return df
    

if __name__ == '__main__':
    df = Data().get()
    df.to_csv('fx_data/full_df.csv', index=True)