import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

def brownian_motion(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Creates a new time series by applying brownian motion with mean and standard deviation from an existing time series
    NOTE: this is probably not good for augmenting stock data since it is not stationary e.g. the standard deviation and variance change while time passes
    '''
    df = df.copy()
    days = df.shape[0]

    # calculate daily log returns and put them in an array
    # df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
    # log_returns = df['LogReturns'].dropna()

    # calculate daily gains / losses and use them to calculate mean
    df['Diff'] = df['Close'].diff()
    diff = df['Diff'].dropna()

    # calculate mean and standard deviation from log returns 
    # mean, standard_deviation = log_returns.mean(), log_returns.std()
    mean, standard_deviation = diff.mean(), diff.std()

    # get normal distributed values which represent the daily gain / loss of the augmented time series
    random_values = rng.normal(mean, standard_deviation, days)

    # step size (1 day)
    step = 1

    # create mock array for augmented prices and set first value to first closing price of the time series
    augmented_prices = np.zeros(days)
    augmented_prices[0] = df.loc[0, 'Close']
    
    # create brownian motion
    for idx in range(days-1):
        # get real index, because we need real_idx - 1 later on 
        real_idx = idx + 1
        # calculate value for current index for every path of the matrix
        augmented_prices[real_idx] = augmented_prices[real_idx - 1] + mean * step + standard_deviation * np.sqrt(step) * random_values[idx]


    # apply augmented prices and drop LogReturns
    df['Close'] = augmented_prices
    df = df.drop(columns=['Diff'])   

    return df