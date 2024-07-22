import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

def brownian_motion(df: pd.DataFrame, features=['Close']) -> pd.DataFrame:
    '''
    Creates a new time series by applying brownian motion with mean and standard deviation from an existing time series
    NOTE: this is probably not good for augmenting stock data since it is not stationary e.g. the standard deviation and variance change while time passes
    '''
    df = df.copy()
    days = df.shape[0]

    # remove unnecassary columns from original dataframe
    features_incl_date = ['Date'] + features
    df = df[features_incl_date]

    # check if all selected features are present in the dataframe
    columns = df.columns.tolist()
    for feature in features:
        if feature not in columns:
            raise ValueError(f'Dataframe does not include the following feature: {feature}.')
        
    print(f'Augmenting the following columns with brownian motion: {features}...')
    for feature in features:
        # calculate daily log returns and put them in an array
        # df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        # log_returns = df['LogReturns'].dropna()

        # calculate daily gains / losses and use them to calculate mean
        df[f'Diff_{feature}'] = df[feature].diff()
        diff = df[f'Diff_{feature}'].dropna()

        # calculate mean and standard deviation from log returns 
        # mean, standard_deviation = log_returns.mean(), log_returns.std()
        mean, standard_deviation = diff.mean(), diff.std()

        # get normal distributed values which represent the daily change for the specified feature of the augmented time series
        random_values = rng.normal(mean, standard_deviation, days)

        # step size (1 day)
        step = 1

        # create mock array for augmented prices and set first value to first closing price of the time series
        synthetic_values = np.zeros(days)
        synthetic_values[0] = df.loc[0, feature]
        
        # create brownian motion
        for idx in range(days-1):
            # get real index, because we need real_idx - 1 later on 
            real_idx = idx + 1
            # calculate value for current index for every path of the matrix
            synthetic_values[real_idx] = round(synthetic_values[real_idx - 1] + mean * step + standard_deviation * np.sqrt(step) * random_values[idx], 6)

            # Volume coloumn allows only integer values
            if feature == 'Volume':
                synthetic_values[real_idx] = round(synthetic_values[real_idx])

            # no negative values allowed
            if synthetic_values[real_idx] < 0:
                synthetic_values[real_idx] = 0


        # apply augmented prices and drop column which saves the diff
        df[feature] = synthetic_values
        df = df.drop(columns=[f'Diff_{feature}'])   

    return df