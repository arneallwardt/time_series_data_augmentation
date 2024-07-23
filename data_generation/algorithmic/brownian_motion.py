import pandas as pd
import numpy as np
from copy import deepcopy as dc

rng = np.random.default_rng(42)

def brownian_motion(data: pd.DataFrame, features=['Close'], other_columns=[]) -> pd.DataFrame:
    '''
    Creates a new time series by applying brownian motion with mean and standard deviation from an existing time series
    Categorical columns are shuffled.
    '''

    dc_data = dc(data)
    days = dc_data.shape[0]

    # check if all selected features are present in the dataframe
    columns = dc_data.columns.tolist()
    for col in features + other_columns:
        if col not in columns:
            raise ValueError(f'Dataframe does not include the following feature: {col}.')
        

    print(f'Augmenting the following columns with brownian motion: {features}...')
    for feature in features:
        
        # calculate daily gains / losses and use them to calculate mean
        dc_data[f'Diff_{feature}'] = dc_data[feature].diff()
        diff = dc_data[f'Diff_{feature}'].dropna()

        # calculate mean and standard deviation
        mean, standard_deviation = 0, diff.std()

        # get normal distributed values which represent the daily change for the specified feature of the augmented time series
        random_values = rng.normal(mean, standard_deviation, days)

        # step size (1 day)
        step = 1

        # create mock array for augmented prices and set first value to first closing price of the time series
        synthetic_values = np.zeros(days)
        synthetic_values[0] = dc_data.loc[0, feature]
        
        # create brownian motion
        for idx in range(days-1):
            # get real index, because we need real_idx - 1 later on 
            real_idx = idx + 1

            # calculate value for current index for every path of the matrix
            # including drift term
            # synthetic_values[real_idx] = synthetic_values[real_idx - 1] + mean * step + standard_deviation * np.sqrt(step) * random_values[idx], 6
            # without drift term
            synthetic_values[real_idx] = synthetic_values[real_idx - 1] + np.sqrt(step) * random_values[idx]

            # Volume coloumn allows only integer values
            if feature in ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']:
                # no negative values allowed
                if synthetic_values[real_idx] < 0:
                    synthetic_values[real_idx] = 0

            elif feature in ['clouds_all']:
                if synthetic_values[real_idx] > 100:
                    synthetic_values[real_idx] = 100


        # apply augmented prices and drop column which saves the diff
        dc_data[feature] = synthetic_values
        dc_data = dc_data.drop(columns=[f'Diff_{feature}'])   

    
    # shuffle values of columns which cant be augmented via brownian motion since they are categorical
    for col in other_columns:
        dc_data[col] = dc_data[col].sample(frac=1).values
        

    return dc_data