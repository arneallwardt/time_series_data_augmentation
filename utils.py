import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler


### DATA VISUALIZATION ###

Y1_COLOR = '#e07a5f'
Y2_COLOR = '#81b29a'

def load_time_series(path):
    '''Loads time series and converts Data column into datetime object'''

    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def plot_time_series_attribute(df: pd.DataFrame, title='Time Series', x='Date', y='Close'):
    '''Plots the closing price of a timeseries'''

    # check if date and Close columns exist
    columns = df.columns.tolist()
    for column in [x, y]:
        if column not in columns:
            raise ValueError(f'Dataframe does not include one of the following column names: {[x, y]}.')

    # Plot time series using matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(df[x], df[y], color=Y1_COLOR)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y[0])
    plt.grid(False)
    plt.show()


def plot_multiple_time_series_attributes(df: pd.DataFrame, title='Time Series', x='Date', y1='Close', y2='Volume'):
    '''Plots closing price and volume of a timeseries'''

    # check if date and Close columns exist
    columns = df.columns.tolist()
    for column in [x, y1, y2]:
        if column not in columns:
            raise ValueError(f'Dataframe does not include the following column: {column}.')
        
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # primary axis
    ax1.plot(df[x], df[y1], Y1_COLOR, zorder=2)
    ax1.set_xlabel(x)
    ax1.set_ylabel(y1, color=Y1_COLOR)

    # secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df[x], df[y2], Y2_COLOR, zorder=1, alpha=0.4)
    ax2.set_ylabel(y2, color=Y2_COLOR)

    fig.tight_layout()
    plt.title(title)
    plt.show()


def slice_years(df: pd.DataFrame, years, index='Date') -> pd.DataFrame:
    '''Slices certain years of a time series and returns the corresponding dataframe'''

    # create new df with Date as index to make slicing possible
    df_sliced = df.set_index(index, inplace=False)

    # check if years are present
    years_in_index = df_sliced.index.year.unique()
    for year in years:
        if int(year) not in years_in_index:
            raise ValueError(f'Years {years} not present in dataframe')

    # slice dataframe to only include specific years
    if len(years) > 1:
        # check if first year smaller than last year
        if int(years[0]) > int(years[1]):
            raise ValueError(f'Years {years} have to be in ascending order')
        df_sliced = df_sliced.loc[years[0]:years[-1]]
    else:
        df_sliced = df_sliced.loc[years[0]]
        
    # reset index to prevent errors later on
    df_sliced.reset_index(inplace=True)

    return df_sliced



### BASELINE MODEL DATA PREPROCESSING ###

def add_lagged_data(df: pd.DataFrame, lag, columns, convert_to_numpy=True):
    '''Returns a dataframe or a numpy array of the dataframe with lagged features, ensuring the shape is (n_samples, num_original_columns * (lag+1), num_original_columns) when converted to numpy array.'''
    lagged_data = [] # list to store lagged data

    df = df.copy()  # make a copy of the dataframe to prevent changes to the original dataframe
    df.set_index('Date', inplace=True)  # set Date as index
    df_columns = df.columns.tolist()

    # check if columns are present and in correct order
    if df_columns != columns:
        raise ValueError(f'Order of given columns does not match the order of the actual columns:\ngiven columns: {columns}\nactual columns: {df_columns}')
    
    print(f'Adding lagged data for columns: {columns}')
    for i in range(1, lag + 1):
        for column in columns:
            # create lagged features
            lagged = df[column].shift(i).rename(f'{column}_Lagged_{i}')
            
            # make sure, that the columns have the same order as in the original dataframe
            lagged_data.append(lagged)

    # Concatenate the lagged data with the original dataframe
    # shape: (n_samples, num_original_columns * (lag+1))
    # e.g. (7000, 16) for 7000 samples, 2 original columns and lag=7
    df_lagged = pd.concat([df] + lagged_data, axis=1)

    # Drop the initial rows which contain NaN values due to the shifting
    df_lagged.dropna(inplace=True)

    if convert_to_numpy:
        # Reshape the dataframe to have the shape (n_samples, num_original_columns * (lag+1), num_original_columns)
        np_array = df_lagged.to_numpy().reshape(df_lagged.shape[0], -1, len(columns))
        print(f'Shape of the numpy array wit lagged data: {np_array.shape}')
        return np_array
    else:
        return df_lagged


def scale_data(np_array):
    '''Scales each feature individually using MinMaxScaler and returns the scaled numpy array aswell as the scaler used for scaling the closing price.'''
    n_features_per_timestep = np_array.shape[-1]
    scalers = []

    # scale each feature individually and save the scalers to inverse scale the data later
    for i in range(n_features_per_timestep):
        scalers.append(MinMaxScaler(feature_range=(0, 1))) 
        np_array[:, :, i] = scalers[i].fit_transform(np_array[:, :, i])

    return np_array, scalers[0]


def inverse_scale_data(np_array, scaler, lag):
    # create dummies to match the required shape of the scaler and set the first column to the array to scale
    dummies = np.zeros((np_array.shape[0], lag+1))
    dummies[:, 0] = np_array.flatten()

    # inverse scale the data
    dummies_scaled = scaler.inverse_transform(dummies)

    # get only first column of the dummies_scaled array, since this is where the original data was
    np_array = dc(dummies_scaled[:, 0])

    print(f'Shape of the inverse scaled numpy array: {np_array.shape}')
    return np_array


def scale_data_same_scaler(np_array, scaler):
    '''CURRENTLY NOT IN USE: Scales features together using the given scaler and returns the scaled numpy array.'''
    n_samples = np_array.shape[0]  
    n_timesteps = np_array.shape[1]

    np_array = np_array.reshape(-1, 2)

    np_array = scaler.fit_transform(np_array)
    
    np_array = np_array.reshape(n_samples, n_timesteps, 2)

    return np_array


def train_test_split_to_tensor(np_array, split_ratio=0.95):
    '''Splits the data into train and test set, flips the column order of the features and converts them to tensors.'''

    X = np_array[:, 1:]
    X = dc(np.flip(X, axis=1)) # flip coloumns to change order from t-1, t-2, ... to t-2, t-1, ...
    X = torch.tensor(X, dtype=torch.float32)
    y = np_array[:, 0, 0] # only take the closing price as target, ignore the other features
    y = torch.tensor(y, dtype=torch.float32)

    split_index = int(len(X) * split_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index].unsqueeze(1)
    y_test = y[split_index:].unsqueeze(1)

    print(f'Shape of X_train: {X_train.shape} \n Shape of y_train: {y_train.shape} \n Shape of X_test: {X_test.shape} \n Shape of y_test: {y_test.shape}')
    return X_train, y_train, X_test, y_test



### GENERAL DATA PREPROCESSING ###

def split_data_into_sequences(data, seq_len):
    '''
    Splits data into sequences of length seq_len.

    Input: 
        - data: 2 dimensional np array in the shape of (n_samples, n_features).
        - seq_len: length of the sequences

    Output:
        - split_data: 3 dimensional np array in the shape of (n_samples-seq_len, seq_len, n_features).
    '''
    
    split_data = []
    for i in range(len(data)-seq_len):
        split_data.append(data[i:i+seq_len])
    return np.array(split_data)