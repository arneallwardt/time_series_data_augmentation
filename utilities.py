import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from typing import Dict


### DATA VISUALIZATION ###

Y1_COLOR = '#e07a5f'
Y2_COLOR = '#81b29a'

def load_complete_time_series(path):
    '''
    Loads time series and converts Data column into datetime object

    Args:
        - path: str, path to the csv file

    Returns:
        - df: pd.DataFrame, dataframe containing the time series data with shape (n_samples, n_features)
    '''

    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    return df

def load_sequential_time_series(path, shape=None):
    '''
    Loads sequential time series data from a csv file and reshapes it to the given shape.

    Args:
        - path: str, path to the csv file
        - shape: tuple, shape of the np array to be returned in the form of (n_samples, seq_len, n_features)

    Returns:
        - loaded_generated_data: np.array, array containing the time series data with shape (n_samples, seq_len, n_features)
    '''

    loaded_generated_data = np.loadtxt(path, delimiter=',')

    if shape:
        no, seq_len, dim = shape
        return loaded_generated_data.reshape(no, seq_len, dim)
    
    return loaded_generated_data


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



### GENERAL DATA PREPROCESSING ###

def train_test_split(data, split_ratio=0.8):
    '''
    Splits the data into train and test set.
    
    Args:
        - data: np.array, data to split with shape (n_samples, seq_len, n_features) OR (n_samples, n_features)
        - split_ratio: float, ratio to split the data into train and test set

    Returns:
        - train_data, test_data: np.arrays of the splitted data
    '''

    if not isinstance(data, np.ndarray):
        raise ValueError('Data is not a numpy array.')
    
    data_dc = dc(data)

    split_index = int(data_dc.shape[0] * split_ratio)

    train_data = data_dc[:split_index]
    test_data = data_dc[split_index:]

    return train_data, test_data


def extract_features_and_targets(train_data, test_data=None, val_data=None):
    '''
    Extracts the features and target from the given data.

    Args:
        - train_data: np.array, data of the training set with shape (n_samples, seq_len, n_features)
        - test_data: np.array, data of the test set with shape (n_samples, seq_len, n_features)

    Returns:
        - X_train, y_train, (X_test, y_test):  torch.tensor, features and targets
    '''

    X_train = torch.tensor(train_data[:, :-1, 1:], dtype=torch.float32)
    y_train = torch.tensor(train_data[:, -1, 0], dtype=torch.float32).reshape(-1, 1)

    # return training data
    if test_data is None:
        print(f'Extracted features and target from training data.\nShape of X_train: {X_train.shape}\nShape of y_train: {y_train.shape}')
        return X_train, y_train

    X_test = torch.tensor(test_data[:, :-1, 1:], dtype=torch.float32)
    y_test = torch.tensor(test_data[:, -1, 0], dtype=torch.float32).reshape(-1, 1)

    # return training and test data
    if val_data is None: 
        print(f'Extracted features and target from training and test data.\nShape of X_train: {X_train.shape}\nShape of y_train: {y_train.shape}\nShape of X_test: {X_test.shape}\nShape of y_test: {y_test.shape}')
        return X_train, y_train, X_test, y_test
    
    X_val = torch.tensor(val_data[:, :-1, 1:], dtype=torch.float32)
    y_val = torch.tensor(val_data[:, -1, 0], dtype=torch.float32).reshape(-1, 1)

    # return training, test and validation data
    return X_train, y_train, X_test, y_test, X_val, y_val


def split_data_into_sequences(data, seq_len, shuffle_data=False):
    '''
    Splits data into sequences of length seq_len.

    Input: 
        - data: 2 dimensional pd dataframe / np array in the shape of (n_samples, n_features).
        - seq_len: length of the sequences

    Output:
        - split_data: 3 dimensional np array in the shape of (n_samples-seq_len, seq_len, n_features).
            - sequences in DESCENDING order (earlier dates first) 
            -> 01.01.2021, 02.01.2021, 03.01.2021, 04.01.2021, 05.01.2021, ...
    '''

    if not isinstance(data, np.ndarray):
        raise ValueError('Data is not a numpy array.')

    dc_data = dc(data)
    
    split_data = []
    for i in range(len(dc_data)-seq_len+1): # +1 to include the last possible element
        split_data.append(dc_data[i:i+seq_len])

    if shuffle_data:
        split_data = np.array(split_data)
        np.random.shuffle(split_data)

    print(f'Shape of the data after splitting into sequences: {np.array(split_data).shape}')
    return np.array(split_data)


def reconstruct_sequential_data(data):
    '''
    Reconstructs the original data from sequences of data.

    Input: 
        - data: 3 dimensional np array in the shape of (n_samples, seq_len, n_features).

    Output:
        - reconstructed_data: 2 dimensional np array in the shape of (n_samples, n_features).
    '''

    return data[:, -1, :]


### Metrics ###
def accuracy(y_true, y_pred):
    
    print(f'y_pred: {y_pred}')
    print(f'y_true: {y_true}')

    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


### Scaler Class ###
class Scaler:
    '''
    Class for scaling and inverse scaling data using MinMaxScaler. 
    NOTE: The first feature of the data has to be the closing price and the last feature has to be the volume.
    '''

    def __init__(self, data: np.array):
        self.data_is_split = data.ndim == 3

        # NOTE: MinMaxScaler scales data individually for each column
        # -> [[-1, 2], [-0.5, 6], [0, 10], [1, 18]] -> [[0, 0], [0.25, 0.25], [0.5, 0.5], [1, 1]]
        self.close_scaler = MinMaxScaler(feature_range=(0, 1))
        self.volume_scaler = MinMaxScaler(feature_range=(0, 1))
        self.returns_scaler = MinMaxScaler(feature_range=(-1, 1))

        self.__fit_data(data)


    def __fit_data(self, data):

        if not isinstance(data, np.ndarray):
            raise ValueError('Data is not a numpy array.')

        dc_data = dc(data)

        close, volume, returns = self.__get_individual_features(dc_data)

        self.close_scaler.fit(close)
        self.volume_scaler.fit(volume)
        self.returns_scaler.fit(returns)


    def scale_data(self, data):
        '''
        Scales data using MinMaxScaler and returns the scaled numpy array aswell as the scaler used for scaling the price features.

        Args:
            - data: numpy array of shape (no_samples, no_features).

        Returns:
            - np_array: scaled numpy array of shape (no_samples, no_features).
        '''

        if not isinstance(data, np.ndarray):
            raise ValueError('Data is not a numpy array.')

        dc_data = dc(data)
        close, volume, returns = self.__get_individual_features(dc_data)

        close_scaled = self.close_scaler.transform(close)
        volume_scaled = self.volume_scaler.transform(volume)
        returns_scaled = self.returns_scaler.transform(returns)

        scaled_data = self.__reconstruct_original_shape(dc_data, close_scaled, volume_scaled, returns_scaled)

        return scaled_data


    def __get_individual_features(self, data):
        '''Returns the closing prices and volume of the given numpy array.'''

        if self.data_is_split:
            close = data[:, :, 1].reshape(-1, 1) 
            volume = data[:, :, 2].reshape(-1, 1)
            returns = data[:, :, -1].reshape(-1, 1)
        else:
            close = data[:, 1].reshape(-1, 1) 
            volume = data[:, 2].reshape(-1, 1)
            returns = data[:, -1].reshape(-1, 1)

        return close, volume, returns


    def __reconstruct_original_shape(self, original_data, close_scaled, volume_scaled, returns_scaled):
        '''Reconstructs the original numpy array with the scaled prices and volume.'''

        if self.data_is_split:
            original_data[:, :, 1] = close_scaled.reshape(original_data[:, :, 1].shape)
            original_data[:, :, 2] = volume_scaled.reshape(original_data[:, :, 2].shape)
            original_data[:, :, -1] = returns_scaled.reshape(original_data[:, :, -1].shape)
        else:
            original_data[:, 1] = close_scaled.reshape(original_data[:, 1].shape)
            original_data[:, 2] = volume_scaled.reshape(original_data[:, 2].shape)
            original_data[:, -1] = returns_scaled.reshape(original_data[:, -1].shape)

        return original_data


    def inverse_scale_data(self, data, feature_type):
        '''
        Inverse scales the data using the given scaler and returns the inverse scaled numpy array.
        
        Args:
            - data: numpy array of shape (no_samples, no_features).
            - feature_type: str, either 'close', 'volume', 'returns

        Returns:
            - scaled_data: numpy array of shape (no_samples, 1).
        '''

        if not isinstance(data, np.ndarray):
            raise ValueError('Data is not a numpy array.')

        # create dummies to match the required shape of the scaler and set the first column to the array to scale
        dummies = np.zeros((data.shape[0], 1))
        dummies[:, 0] = data.flatten()

        # inverse scale the data
        if feature_type == 'close':
            dummies_scaled = self.close_scaler.inverse_transform(dummies)
        elif feature_type == 'volume':
            dummies_scaled = self.volume_scaler.inverse_transform(dummies)
        elif feature_type == 'returns':
            dummies_scaled = self.returns_scaler.inverse_transform(dummies)
        else:
            raise ValueError('Invalid feature type. Choose either "close", "volume" or "returns".')

        # get only first column of the dummies_scaled array, since this is where the original data was
        scaled_data = dummies_scaled[:, 0]

        return scaled_data
    
class ValidationLossAccumulationCallback(xgb.callback.TrainingCallback):
    def __init__(self, losses) -> None:
        self.losses = losses

    def after_iteration(
    self, model, epoch: int, evals_log: Dict[str, dict]
    ) -> bool:
        """Accumulate the mae after each iteration."""
        self.losses.append(evals_log['validation_0']['mean_absolute_error'][-1])
        return False