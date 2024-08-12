import torch
import numpy as np
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import xgboost as xgb
from typing import Dict
import pandas as pd
import os

### DATA LOADING ###
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


def extract_features_and_targets_clas(train_data, test_data=None, val_data=None):
    '''
    Extracts the features and target from the given data.

    Args:
        - train_data: np.array, data of the training set with shape (n_samples, seq_len, n_features)
        - test_data: np.array, data of the test set with shape (n_samples, seq_len, n_features)

    Returns:
        - X_train, y_train, (X_test, y_test, X_val, y_val):  torch.tensor, features and targets
    '''

    X_train = train_data[:, :-1, :]
    y_train = train_data[:, -1, 0].reshape(-1, 1)

    # return training data
    if test_data is None:
        print(f'Extracted features and target from training data.\nShape of X_train: {X_train.shape}\nShape of y_train: {y_train.shape}')
        return X_train, y_train

    X_test = test_data[:, :-1, :]
    y_test = test_data[:, -1, 0].reshape(-1, 1)

    # return training and test data
    if val_data is None: 
        print(f'Extracted features and target from training and test data.\nShape of X_train: {X_train.shape}\nShape of y_train: {y_train.shape}\nShape of X_test: {X_test.shape}\nShape of y_test: {y_test.shape}')
        return X_train, y_train, X_test, y_test
    
    X_val = val_data[:, :-1, :]
    y_val = val_data[:, -1, 0].reshape(-1, 1)

    # return training, test and validation data
    return X_train, y_train, X_test, y_test, X_val, y_val


def extract_features_and_targets_reg(train_data, test_data=None, val_data=None):
    '''
    Extracts the features and target from the given data.

    Args:
        - train_data: np.array, data of the training set with shape (n_samples, seq_len, n_features)
        - test_data: np.array, data of the test set with shape (n_samples, seq_len, n_features)

    Returns:
        - X_train, y_train, (X_test, y_test, X_val, y_val):  torch.tensor, features and targets
    '''

    X_train = torch.tensor(train_data[:, :-1, :], dtype=torch.float32)
    y_train = torch.tensor(train_data[:, -1, 0], dtype=torch.float32).reshape(-1, 1)

    # return training data
    if test_data is None:
        print(f'Extracted features and target from training data.\nShape of X_train: {X_train.shape}\nShape of y_train: {y_train.shape}')
        return X_train, y_train

    X_test = torch.tensor(test_data[:, :-1, :], dtype=torch.float32)
    y_test = torch.tensor(test_data[:, -1, 0], dtype=torch.float32).reshape(-1, 1)

    # return training and test data
    if val_data is None: 
        print(f'Extracted features and target from training and test data.\nShape of X_train: {X_train.shape}\nShape of y_train: {y_train.shape}\nShape of X_test: {X_test.shape}\nShape of y_test: {y_test.shape}')
        return X_train, y_train, X_test, y_test
    
    X_val = torch.tensor(val_data[:, :-1, :], dtype=torch.float32)
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
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


def get_discriminative_test_performance(model, device, test_data, scaler, method, results):

    X_test, y_test = extract_features_and_targets_clas(test_data)
    X_test_scaled = scaler.scale_data(X_test)
    
    X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    with torch.inference_mode(): 
        test_logits = model(X_test_scaled.to(device)) # get plain model output (logits)
        test_probs = torch.sigmoid(test_logits) # get probabilities
        test_preds = torch.round(test_probs) # get classes
        test_preds = test_preds.clone().detach()

        test_acc = accuracy(y_true=y_test, y_pred=test_preds)
        print(test_acc)
        
        results = pd.concat([results, pd.DataFrame([{'Method': method, 'Accuracy': test_acc}])], ignore_index=True)

    return results


### Scaler Class ###
class Scaler:
    '''
    Class for scaling and inverse scaling data using MinMaxScaler. 
    NOTE: The first feature of the data has to be the closing price and the last feature has to be the volume.
    '''

    def __init__(self, data: np.array, no_features_to_scale=None):
        self.no_features_to_scale = no_features_to_scale if no_features_to_scale else data.shape[-1]
        self.data_is_sequential = data.ndim == 3

        self.universal_scaler = MinMaxScaler(feature_range=(0, 1))
        # self.universal_scaler = MaxAbsScaler()

        self.__fit_data(data)


    def __fit_data(self, data):

        if not isinstance(data, np.ndarray):
            raise ValueError('Data is not a numpy array.')

        dc_data = dc(data)

        # reshape to put the same features in 1 column
        if self.data_is_sequential:
            dc_data = dc_data.reshape(-1, dc_data.shape[-1])
        
        # only scale selected first n features
        self.universal_scaler.fit(dc_data[:, :self.no_features_to_scale])


    def scale_data(self, data, input_data_is_sequential=False):
        '''
        Scales data using specified scaler and returns the scaled numpy array aswell as the scaler used for scaling the price features.

        Args:
            - data: numpy array of shape (no_samples, no_features).
            - input_data_is_split: bool, whether the input data is split into sequences or not.
            NOTE: bool ist needed since we wnat to scale sequential data with a scaler fitted on ori data in case of TRTS

        Returns:
            - np_array: scaled numpy array of shape (no_samples, no_features).
        '''

        if not isinstance(data, np.ndarray):
            raise ValueError('Data is not a numpy array.')

        dc_data = dc(data)

        # reshape to put the same features in 1 column
        if self.data_is_sequential or input_data_is_sequential: 
            dc_data = dc_data.reshape(-1, dc_data.shape[-1])

        # only scale selected first n features
        scaled_data = self.universal_scaler.transform(dc_data[:, :self.no_features_to_scale]) # scale
        dc_data[:, :self.no_features_to_scale] = scaled_data # replace old data in original array
        scaled_data = dc_data # assign to new variable to return later

        # reshape into original shape
        if self.data_is_sequential or input_data_is_sequential:
            scaled_data = scaled_data.reshape(data.shape)

        return scaled_data


    def inverse_scale_target(self, data):
        '''
        Inverse scales the data using the given scaler and returns the inverse scaled numpy array.
        
        Args:
            - data: numpy array of shape (no_samples, no_features). Remember to watch out for the right order of features!

        Returns:
            - scaled_data: numpy array of shape (no_samples, 1).
        '''

        if not isinstance(data, np.ndarray):
            raise ValueError('Data is not a numpy array.')
        
        # create dummies to match the required shape of the scaler and set the first column to the array to scale
        dummies = np.zeros((data.shape[0], self.no_features_to_scale))
        dummies[:, :data.shape[-1]] = data
        dummies_scaled = self.universal_scaler.inverse_transform(dummies)
        scaled_data = dummies_scaled[:, :data.shape[-1]]

        return scaled_data
    

    def inverse_scale_complete_dataset(self, data, input_data_is_sequential):
        '''
        Inverse scales the data using the given scaler and returns the inverse scaled numpy array.
        
        Args:
            - data: numpy array of shape (no_samples, seq_len, no_features). Remember to watch out for the right order of features!
            - data_is_split: bool, whether the data is split into sequences or not. 
            NOTE: We're not using the instance attribute self.data_is_split, since we want to scale sequential data that has been scaled in original shape

        Returns:
            - scaled_data: numpy array of shape (no_samples, seq_len, no_features).
        '''

        if not isinstance(data, np.ndarray):
            raise ValueError('Data is not a numpy array.')

        dc_data = dc(data)

        # reshape to put the same features in 1 column
        if input_data_is_sequential:
            dc_data = dc_data.reshape(-1, data.shape[-1])
        
        # only scale selected first n features
        scaled_data = self.universal_scaler.inverse_transform(dc_data[:, :self.no_features_to_scale]) # scale
        dc_data[:, :self.no_features_to_scale] = scaled_data # replace old data in original array
        scaled_data = dc_data # assign to new variable to return later

        # reshape into original shape
        if input_data_is_sequential:
            scaled_data = scaled_data.reshape(data.shape)

        return scaled_data
    

### inverse scaling of generated data ###

def save_unscaled_sequential_data(ori_data_path, scaled_data_path, scaled_data_shape, no_features_to_inverse_scale):
    '''
    Inverse scales the generated data using the scaler of the original data and saves the unscaled data to a new csv file.

    Args:
        - ori_data_path: str, path to the original data
        - scaled_data_path: str, path to the scaled data
        - scaled_data_shape: tuple, shape of the scaled data in the form of (n_samples, seq_len, n_features)
        - no_features_to_inverse_scale: int, number of features to inverse scale
    '''

    # load ori and data to scale
    ori_data = pd.read_csv(ori_data_path).to_numpy()
    scaled_seq_data = load_sequential_time_series(scaled_data_path, shape=scaled_data_shape)

    inverse_scaler = Scaler(ori_data, no_features_to_scale=no_features_to_inverse_scale) # fit scaler on ori data since synthetic data was generated based on scaled ori data

    inverse_scaled_data = inverse_scaler.inverse_scale_complete_dataset(scaled_seq_data, input_data_is_sequential=True) # rescale syn data

    # reshape and save data
    no, seq, dim = inverse_scaled_data.shape
    inverse_scaled_data_reshaped = inverse_scaled_data.reshape(no, seq*dim)
    np.savetxt(f'{os.path.splitext(scaled_data_path)[0]}_unscaled.csv', inverse_scaled_data_reshaped, delimiter=',')
    

class EvaluationDataset():
    def __init__(self, type, data_path, predictive_results_path, data_shape=(3000, 13, 5)):
        self.type = type
        self.discriminative_data = load_sequential_time_series(data_path, data_shape)
        self.syn_data = self.discriminative_data[:, :-1, :]
        self.predictive_results = pd.read_csv(predictive_results_path)
        
        self.pca_results = None
        self.tsne_results = None

    def get_specific_results(self, metric, model=None):

        if model:
            filtered_df = self.predictive_results[(self.predictive_results['Metric'] == metric) & (self.predictive_results['Model'] == model)]
            filtered_df.loc[:, 'Model'] = filtered_df['Model'].replace(model, f'{model}-{self.type}')
        else:
            filtered_df = self.predictive_results[(self.predictive_results['Metric'] == metric)]

        return filtered_df
    
    def get_baseline_results(self, metric):
        filtered_df = self.predictive_results[(self.predictive_results['Metric'] == metric) & (self.predictive_results['Model'] == 'baseline')]
        return filtered_df



class ValidationLossAccumulationCallback(xgb.callback.TrainingCallback):
    def __init__(self, losses) -> None:
        self.losses = losses

    def after_iteration(
    self, model, epoch: int, evals_log: Dict[str, dict]
    ) -> bool:
        """Accumulate the mae after each iteration."""
        self.losses.append(evals_log['validation_0']['logloss'][-1])
        return False