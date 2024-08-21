import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy as dc
from torch.utils.data import DataLoader
from data_evaluation.predictive.LSTM import LSTMRegression, train_model
from utilities import Scaler, train_test_split, extract_features_and_targets_reg, split_data_into_sequences
from data_evaluation.predictive.TimeSeriesDataset import TimeSeriesDataset

def predictive_evaluation(data_real: np.array, data_syn: np.array, hyperparameters, include_baseline=False, verbose=True):

    data_real_dc = dc(data_real)
    data_syn_dc = dc(data_syn)
    results = pd.DataFrame(columns=['Model', 'Metric', 'Error'])

    print('HYPERPARAMETERS:')
    for key, value in hyperparameters.items():
        print(key, ': ', value)

    data_syn_is_sequential = data_syn_dc.ndim == 3
    print('Synthetic Data is sequential:', data_syn_is_sequential)

    if include_baseline:

        ### Baseline ###
        baseline_train_data, baseline_test_data = train_test_split(data_real_dc, split_ratio=0.8) # split real data into train and test
        baseline_data, baseline_scaler = get_distinct_data(train_data=baseline_train_data, test_data=baseline_test_data,
                                                        evaluation_method='baseline',
                                                        syn_data_is_sequential=data_syn_is_sequential,
                                                        hyperparameters=hyperparameters)

        results = run_model(data=baseline_data, scaler=baseline_scaler,
                            evaluation_method='baseline',
                            hyperparameters=hyperparameters,
                            results=results,
                            verbose=verbose)


    ### TRTS ###
    TRTS_data, TRTS_scaler = get_distinct_data(train_data=data_real_dc, test_data=data_syn_dc,
                                            evaluation_method='TRTS',
                                            syn_data_is_sequential=data_syn_is_sequential,
                                            hyperparameters=hyperparameters)
    
    results = run_model(data=TRTS_data, scaler=TRTS_scaler,
                        evaluation_method='TRTS',
                        hyperparameters=hyperparameters,
                        results=results, 
                        verbose=verbose)
    
    
    ### TSTR ###
    TSTR_data, TSTR_scaler = get_distinct_data(train_data=data_syn_dc, test_data=data_real_dc,
                                            evaluation_method='TSTR',
                                            syn_data_is_sequential=data_syn_is_sequential,
                                            hyperparameters=hyperparameters)
    
    results = run_model(data=TSTR_data, scaler=TSTR_scaler,
                        evaluation_method='TSTR',
                        hyperparameters=hyperparameters,
                        results=results, 
                        verbose=verbose)
    

    ### Combined ###
    combined_data, combined_scaler = get_combined_data(real_data=data_real_dc, syn_data=data_syn_dc,
                                            syn_data_is_sequential=data_syn_is_sequential,  
                                            hyperparameters=hyperparameters)
    
    results = run_model(data=combined_data, scaler=combined_scaler,
                        evaluation_method='combined',
                        hyperparameters=hyperparameters,
                        results=results, 
                        verbose=verbose)
        

    return results


def get_combined_data(real_data, syn_data, syn_data_is_sequential, hyperparameters):
    # split real data (one part for train and val and one part for test)
    real_train, real_test = train_test_split(real_data, split_ratio=0.8)
    real_test, real_val = train_test_split(real_test, split_ratio=0.5)

    # split synthetic data (only train and val)
    syn_train, syn_val = train_test_split(syn_data, split_ratio=0.9)
    # syn_train = syn_train[:int(len(syn_train)*0.75)] # limit size of synthetic data to half

    ### Scale data
    # create temporary array to fit scaler
    # reason is, we want to scale based on real and syn data, but we can't combine real and syn data into one sequence
    temp_train = np.concatenate((real_train, syn_train.reshape(-1, syn_train.shape[-1])), axis=0)

    # create scaler
    scaler = Scaler(temp_train)

    # scale data
    real_train_scaled = scaler.scale_data(real_train)
    real_val_scaled = scaler.scale_data(real_val)
    real_test_scaled = scaler.scale_data(real_test)

    syn_train_scaled = scaler.scale_data(syn_train, input_data_is_sequential=syn_data_is_sequential)
    syn_val_scaled = scaler.scale_data(syn_val, input_data_is_sequential=syn_data_is_sequential)


    # split data into sequences
    real_train_seq_scaled = split_data_into_sequences(real_train_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)
    real_val_seq_scaled = split_data_into_sequences(real_val_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)
    real_test_seq_scaled = split_data_into_sequences(real_test_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)

    if syn_data_is_sequential:
        syn_train_seq_scaled = syn_train_scaled
        syn_val_seq_scaled = syn_val_scaled
    else:
        syn_train_seq_scaled = split_data_into_sequences(syn_train_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)
        syn_val_seq_scaled = split_data_into_sequences(syn_val_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)

    
    # combine real and syn data
    train_seq_scaled = np.concatenate((real_train_seq_scaled, syn_train_seq_scaled), axis=0)
    val_seq_scaled = np.concatenate((real_val_seq_scaled, syn_val_seq_scaled), axis=0)
    test_seq_scaled = real_test_seq_scaled


    # extract features and targets
    features_and_targets = extract_features_and_targets_reg(train_seq_scaled, test_seq_scaled, val_seq_scaled)

    return features_and_targets, scaler


def get_distinct_data(train_data, test_data, evaluation_method, syn_data_is_sequential, hyperparameters):
    
    ### Data Preprocessing

    # train and test split is already done, just add val data
    if evaluation_method == 'baseline':
        train = train_data
        test, val = train_test_split(test_data, split_ratio=0.5)
    else:
        # split data into train, test, val
        train, val = train_test_split(train_data, split_ratio=0.8)
        test = test_data


    # scale data
    scaler = Scaler(train)
    train_scaled = scaler.scale_data(train)
    val_scaled = scaler.scale_data(val)
    test_scaled = scaler.scale_data(test, input_data_is_sequential = test.ndim == 3)


    # no need to be aware of sequential data as real data is always non-sequential
    if evaluation_method == 'baseline':
        train_seq_scaled = split_data_into_sequences(train_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)
        val_seq_scaled = split_data_into_sequences(val_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)
        test_seq_scaled = split_data_into_sequences(test_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)


    # split test data BUT split TRAINING data only if synthetic data is not sequential
    elif evaluation_method == 'TSTR':

        # Train and val data
        if syn_data_is_sequential:
            train_seq_scaled = train_scaled
            val_seq_scaled = val_scaled
        else:
            train_seq_scaled = split_data_into_sequences(train_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)
            val_seq_scaled = split_data_into_sequences(val_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)

        # Test data
        test_seq_scaled = split_data_into_sequences(test_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)


    # split train data BUT split TEST data only if synthetic data is not sequential
    elif evaluation_method == 'TRTS':

        # Train and val data
        train_seq_scaled = split_data_into_sequences(train_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)
        val_seq_scaled = split_data_into_sequences(val_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)

        # Test data
        if syn_data_is_sequential:
            test_seq_scaled = test_scaled # keep data as is
        else:
            test_seq_scaled = split_data_into_sequences(test_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)


    # extract features and targets
    features_and_targets = extract_features_and_targets_reg(train_seq_scaled, test_seq_scaled, val_seq_scaled)

    return features_and_targets, scaler


def run_model(data, scaler, evaluation_method, hyperparameters, results, verbose):
    # unpack data
    X_train, y_train, X_test, y_test, X_val, y_val = data

    if len(X_test) > len(X_val):
        # Limit test data to same size as validation data (for TSTR, TRTS)
        # as we are using the entire synthetic data as test set
        no_samples = min([len(X_val), len(X_test)])
        sampling_indices = np.random.permutation(len(X_test))[:no_samples]
        X_test = X_test[sampling_indices]
        y_test = y_test[sampling_indices]

    # Create Datasets and Dataloader
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)


    ### criterion

    criterion_MSE = nn.MSELoss()
    criterion_MAE = nn.L1Loss()

    for num_run in tqdm(range(hyperparameters["num_evaluation_runs"])):

        model_path = f'lstm_{evaluation_method}_{num_run+1}.pth'

        ### Training ###
        # get model and optimizer
        model = LSTMRegression(
            device=hyperparameters["device"],
            input_size=X_train.shape[-1],
            hidden_size=hyperparameters["hidden_size"],
            num_stacked_layers=hyperparameters["num_layers"],
            bidirectional=hyperparameters["bidirectional"],
        ).to(hyperparameters["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"])
        
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion_MSE,
            optimizer=optimizer,
            device=hyperparameters["device"],
            num_epochs=hyperparameters["num_epochs"],
            verbose=verbose,
            save_path=model_path, 
        )

        ### Evaluation ###

        test_model = LSTMRegression(
            device=hyperparameters["device"],
            input_size=X_train.shape[-1],
            hidden_size=hyperparameters["hidden_size"],
            num_stacked_layers=hyperparameters["num_layers"],
            bidirectional=hyperparameters["bidirectional"],
        ).to(hyperparameters["device"])

        test_model.load_state_dict(torch.load(model_path))

        # evaluate model on test data and save results
        with torch.inference_mode():
            test_model.eval()
            preds = test_model(X_test.to(hyperparameters["device"]))

            # inverse scale data to get real values for error calculation
            preds_unscaled = torch.tensor(scaler.inverse_scale_target(preds.cpu().numpy().reshape(-1, 1)))
            y_test_unscaled = torch.tensor(scaler.inverse_scale_target(y_test.cpu().numpy().reshape(-1, 1)))

            mae = criterion_MAE(preds_unscaled, y_test_unscaled).item()
            mse = criterion_MSE(preds_unscaled, y_test_unscaled).item()

            results = pd.concat([results, pd.DataFrame([{'Model': evaluation_method, 'Metric': 'MAE', 'Error': mae}])], ignore_index=True)
            results = pd.concat([results, pd.DataFrame([{'Model': evaluation_method, 'Metric': 'MSE', 'Error': mse}])], ignore_index=True)

        # remove saved model
        if os.path.isfile(model_path):
            os.remove(model_path)

    return results