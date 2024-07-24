import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from baseline_model.LSTM import LSTMRegression, train_model
from utilities import Scaler, train_test_split, extract_features_and_targets_reg, split_data_into_sequences
from baseline_model.TimeSeriesDataset import TimeSeriesDataset

def predictive_evaluation(data_real: np.array, data_syn: np.array, hyperparameters, verbose=True):

    results = pd.DataFrame(columns=['Model', 'Metric', 'Error'])

    print('HYPERPARAMETERS:')
    for key, value in hyperparameters.items():
        print(key, ': ', value)

    data_syn_is_sequential = data_syn.ndim == 3
    print('Synthetic Data is sequential:', data_syn_is_sequential)

    # Evaluate
    results = run_evaluation(train_data=data_real, test_data=data_syn,
                evaluation_method='TRTS',
                data_syn_is_sequential=data_syn_is_sequential,
                hyperparameters=hyperparameters,
                results=results,
                verbose=verbose)
    
    results = run_evaluation(train_data=data_syn, test_data=data_real,
                evaluation_method='TSTR',
                data_syn_is_sequential=data_syn_is_sequential,
                hyperparameters=hyperparameters,
                results=results,
                verbose=verbose)
    
    return results




def run_evaluation(train_data, test_data, evaluation_method, data_syn_is_sequential, hyperparameters, results, verbose):
    
    ### Data Preprocessing

    # split data into train, test, val
    train, val = train_test_split(train_data, split_ratio=0.9)
    test = test_data

    # scale data
    scaler = Scaler(train, no_features_to_scale=9)
    train_scaled = scaler.scale_data(train)
    val_scaled = scaler.scale_data(val)
    test_scaled = scaler.scale_data(test, input_data_is_sequential=data_syn_is_sequential)

    # split test data BUT split TRAINING data only if synthetic data is not sequential
    if evaluation_method == 'TSTR':

        # Train and val data
        if data_syn_is_sequential:
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
        if data_syn_is_sequential:
            test_seq_scaled = test_scaled # keep data as is
        else:
            test_seq_scaled = split_data_into_sequences(test_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)

    
    elif evaluation_method == 'combined':
        pass


    # extract features and targets
    X_train, y_train, X_test, y_test, X_val, y_val = extract_features_and_targets_reg(train_seq_scaled, test_seq_scaled, val_seq_scaled)


    # Create Datasets and Dataloader
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=hyperparameters["batch_size"], shuffle=False)


    ### criterion

    criterion_MSE = nn.MSELoss()
    criterion_MAE = nn.L1Loss()

    for _ in tqdm(range(hyperparameters["num_evaluation_runs"])):

        # get model and optimizer
        model = LSTMRegression(
            device=hyperparameters["device"],
            input_size=train_data.shape[-1],
            hidden_size=hyperparameters["hidden_size"],
            num_stacked_layers=hyperparameters["num_layers"]
        ).to(hyperparameters["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["lr"])

        # train model once
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion_MSE,
            optimizer=optimizer,
            device=hyperparameters["device"],
            num_epochs=hyperparameters["num_epochs"],
            verbose=verbose
        )

        # evaluate model on test data and save results
        with torch.inference_mode():
            model.eval()
            preds = model(X_test.to(hyperparameters["device"]))

            # inverse scale data to get real values for error calculation
            preds_unscaled = torch.tensor(scaler.inverse_scale_target(preds.cpu().numpy().reshape(-1, 1)))
            y_test_unscaled = torch.tensor(scaler.inverse_scale_target(y_test.cpu().numpy().reshape(-1, 1)))

            mae = criterion_MAE(preds_unscaled, y_test_unscaled).item()
            mse = criterion_MSE(preds_unscaled, y_test_unscaled).item()

            results = pd.concat([results, pd.DataFrame([{'Model': evaluation_method, 'Metric': 'MAE', 'Error': mae}])], ignore_index=True)
            results = pd.concat([results, pd.DataFrame([{'Model': evaluation_method, 'Metric': 'MSE', 'Error': mse}])], ignore_index=True)

        return results


def run_asdf(data_real, data_syn, data_syn_is_sequential, hyperparameters, results, verbose):
    
    ### Data Preprocessing

    # split data into train, test, val
    train_TRTS, val_TRTS = train_test_split(data_real, split_ratio=0.9)
    test_TRTS = data_syn

    # scale data
    scaler_TRTS = Scaler(train_TRTS, no_features_to_scale=9)
    train_TRTS_scaled = scaler_TRTS.scale_data(train_TRTS)
    val_TRTS_scaled = scaler_TRTS.scale_data(val_TRTS)
    test_TRTS_scaled = scaler_TRTS.scale_data(test_TRTS, input_data_is_sequential=data_syn_is_sequential)

    # turn into sequential data
    train_TRTS_seq_scaled = split_data_into_sequences(train_TRTS_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)
    val_TRTS_seq_scaled = split_data_into_sequences(val_TRTS_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)

    if data_syn_is_sequential:
        test_TRTS_seq_scaled = test_TRTS_scaled # keep data as is
    else:
        test_TRTS_seq_scaled = split_data_into_sequences(test_TRTS_scaled, seq_len=hyperparameters["seq_len"], shuffle_data=True)


    # extract features and targets
    X_train_TRTS, y_train_TRTS, X_test_TRTS, y_test_TRTS, X_val_TRTS, y_val_TRTS = extract_features_and_targets_reg(train_TRTS_seq_scaled, test_TRTS_seq_scaled, val_TRTS_seq_scaled)


    # Create Datasets and Dataloader
    train_dataset_TRTS = TimeSeriesDataset(X_train_TRTS, y_train_TRTS)
    val_dataset_TRTS = TimeSeriesDataset(X_val_TRTS, y_val_TRTS)

    train_loader_TRTS = DataLoader(train_dataset_TRTS, batch_size=hyperparameters["batch_size"], shuffle=False)
    val_loader_TRTS = DataLoader(val_dataset_TRTS, batch_size=hyperparameters["batch_size"], shuffle=False)


    ### criterion

    criterion_MSE = nn.MSELoss()
    criterion_MAE = nn.L1Loss()

    for _ in tqdm(range(hyperparameters["num_evaluation_runs"])):

        # get model and optimizer
        TRTS_model = LSTMRegression(
            device=hyperparameters["device"],
            input_size=data_real.shape[-1],
            hidden_size=hyperparameters["hidden_size"],
            num_stacked_layers=hyperparameters["num_layers"]
        ).to(hyperparameters["device"])
        TRTS_optimizer = torch.optim.Adam(TRTS_model.parameters(), lr=hyperparameters["lr"])

        # train model once
        train_model(
            model=TRTS_model,
            train_loader=train_loader_TRTS,
            val_loader=val_loader_TRTS,
            criterion=criterion_MSE,
            optimizer=TRTS_optimizer,
            device=hyperparameters["device"],
            num_epochs=hyperparameters["num_epochs"],
            verbose=verbose
        )

        # evaluate model on test data and save results
        with torch.inference_mode():
            TRTS_model.eval()
            TRTS_preds = TRTS_model(X_test_TRTS.to(hyperparameters["device"]))

            # inverse scale data to get real values for error calculation
            TRTS_preds_unscaled = torch.tensor(scaler_TRTS.inverse_scale_target(TRTS_preds.cpu().numpy().reshape(-1, 1)))
            y_test_TRTS_unscaled = torch.tensor(scaler_TRTS.inverse_scale_target(y_test_TRTS.cpu().numpy().reshape(-1, 1)))

            mae = criterion_MAE(TRTS_preds_unscaled, y_test_TRTS_unscaled).item()
            mse = criterion_MSE(TRTS_preds_unscaled, y_test_TRTS_unscaled).item()

            results = pd.concat([results, pd.DataFrame([{'Model': 'TRTS', 'Metric': 'MAE', 'Error': mae}])], ignore_index=True)
            results = pd.concat([results, pd.DataFrame([{'Model': 'TRTS', 'Metric': 'MSE', 'Error': mse}])], ignore_index=True)