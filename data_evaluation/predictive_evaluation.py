import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from baseline_model.LSTM import LSTMRegression, train_model
from utilities import Scaler, train_test_split, extract_features_and_targets, split_data_into_sequences
from baseline_model.TimeSeriesDataset import TimeSeriesDataset

def predictive_evaluation(data_real: np.array, data_syn: np.array, hyperparameters, syn_data_is_sequential, verbose=True):

    EVALUATION_RUNS = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = pd.DataFrame(columns=['Model', 'Metric', 'Error'])

    ### Data Preprocessing

    # Scale data
    scaler_real = Scaler(data_real)
    scaler_syn = Scaler(data_syn)

    # Split data 
    









    # save unscaled targets for evaluation later on
    _, y_real_unscaled = train_test_split(data_real_split, split_ratio=-1)
    _, y_syn_unscaled = train_test_split(data_syn_split, split_ratio=-1)

    # scale data
    prep_data_real, scaler_real = scale_data(data_real_split)
    prep_data_syn, scaler_syn = scale_data(data_syn_split)

    # split into features and target
    # no train test split, because we will use real data for training and synthetic data for testing and vice versa
    X_real, y_real = train_test_split_to_tensor(prep_data_real, split_ratio=-1)
    X_syn, y_syn = train_test_split_to_tensor(prep_data_syn, split_ratio=-1)

    dataset_real = TimeSeriesDataset(X_real, y_real)
    dataset_syn = TimeSeriesDataset(X_syn, y_syn)   

    real_data_loader = DataLoader(dataset_real, batch_size=hyperparameters["batch_size"], shuffle=False)
    syn_data_loader = DataLoader(dataset_syn, batch_size=hyperparameters["batch_size"], shuffle=False)
    

    ### Data Preprocessing (Combined)
    
    # combine real and synthetic data
    # remember to put synthetic data first, since train_test_split_to_tensor() will take the last samples (real data) as test set
    data_comb_unscaled = np.vstack((data_syn_split, data_real_split))

    # get unscaled test data for actual MAE later on
    _, _, _, y_comb_test_unscaled = train_test_split_to_tensor(data_comb_unscaled)

    data_comb, scaler_comb = scale_data(data_comb_unscaled)
    X_comb_train, y_comb_train, X_comb_test, y_comb_test = train_test_split_to_tensor(data_comb)
    
    dataset_comb_train = TimeSeriesDataset(X_comb_train, y_comb_train)
    dataset_comb_test = TimeSeriesDataset(X_comb_test, y_comb_test)

    comb_data_loader_train = DataLoader(dataset_comb_train, batch_size=hyperparameters["batch_size"], shuffle=False)
    comb_data_loader_test = DataLoader(dataset_comb_test, batch_size=hyperparameters["batch_size"], shuffle=False)


    ### criterion

    criterion_MSE = nn.MSELoss()
    criterion_MAE = nn.L1Loss()

    for _ in tqdm(range(EVALUATION_RUNS)):

        ### Train on Real, Test on Synthetic (TRTS)

        TRTS_model = LSTMRegression(
            device=device,
            input_size=dim,
            hidden_size=hyperparameters["hidden_size"],
            num_stacked_layers=hyperparameters["num_layers"]
        ).to(device)

        TRTS_optimizer = torch.optim.Adam(TRTS_model.parameters(), lr=hyperparameters["lr"])

        _, _ = train_model(
            model=TRTS_model,
            train_loader=real_data_loader,
            val_loader=syn_data_loader,
            criterion=criterion_MSE,
            optimizer=TRTS_optimizer,
            device=device,
            verbose=verbose
        )

        with torch.inference_mode():
            TRTS_model.eval()
            TRTS_preds = TRTS_model(X_syn.to(device))
            TRTS_preds_unscaled = torch.tensor(inverse_scale_data(TRTS_preds.cpu().numpy(), scaler_syn, seq_len))

            mae = criterion_MAE(TRTS_preds_unscaled, y_syn_unscaled).item()
            mse = criterion_MSE(TRTS_preds_unscaled, y_syn_unscaled).item()

            results = pd.concat([results, pd.DataFrame([{'Model': 'TRTS', 'Metric': 'MAE', 'Error': mae}])], ignore_index=True)
            results = pd.concat([results, pd.DataFrame([{'Model': 'TRTS', 'Metric': 'MSE', 'Error': mse}])], ignore_index=True)


        ### Train on Synthetic, Test on Real (TSTR)

        TSTR_model = LSTMRegression(
            device=device,
            input_size=dim,
            hidden_size=hyperparameters["hidden_size"],
            num_stacked_layers=hyperparameters["num_layers"]
        ).to(device)

        TSTR_optimizer = torch.optim.Adam(TSTR_model.parameters(), lr=hyperparameters["lr"])

        _, _ = train_model(
            model=TSTR_model,
            train_loader=syn_data_loader,
            val_loader=real_data_loader,
            criterion=criterion_MSE,
            optimizer=TSTR_optimizer,
            device=device,
            verbose=verbose
        )

        with torch.inference_mode():
            TSTR_model.eval()
            TSTR_preds = TSTR_model(X_real.to(device))
            TSTR_preds_unscaled = torch.tensor(inverse_scale_data(TSTR_preds.cpu().numpy(), scaler_real, seq_len))

            mae = criterion_MAE(TSTR_preds_unscaled, y_real_unscaled).item()
            mse = criterion_MSE(TSTR_preds_unscaled, y_real_unscaled).item()

            results = pd.concat([results, pd.DataFrame([{'Model': 'TSTR', 'Metric': 'MAE', 'Error': mae}])], ignore_index=True)
            results = pd.concat([results, pd.DataFrame([{'Model': 'TSTR', 'Metric': 'MSE', 'Error': mse}])], ignore_index=True)


        ### Combined Testing

        comb_model = LSTMRegression(
            device=device,
            input_size=dim,
            hidden_size=hyperparameters["hidden_size"],
            num_stacked_layers=hyperparameters["num_layers"]
        ).to(device)

        comb_optimizer = torch.optim.Adam(comb_model.parameters(), lr=hyperparameters["lr"])

        _, _ = train_model(
            model=comb_model,
            train_loader=comb_data_loader_train,
            val_loader=comb_data_loader_test,
            criterion=criterion_MSE,
            optimizer=comb_optimizer,
            device=device,
            verbose=verbose
        )

        with torch.inference_mode():
            comb_model.eval()
            comb_preds = comb_model(X_comb_test.to(device))
            comb_preds_unscaled = torch.tensor(inverse_scale_data(comb_preds.cpu().numpy(), scaler_comb, seq_len))

            mae = criterion_MAE(comb_preds_unscaled, y_comb_test_unscaled).item()
            mse = criterion_MSE(comb_preds_unscaled, y_comb_test_unscaled).item()

            results = pd.concat([results, pd.DataFrame([{'Model': 'combined', 'Metric': 'MAE', 'Error': mae}])], ignore_index=True)
            results = pd.concat([results, pd.DataFrame([{'Model': 'combined', 'Metric': 'MSE', 'Error': mse}])], ignore_index=True)

    return results