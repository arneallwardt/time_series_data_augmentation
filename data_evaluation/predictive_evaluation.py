import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from baseline_model.LSTM import LSTM, train_model, scale_data, train_test_split_to_tensor
from baseline_model.TimeSeriesDataset import TimeSeriesDataset

def predictive_evaluation(data_real_split, data_syn_split, hyperparameters, features):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device

    results = {
        "TRTS_validation_loss": None,
        "TSTR_validation_loss": None
    }


    ### Data Preprocessing

    X_real_unscaled, y_real_unscaled = train_test_split_to_tensor(data_real_split, split_ratio=-1)
    X_syn_unscaled, y_syn_unscaled = train_test_split_to_tensor(data_syn_split, split_ratio=-1)

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

    criterion_MSE = nn.MSELoss()
    criterion_MAE = nn.L1Loss()


    ### Train on Real, Test on Synthetic (TRTS)

    TRTS_model = LSTM(
        device=device,
        input_size=len(features),
        hidden_size=hyperparameters["hidden_size"],
        num_stacked_layers=hyperparameters["num_layers"]
    ).to(device)

    TRTS_optimizer = torch.optim.Adam(TRTS_model.parameters(), lr=hyperparameters["lr"])

    results["TRTS_MSE"], _ = train_model(
        model=TRTS_model,
        train_loader=real_data_loader,
        test_loader=syn_data_loader,
        criterion=criterion_MSE,
        optimizer=TRTS_optimizer,
        device=device
    )

    with torch.inference_mode():
        TRTS_model.eval()
        TRTS_preds_unscaled = TRTS_model(X_syn_unscaled)
        results["TRTS_MAE"] = criterion_MAE(TRTS_preds_unscaled, y_syn_unscaled).item()


    ### Train on Synthetic, Test on Real (TSTR)

    TSTR_model = LSTM(
        device=device,
        input_size=len(features),
        hidden_size=hyperparameters["hidden_size"],
        num_stacked_layers=hyperparameters["num_layers"]
    ).to(device)

    TSTR_optimizer = torch.optim.Adam(TSTR_model.parameters(), lr=hyperparameters["lr"])

    results["TSTR_MSE"], _ = train_model(
        model=TSTR_model,
        train_loader=syn_data_loader,
        test_loader=real_data_loader,
        criterion=criterion_MSE,
        optimizer=TSTR_optimizer,
        device=device
    )

    with torch.inference_mode():
        TSTR_model.eval()
        TSTR_preds_unscaled = TSTR_model(X_real_unscaled)
        results["TSTR_MAE"] = criterion_MAE(TSTR_preds_unscaled, y_real_unscaled).item()


    return results