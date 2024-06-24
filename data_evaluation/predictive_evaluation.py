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

    criterion = nn.MSELoss()


    ### Train on Real, Test on Synthetic (TRTS)

    TRTS_model = LSTM(
        device=device,
        input_size=len(features),
        hidden_size=hyperparameters["hidden_size"],
        num_stacked_layers=hyperparameters["num_layers"]
    ).to(device)

    TRTS_optimizer = torch.optim.Adam(TRTS_model.parameters(), lr=hyperparameters["lr"])

    results["TRTS_validation_loss"], _ = train_model(
        model=TRTS_model,
        train_loader=real_data_loader,
        test_loader=syn_data_loader,
        criterion=criterion,
        optimizer=TRTS_optimizer,
        device=device
    )


    ### Train on Synthetic, Test on Real (TSTR)

    TSTR_model = LSTM(
        device=device,
        input_size=len(features),
        hidden_size=hyperparameters["hidden_size"],
        num_stacked_layers=hyperparameters["num_layers"]
    ).to(device)

    TSTR_optimizer = torch.optim.Adam(TSTR_model.parameters(), lr=hyperparameters["lr"])

    results["TSTR_validation_loss"], _ = train_model(
        model=TSTR_model,
        train_loader=syn_data_loader,
        test_loader=real_data_loader,
        criterion=criterion,
        optimizer=TSTR_optimizer,
        device=device
    )

    return results