import numpy as np
import pandas as pd
from copy import deepcopy as dc

def create_jittered_time_series(data: pd.DataFrame, jitter_factor: float, no_features) -> np.ndarray:
    """Create a jittered time series by adding random noise to the original time series.
    The jittering only affects continuous features and leaves categorical features as is.
    Jittered values below 0 are set to 0.

    Args:
        data: The original time series.
        jitter: The amount of noise to add to the time series relative to the standard deviation of the feature.
        seed: The random seed to use.

    Returns:
        The jittered time series.
    """

    data_dc = dc(data)

    # save the columns and convert the data to a numpy array
    columns = data_dc.columns
    data_np = data_dc.to_numpy()

    mean = 0
    noise = np.zeros_like(data_np)

    # Add noise scaled by the standard deviation of each feature
    for i in range(no_features):
        std = jitter_factor * np.std(data_np[:, i])
        noise[:, i] = np.random.normal(mean, std, data_np.shape[0])

    # add noise and set values below 0 to 0
    noisy_data = data_np + noise
    noisy_data[noisy_data < 0] = 0

    noisy_data_df = pd.DataFrame(noisy_data, columns=columns)

    noisy_data_df.to_csv(f'jittered_data_{str(jitter_factor).replace(".", "")}.csv', index=False)