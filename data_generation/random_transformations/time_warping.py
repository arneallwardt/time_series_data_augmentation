from tsaug import TimeWarp
from copy import deepcopy as dc
import numpy as np
import pandas as pd

def create_time_warped_data(data, save_data=True):

    dc_data = dc(data) 

    # save columns and convert to numpy
    columns = dc_data.columns
    dc_data = dc_data.to_numpy()

    # add extra dim since tsaug works with multivariate data
    # shape: (10000, 5) -> (1, 10000, 5)
    data_3d = np.expand_dims(dc_data, axis=0) 
    
    time_warp = TimeWarp()
    data_3d_time_warped = time_warp.augment(data_3d)

    # remove extra dim and convert to pandas dataframe
    data_time_warped = data_3d_time_warped[0]
    data_time_warped = pd.DataFrame(data_time_warped, columns=columns)

    if save_data:
        data_time_warped.to_csv('mitv_time_warped.csv', index=False)
    else:
        return data_time_warped

