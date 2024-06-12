import numpy as np
from scipy.spatial.distance import pdist, squareform 
import matplotlib.pyplot as plt
import pandas as pd


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
            raise ValueError(f'Dataframe does not include one of the following column names: {[x, y1, y2]}.')
        
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


### EVALUATION METRICS ###

def maximum_mean_discrepancy(X, Y):
    '''
    maximum mean discrepancy (mmd) is a metric for comparing the distribution of 2 datasets
    '''

    def _gaussian_kernel(x, y, sigma=1.0):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
    
    XX = squareform(pdist(X, metric=lambda x, y: _gaussian_kernel(x, y)))
    YY = squareform(pdist(Y, metric=lambda x, y: _gaussian_kernel(x, y)))
    XY = np.array([[_gaussian_kernel(x, y) for y in Y] for x in X])
    return XX.mean() + YY.mean() - 2 * XY.mean()