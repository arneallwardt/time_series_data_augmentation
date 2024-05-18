import matplotlib.pyplot as plt

class DataHelper:
    def plot_close_price(df, title='Time Series', x='Date', y='Close'):
        plt.figure(figsize=(10, 6))
        plt.plot(df[x], df[y])
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y[0])
        plt.grid(True)
        plt.show()

    def slice_years(df, years, index='Date'):
        # set index to Date to make slicing possible
        dataframe = df.set_index(index, inplace=False)

        # check if years are present
        years_in_index = dataframe.index.year.unique()
        for year in years:
            if int(year) not in years_in_index:
                raise ValueError(f'years {years} not in dataframe')

        # slice dataframe to only include specific years
        if len(years) > 1:
            # check if first year smaller than last year
            if int(years[0]) > int(years[1]):
                raise ValueError(f'years {years} have to be in ascending order')
            dataframe = dataframe.loc[years[0]:years[-1]]
        else:
            dataframe = dataframe.loc[years[0]]
            
        # reset index to prevent errors later on
        dataframe.reset_index(inplace=True)

        return dataframe
