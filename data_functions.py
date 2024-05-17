import matplotlib.pyplot as plt

def plot_close_price(df, title='Time Series', x='Date', y='Close'):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y[0])
    plt.grid(True)
    plt.show()