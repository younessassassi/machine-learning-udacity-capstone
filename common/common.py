""" Common Functions """

import pandas as pd
import matplotlib.pyplot as plt

"""Plot stock prices with a custom title and axis labels."""
def plot_data(df, title="Stock performance", xlabel="Date", ylabel="Price"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()