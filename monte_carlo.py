import numpy as np
from numpy.random import standard_normal
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def monte_carlo(s0, r, vol, div = 0, time = 1/252, num_steps = 252, num_paths = 1):
    """
        Simulates a log-normal asset price 

        s0 = initial asset price
        r = risk-free rate
        vol = annual volatility
        div = dividend yield (default = 0)
        time = time between simulations (default = 1/252)
        steps = number of points to simulate (default = 252)
        paths = number of simulations to run (default = 1)
    """
    paths = np.zeros((num_paths, num_steps))
    for path in range(num_paths):
        paths[path, 0] = s0
        for step in range(1, num_steps):
            paths[path, step] = paths[path, step - 1]*np.exp( (r - div - vol/2)*time + vol*np.sqrt(time)*standard_normal())
    return paths

def plot_simulations(paths, params = None):
    _, num_steps = paths.shape
    df = pd.DataFrame(paths.T, index = np.arange(num_steps))
    df = df.melt(var_name = "path", value_name="price", ignore_index = False)
    df["step"] = df.index
    df = df.reset_index()
    sns.lineplot(data = df, x = "step", y = "price", hue = "path")
    plt.legend(title = "Path")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    if params is not None:
        plt.title(f"Simulated Asset Price Paths\n $s_0={params[0]},r={params[1]},\sigma={params[2]},\delta={params[3]}$")
    else:
        plt.title("Simulated Asset Price Paths")
    plt.show()







