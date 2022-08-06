import numpy as np
from numpy.random import standard_normal
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def monte_carlo(s0, r, vol, div = 0, time = 1, num_steps = 252, num_paths = 1, plot = False):
    """
        Simulates a log-normal asset price.

            Parametres:
                s0: initial asset price
                r: risk-free rate
                vol: annual volatility
                div: dividend yield
                time: time of simulation (default = 1 i.e. one year)
                steps: number of points to simulate (default = 252 i.e. one trading year)
                paths: number of simulations to run (default = 1)
                plot: plot simulations (default = False)
            Returns:
                paths: array where each row corresponds to one simulation
    """
    paths = np.zeros((num_paths, num_steps))
    time_step = time / (num_steps - 1)
    for path in range(num_paths):
        paths[path, 0] = s0
        for step in range(1, num_steps):
            paths[path, step] = paths[path, step - 1]*np.exp( (r - div - vol**2 / 2.)*time_step + vol*np.sqrt(time_step)*standard_normal())

    if plot: plot_simulations(paths)
    return paths

def plot_simulations(paths, params = None):
    """
        Plots asset price simulations.

            Parametres:
                paths: simulations to plot
                params: optional tuple (s0, r, vol, div) of asset information
    """
    _, num_steps = paths.shape
    # data frame where each column is a different simulation, indexed by
    # the step of the simulation
    df = pd.DataFrame(paths.T, index = np.arange(num_steps))
    # simulations are melted into one column `price` containing all price 
    # information and a new auxiliary column `path` indicates which simulation
    # they came from, and a new index column `step` indicating the step
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