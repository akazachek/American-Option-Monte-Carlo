import estimators
import numpy as np
import scipy.stats as ss

def pricer(strike, s0, r, vol, div, duration = 1, exercise_dates = 252, num_paths = 50000, num_simplex = 100, estimates = 10, alpha = 0.05):
    """
     Gets confidence interval for option value and exercise rules.
            Parametres:
                strike: strike price
                s0: initial asset price
                r: risk-free rate
                vol: annual volatility
                div: dividend yield (default = 0)
                duration: time until option expiry (default = 1 i.e. one year)
                exercise_dates: how many days the option can be exercised, spaced linearly
                                until its expiration (default = 252 i.e. each trading day)
                num_paths: number of Monte-Carlo price simulations to do (default = 50000)
                num_simplex: number of simplexes to try during optimization
                estimates: number of estimators to generate
                alpha: confidence interval percentile 
            Returns:
                interval: (1-alpha)*100% confidence interval for option value
                point: point estimate for option value
                rules: exercising rules
    """
    lows = np.zeros(estimates)
    highs = np.zeros(estimates)
    rules = np.zeros((estimates, exercise_dates - 1))

    # perform repeated simulations for the high and low estimators
    for i in range(estimates):
        highs[i], rules[i,] = estimators.high_estimator(strike, s0, r, vol, div, duration, exercise_dates, num_paths, num_simplex)
        lows[i] = estimators.low_estimator(rules[i,], strike, s0, r, vol, div, duration, num_paths)

    # get sample standard deviations
    if estimates == 1:
        low_std = 0
        high_std = 0
    else:
        low_std = np.std(lows, ddof = 1)
        high_std = np.std(highs, ddof = 1)
    low = np.mean(lows)
    high = np.mean(highs)
    rules = np.mean(rules, axis=0)

    z = ss.norm.ppf(1 - alpha / 2.)
    interval = (low - z*low_std, high + z*high_std)
    point = (low + high) / 2

    return (interval, point, rules)

##########
### replicating table from the paper
#########

initial_prices = [70 + 10*i for i in range(6)]
true_prices = [0.121, 0.670, 2.303, 5.731, 11.341, 20]
results = []
for s0 in initial_prices: results.append(pricer(100, s0, 0.05, 0.2, 0.1, exercise_dates = 4, num_paths=10000, num_simplex=10, estimates = 40))

for i in range(6):
    print("-------")
    print(f"Initial price: {initial_prices[i]}.")
    print(f"Confidence interval: {results[i][0]}.")
    print(f"Point estimate: {results[i][1]}.")
    print(f"True price: {true_prices[i]}")
    print(f"Relative error: {np.abs(true_prices[i] - results[i][1]) / results[i][1] * 100}%.")
    print(f"Rules: {results[i][2]}.")