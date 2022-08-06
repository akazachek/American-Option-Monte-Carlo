from monte_carlo import monte_carlo
import numpy as np
import scipy.optimize as so
import scipy.stats as ss

def rule_value(exercise_rules, paths, strike, r):
    """
        Computes the expected discounted payoff for a particular exercise 
        rule on a simulation.
            
            Parametres:
                exercise_rules: daily prices at which option is to exercised
                                except for the last day
                paths: simulated asset prices
                strike: strike price
                r: risk-free rate
            Returns:
                -value: negative average discounted payoff
    """
    num_paths, exercise_dates = paths.shape
    # append exercise rule on d.o.e. (strike price)
    exercise_rules = np.append(exercise_rules, strike)

    # find stopping times for each path by finding earliest point at 
    # which asset price exceeds or matches the exercise rule
    stopping_times = np.argmax(paths >= exercise_rules, axis = 1)

    # since stopping_times works on a binary array, if the path never
    # exceeds the stopping price, it will be filled with all 0s, and so
    # the argmax will return 0. however, in this case we actually want to
    # exercise the option on the d.o.e., so we must catch this edge case
    zeros = np.nonzero(stopping_times == 0)
    if len(zeros) != 0:
        zeros = zeros[0]
        # find profit if option were exercised on the first day according
        # to the rule on these indices
        initial_exercise_profit = paths[zeros, 0] - exercise_rules[0]
        # get the indices for when this profit is negative and change the
        # stopping time to the last day
        false_exercises = zeros[np.where(initial_exercise_profit < 0)]
        stopping_times[false_exercises] = exercise_dates - 1


    # find the value of the asset at the exercise point on each path
    exercise_values = paths[np.arange(num_paths), stopping_times]
    # compute the discounted payoff for each path and average
    discounts = np.exp(-r*stopping_times / (exercise_dates - 1))
    payoffs = np.maximum(exercise_values - strike, np.zeros(num_paths))
    value = np.dot(discounts, payoffs) / num_paths

    # note we return the negative for use in optimization
    return -value  

def high_estimator(strike, s0, r, vol, div = 0, duration = 1, exercise_dates = 252, num_paths = 50000, num_optim = 100):
    """
        Optimizes the high-bias estimate for the option value as well as 
        the exercise rule.

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
                num_optim: number of linearly-interpolated minima to average
            Returns:
                high_estimator: high-biased estimate for option value
                rules: daily prices at which option should be exercised except for
                        the last day
    """
    # generate simulated asset paths
    paths = monte_carlo(s0, r, vol, div, duration, exercise_dates, num_paths)
   
    # lower bound for exercise rule is the strike price. also
    # note that one exercise rule is always fixed, the one for d.o.e
    bounds = ((strike, np.inf),) * (exercise_dates - 1)

    # discounted payoff is not differentiable so we use Nelder-Mead.
    # however, since Nelder-Mead can get stuck in local minima, we
    # optimize the same rule starting at many different polytopes, interpolating
    # a daily exercise rule from the strike price to 120% of the strike price
    optims = []
    guesses = np.linspace(1., 1.2, num_optim)
    for i in range(num_optim):
        optims.append(so.minimize(rule_value, np.full(exercise_dates - 1, strike * guesses[i]), args = (paths, strike, r),
                                                method = "nelder-mead", bounds = bounds, options = {
                                                    "xatol": 1e-7,
                                                    "fatol": 1e-7,
                                                }))

    # these rules and valuations are then averaged out, trimming
    # the worst and best 15%
    rules = np.zeros((num_optim, exercise_dates - 1))
    high_estimators = np.zeros(num_optim)
    for i in range(num_optim):
        rules[i] = optims[i]["x"]
        high_estimators[i] = optims[i]["fun"]
    rules = ss.trim_mean(rules, .15)
    high_estimator_mean = ss.trim_mean(high_estimators, .15)
    
    # need to return the minimum as we use negative discounted payoff 
    # during optimization
    return (-high_estimator_mean, rules)

def low_estimator(exercise_rules, strike, s0, r, vol, div = 0, duration = 1, num_paths = 50000):
    """
        Generates the low-bias estimate for option price using the
        high-bias exercise rules.

            Parametres:
                exercise_rules: daily prices at which option is to exercised
                                except for the last day
                strike: strike price
                s0: initial asset price
                r: risk-free rate
                vol: annual volatility
                div: dividend yield (default = 0)
                duration: time until option expiry (default = 1 i.e. one year)
                num_paths: number of Monte-Carlo price simulations to do (default = 50000)
            Returns:
                -low_estimator: low-biased estimate for the option value
    """
    # generate simulated asset paths
    exercise_dates = exercise_rules.shape[0] + 1
    paths = monte_carlo(s0, r, vol, div, duration, exercise_dates, num_paths)
    low_estimator_mean = rule_value(exercise_rules, paths, strike, r)
    return -low_estimator_mean