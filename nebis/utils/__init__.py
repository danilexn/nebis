import numpy as np
import random
import torch


def empty_hook(*args, **kwargs):
    return None


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_survival_y_true(T, E, num_times):
    """
    Get y_true for survival prediction based on T and E
    """
    T_max = T.max()

    # Get time points
    time_points = get_time_points(T_max, num_times)

    # Get the y_true
    y_true = []
    for i, (t, e) in enumerate(zip(T, E)):
        y_true_i = np.zeros(num_times + 1)
        dist_to_time_points = [abs(t - point) for point in time_points[:-1]]
        time_index = np.argmin(dist_to_time_points)
        # if this is a uncensored data point
        if e == 1:
            y_true_i[time_index] = 1
            y_true.append(y_true_i)
        # if this is a censored data point
        else:
            y_true_i[time_index:] = 1
            y_true.append(y_true_i)

    return y_true, time_points


def get_time_points(T_max, num_times, extra_time_percent=0.1):
    """
    Get time points for the MTLR model
    """
    # Get time points in the time axis
    time_points = np.linspace(0, T_max * (1 + extra_time_percent), num_times + 1)

    return time_points
