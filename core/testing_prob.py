import matplotlib.pyplot as plt
from itertools import islice
import numpy as np

def sampling(data, sample_size):
    """Returns sampled data sets"""
    return [data[x:x+sample_size] for x in range(0, len(data), sample_size)] 

def gen_window(seq, n):
    """Returns a sliding window (of width n) over data from the iterable"""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def estimate_prob(data, window_size):
    """Returns a probability estimate for given data sample in 
       n=window_size number of seconds"""
    windows = gen_window(data, n=window_size)

    window_prob = []

    for window in windows:
        ones_freq = 0
        for data_point in window:
            if data_point == 1:
                ones_freq += 1
        window_prob.append(ones_freq/window_size)
    return 1 - window_prob.count(0.0)/len(window_prob)

def plot_data(data, time=range(24)):
    fig = plt.figure(facecolor='white', figsize=(12,3))
    plt.plot(time, data, 'b-.o')
    plt.show()

def plot_prob(probs, time_points):
    plt.plot(range(1, time_points), probs)
    plt.show()

def main():
    data = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    # plot_data(data)

    _ = sampling(data, sample_size=10)

    probs_acc = []
    max_w_size = 15
    for w in range(1, max_w_size):
        probs = estimate_prob(data, window_size=w)
        probs_acc.append(probs)

    for idx, val in enumerate(probs_acc):
        print('for window size', idx+1,
                'estimated probability is', val)

    plot_prob(probs_acc, max_w_size)
   

if __name__ == '__main__':
    main()

