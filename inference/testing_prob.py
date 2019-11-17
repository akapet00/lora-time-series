import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from itertools import islice
import numpy as np
from scipy.interpolate import interp1d
import time 

def sampling(data, sample_size):
    """Returns sampled data sets
        Not applicable for non-stationary data
    """
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

def estimate_prob_vectorised(data, window_size):
    """Returns a probability estimate for given numpy array of 
       samples in n=window_size number of seconds"""
    windows = gen_window(data, n=window_size)

    window_prob = []

    for window in windows:
        window_prob.append(np.sum(window)/window_size)
    return 1 - window_prob.count(0.0)/len(window_prob)
    

def plot_data(data):
    fig = plt.figure(facecolor='white', figsize=(12,3))
    plt.plot(range(len(data)), data, 'b-.o')
    plt.xlabel('time [s]')
    plt.ylabel('Activation')
    plt.show()

def plot_prob(probs, time_points):
    f_interp1d = interp1d(time_points, probs)
    xnew = np.arange(min(time_points), max(time_points), step=max(time_points)/1000)
    ynew = f_interp1d(xnew)

    plt.plot(time_points, probs, 'o', xnew, ynew, '-')
    plt.legend(['calculated', 'interpolation'], loc='best')
    plt.xscale('log')
    plt.xlabel('time [s]')
    plt.ylabel('CDF')
    plt.show()

def main():
    
    data = np.ones(100000)
    data[:99900] = 0
    np.random.shuffle(data)
    print('data generated\n')
    #plot_data(data)

    max_w_size = [1, 100, 1000, 3000, 10000, 30000, 90000]
  
    # vectorised
    probs_acc = []
    start = time.time()

    for w in max_w_size:
        probs = estimate_prob_vectorised(data, window_size=w)
        probs_acc.append(probs)
    
    print(f'vectorised time: {time.time() - start}s')

    for i, j in zip(max_w_size, probs_acc):
        print(f'for window size {i} estimated probability is {j}')

    # non vectorised
    probs_acc = []
    start = time.time()

    for w in max_w_size:
        probs = estimate_prob(data, window_size=w)
        probs_acc.append(probs)
    
    print(f'\n\nnon vectorised time: {time.time() - start}s')

    for i, j in zip(max_w_size, probs_acc):
        print(f'for window size {i} estimated probability is {j}')

    plot_prob(probs_acc, max_w_size)
   

if __name__ == '__main__':
    main()

