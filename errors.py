import numpy as np

def make_jackknife_blocks(x):
    x = np.ma.masked_invalid(np.array(x))
    N = len(x)
    mean = sum(x) / N #mean
    blocks = [(N*mean - x[i])/(N-1) for i in range(N)]
    return np.array(blocks), N

def find_error_blocks(x):
    N = len(x) + 1
    mean = np.average(x)
    error = np.sqrt(np.sum((x-mean)**2)*(N-1)/N)
    return mean, error
