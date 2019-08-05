#!/home/ych/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import utils


dist_fun = utils.DISTANCE_ALGORITHMS['DTW']
dist_fun_params = utils.DISTANCE_ALGORITHMS_PARAMS['DTW']


def calculate_dist_matrix(tseries, dist_fun, dist_fun_params):
    """
    Calculate distance of two sequences all over the serieses.
    Returns the distance matrix.
    """
    n = len(tseries)
    pairwise_dist_matrix = np.zeros((n, n), dtype = np.float64)
    # pre-compute the pairwise distance
    for i in range(n - 1):
        x = tseries[i]
        for j in range(i + 1, n):
            y = tseries[j]
            _, _, dist, _ = dist_fun(x, y, **dist_fun_params)
            # # because dtw returns the sqrt
            # dist = dist*dist 
            pairwise_dist_matrix[i, j] = dist 
            # dtw is symmetric 
            pairwise_dist_matrix[j, i] = dist 
        pairwise_dist_matrix[i, i] = 0 
    return pairwise_dist_matrix

def medoid(tseries, dist_fun, dist_fun_params):
    """
    Calculates the medoid of the given list of MTS

    :param array tseries: the list of time series
    :param func dist_fun: distance used as cost measure
    :param dict dist_fun_params: params needed in dist_fun

    Return the index of the medoid, and the medoid sequence.
    """
    n = len(tseries)
    if n == 1 : 
        return 0, tseries[0]

    pairwise_dist_matrix = calculate_dist_matrix(tseries, dist_fun, dist_fun_params)
    
    sum_dist = np.sum(pairwise_dist_matrix, axis = 0)
    min_idx = np.argmin(sum_dist)
    med = tseries[min_idx]
    return min_idx, med

def _dba_iteration(tseries, avg, dist_fun, dist_fun_params, weights):
    """
    Perform one weighted dba iteration

    :param array tseries: the list of time series
    :param array avg: the initial avg
    :param func dist_fun: distance used as cost measure
    :param dict dist_fun_params: params needed in dist_fun
    :param array weights: the weights of each sequences in tseries

    Return the new average 
    """
    # the number of time series in the set
    n = len(tseries)
    # length of the time series 
    l = len(avg)
    # array containing the new weighted average sequence 
    new_avg = np.zeros(l, dtype=np.float64)
    # array of sum of weights 
    sum_weights = np.zeros(l, dtype=np.float64)
    # loop the time series
    for s in range(n): 
        series = tseries[s]
        _, _, _, path = dist_fun(avg, series, **dist_fun_params)

        # 按照DTW的路径来赋值。原方法是在DTW矩阵里回溯，而本算法可以直接得到path。差异在于找路径时可能方向不一样导致轻微的数值差异
        for position in path:
            i = position[0]
            j = position[1]
            new_avg[i] += series[j] * weights[s]
            sum_weights[i] += weights[s]

        # print(new_avg)
        # print(sum_weights)
    
    # update the new weighted average
    new_avg = new_avg / sum_weights
    
    return new_avg

def dba(tseries, max_iter=10, weights=None):
    """
    Computes the Dynamic Time Warping (DTW) Barycenter Averaging (DBA) of a 
    group of Multivariate Time Series (MTS). 

    :param array tseries: A list containing the series to be averaged, where each 
        MTS has a shape (l,m) where l is the length of the time series and 
        m is the number of dimensions of the MTS - in the case of univariate 
        time series m should be equal to one
    :param int max_iter: The maximum number of iterations for the DBA algorithm.

    :param distance_algorithm: Determine which distance to use when aligning 
        the time series

    :param array weights: An array containing the weights to calculate a weighted dba
        (NB: for MTS each dimension should have its own set of weights)
        expected shape is (n,m) where n is the number of time series in tseries 
        and m is the number of dimensions

    Return the weighted DTW sequence.
    """
    assert len(tseries), 'the number of time series is smaller than 1'

    avg = np.copy(medoid(tseries, dist_fun, dist_fun_params)[1])

    if len(tseries) == 1:
        return avg

    for i in range(max_iter):
        print('Iteration {}'.format(i + 1))
        if weights is None:
            # when giving all time series a weight equal to one we have the 
            # non - weighted version of DBA 
            weights = np.ones(len(tseries), dtype=np.float64)
        # dba iteration 
        avg = _dba_iteration(tseries, avg, dist_fun, dist_fun_params, weights)

    return avg

if __name__ == "__main__":
    tseries = utils.tseries
    # print(tseries)
    # pairwise_dist_matrix = calculate_dist_matrix(tseries, dist_fun, dist_fun_params)
    # print(pairwise_dist_matrix)
    #

    min_idx, med = medoid(tseries, dist_fun, dist_fun_params)
    print(min_idx)
    print(med)
    #
    # avg = np.copy(medoid(tseries, dist_fun, dist_fun_params)[1])
    # weights = np.ones(len(tseries), dtype=np.float64)
    # avg = _dba_iteration(tseries, avg, dist_fun, dist_fun_params, weights)
    # print(avg)

    # avg = dba(tseries)
    # print(avg)