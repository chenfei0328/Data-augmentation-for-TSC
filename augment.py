#!/home/ych/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random
import utils
from dba import calculate_dist_matrix, dba
from knn import get_neighbors

dist_fun = utils.DISTANCE_ALGORITHMS['DTW']
dist_fun_params = utils.DISTANCE_ALGORITHMS_PARAMS['DTW']

def get_weights_average_selected(tseries, dist_pair_mat, distance_algorithm='DTW'):
    """
    Calculate weights with average selected method
    :param array tseries: the list of time series
    :param array dist_pair_mat: the distance matrix
    :param string distance_algorithm: the distance algorithm
    :return: the weights of each sequence and the medoid of the tseries
    """
    # get the number of the train set 
    n = len(tseries)
    # maximum number of K for KNN 
    max_k = 5
    # maximum number of sub neighbors 
    max_subk = 2
    # get the real k for knn 
    k = min(max_k, n - 1)
    # make sure 
    subk = min(max_subk, k)
    # the weight for the center 
    weight_center = 0.5 
    # the total weight of the neighbors
    weight_neighbors = 0.3
    # total weight of the non neighbors 
    weight_remaining = 1.0 - weight_center - weight_neighbors
    # number of non neighbors 
    n_others = n - 1 - subk
    # get the weight for each non neighbor 
    if n_others == 0 : 
        fill_value = 0.0
    else:
        fill_value = weight_remaining / n_others
    # choose a random time series 
    idx_center = random.randint(0, n - 1)
    # get the init dba 
    init_dba = tseries[idx_center]
    # init the weight matrix or vector for univariate time series 
    weights = np.full(n, fill_value, dtype=np.float64)
    # fill the weight of the center 
    weights[idx_center] = weight_center
    # find the top k nearest neighbors
    topk_idx = np.array(get_neighbors(tseries, init_dba, k, dist_fun, dist_fun_params,
                         pre_computed_matrix=dist_pair_mat, 
                         index_test_instance=idx_center))
    # select a subset of the k nearest neighbors
    final_neighbors_idx = np.random.permutation(k)[:subk]
    # 增加判断，判断随机生成的2-NN中不会有最初选定的序列，从而防止权重覆盖
    # fix a bug
    while idx_center in topk_idx[final_neighbors_idx]:
        final_neighbors_idx = np.random.permutation(k)[:subk]
    # adjust the weight of the selected neighbors 
    weights[topk_idx[final_neighbors_idx]] = weight_neighbors / subk
    # return the weights and the instance with maximum weight (to be used as 
    # init for DBA )
    return weights, init_dba

def augment_train_set(x_train, y_train, classes, N, dba_iters=5, 
                      weights_method_name = 'as', distance_algorithm='DTW',
                      limit_N = True):
    """
    This method takes a dataset and augments it using the method in icdm2017. 
    :param x_train: The original train set
    :param y_train: The original labels set 
    :param N: The number of synthetic time series. 
    :param dba_iters: The number of dba iterations to converge.
    :param weights_method_name: The method for assigning weights (see constants.py)
    :param distance_algorithm: The name of the distance algorithm used (see constants.py)
    """
    # get the weights function
    #weights_fun = utils.constants.WEIGHTS_METHODS[weights_method_name]
    weights_fun = get_weights_average_selected

    # synthetic train set and labels 
    synthetic_x_train = []
    synthetic_y_train = []
    # loop through each class
    k = 0
    for c in classes:
        k += 1
        print('class {} of {}'.format(k, len(classes)))
        # get the MTS for this class 
        c_x_train = x_train[np.where(y_train==c)]

        if len(c_x_train) == 1 :
            # skip if there is only one time series per set
            continue

        if limit_N == True:
            # limit the nb_prototypes
            nb_prototypes_per_class = min(N, len(c_x_train))
        else:
            # number of added prototypes will re-balance classes
            nb_prototypes_per_class = N + (N - len(c_x_train))

        # get the pairwise matrix 
        if weights_method_name == 'aa': 
            # then no need for dist_matrix 
            dist_pair_mat = None 
        else:
            dist_pair_mat = calculate_dist_matrix(c_x_train, dist_fun, dist_fun_params)

        t = 0
        # loop through the number of synthtectic examples needed
        for n in range(nb_prototypes_per_class):
            t += 1
            print('nb_prototypes_per_class {} of {}'.format(t, nb_prototypes_per_class))
            # get the weights and the init for avg method 
            weights, init_avg = weights_fun(c_x_train, dist_pair_mat, distance_algorithm=distance_algorithm)
            # get the synthetic data 
            synthetic_mts = dba(c_x_train, dba_iters, weights=weights)  
            # add the synthetic data to the synthetic train set
            synthetic_x_train.append(synthetic_mts)
            # add the corresponding label 
            synthetic_y_train.append(c)
    # return the synthetic set 
    return np.array(synthetic_x_train), np.array(synthetic_y_train)


if __name__ == '__main__':
    tseries = utils.tseries
    dist_pair_mat = calculate_dist_matrix(tseries, dist_fun, dist_fun_params)
    weights, init_dba = get_weights_average_selected(tseries, dist_pair_mat)
    print(weights)
    print(init_dba)