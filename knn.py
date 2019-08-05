#!/home/ych/anaconda3/bin/python
# -*- coding: utf-8 -*-

import operator
import random
import utils
from dba import calculate_dist_matrix

def get_neighbors(x_train, x_test_instance, k, dist_fun, dist_fun_params, 
                  pre_computed_matrix=None, index_test_instance=None,
                  return_distances = False):
    """
    Given a test instance, this function returns its neighbors present in x_train
    NB: If k==0 zero it only returns the distances
    :param x_train:
    :param x_test_instance:
    :param k:
    :param dist_fun:
    :param dist_fun_params:
    :param pre_computed_matrix:
    :param index_test_instance:
    :param return_distances:
    :return:
    """
    distances = []
    # loop through the training set 
    for i in range(len(x_train)): 
        # calculate the distance between the test instance and each training instance
        if pre_computed_matrix is None: 
            _, _, dist, _ = dist_fun(x_test_instance, x_train[i], **dist_fun_params)
        else: 
            # do not re-compute the distance just get it from the precomputed one
            dist = pre_computed_matrix[i, index_test_instance]
        # add the index of the current training instance and its corresponding distance 
        distances.append((i, dist))
    # if k (nb_neighbors) is zero return all the items with their distances 
    # NOT SORTED 
    if k == 0:
        if return_distances == True: 
            return distances
        else:
            print('Not implemented yet')
            exit()
    # sort list by specifying the second item to be sorted on 
    distances.sort(key=operator.itemgetter(1))
    # else do return only the k nearest neighbors
    neighbors = []
    for i in range(k): 
        if return_distances == True: 
            # add the index and the distance of the k nearest instances from the train set 
            neighbors.append(distances[i])
        else:
            # add only the index of the k nearest instances from the train set 
            neighbors.append(distances[i][0])
        
    return neighbors


if __name__ == '__main__':
    tseries = utils.tseries
    dist_fun = utils.DISTANCE_ALGORITHMS['DTW']
    dist_fun_params = utils.DISTANCE_ALGORITHMS_PARAMS['DTW']

    n = len(tseries)
    idx_center = random.randint(0, n - 1)
    init_dba = tseries[idx_center]
    max_k = 5
    max_subk = 2
    k = min(max_k, n - 1)
    dist_pair_mat = calculate_dist_matrix(tseries, dist_fun, dist_fun_params)
    print(dist_pair_mat)
    print(idx_center)
    print(init_dba)
    neighbors = get_neighbors(tseries, init_dba, k, dist_fun, dist_fun_params, dist_pair_mat, idx_center)
    print(neighbors)