#!/home/ych/anaconda3/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import utils
from augment import augment_train_set

dist_fun = utils.DISTANCE_ALGORITHMS['DTW']
dist_fun_params = utils.DISTANCE_ALGORITHMS_PARAMS['DTW']


def read_train_set(path):
    x_train, y_train = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            series_class = int(line[0])
            series = [float(x) for x in line[1:]]
            #series = line[1:]
            x_train.append(series)
            y_train.append(series_class)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    classes, classes_counts = np.unique(y_train, return_counts=True)
    # this means that all classes will have a number of time series equal to
    # nb_prototypes
    nb_prototypes = classes_counts.max()
    # print((x_train))
    # print((y_train))
    # print(classes)
    # print(nb_prototypes)
    synthetic_x_train, synthetic_y_train = augment_train_set(x_train, y_train, classes, nb_prototypes)
    print(synthetic_x_train)
    print(synthetic_y_train)

def main():
    # for name in utils.DATA_SET_NAMES:
    #     data_path = utils.DATA_PATH + name + '/' + name + '_TRAIN'
    #     if os.path.exists(data_path):
    #         print('Data set : {}'.format(name))
    #         read_train_set(data_path)
    #         break

    name = 'Car'
    data_path = utils.DATA_PATH + name + '/' + name + '_TRAIN'
    if os.path.exists(data_path):
        print('Data set : {}'.format(name))
        read_train_set(data_path)

if __name__ == '__main__':
    main()