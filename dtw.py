#!/home/ych/anaconda3/bin/python
# -*- coding: utf-8 -*-

# inspired by https://github.com/pierre-rouanet/dtw

import numpy as np

def distance(x, y):
    return (x - y) * (x - y)

def dynamic_time_warping(x, y, distance=distance, window_size=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array.
    :param array y: N2*M array.
    :param func dist_fun: distance used as cost measure.
    :param float window_size: window size limiting the maximal distance.

    Returns the cost matrix, the accumulated cost, the minumum distance, and the wrap path.
    """

    assert len(x), 'the length of x is smaller than 1'
    assert len(y), 'the length of x is smaller than 1'
    assert int(len(x) * window_size) >= abs(len(x) - len(y)), 'the window size is smaller than the abs of x and y'
    
    lx, ly = len(x), len(y)
    r, c = lx + 1, ly + 1
    w = int(len(x) * window_size)

    #print('x-len = {r} \ny-len = {c}'.format(r=r,c=c))

    if window_size != 1.0:
        D = np.full((r, c), np.inf)
        for i in range(1, r):
            D[i, max(1, i - w): min(c, i + w + 1)] = 0
        D[0, 0] = 0
    else:
        D = np.zeros((r, c), dtype=np.float64)
        D[0, 1:] = np.inf
        D[1:, 0] = np.inf

    # 浅拷贝
    D_copy = D[1:, 1:]
    # 原始的距离矩阵
    for i in range(lx):
        for j in range(ly):
            if (window_size == 1.0 or max(0, i - w) <= j <= min(ly, i + w)):
                D_copy[i, j] = distance(x[i], y[j])
    # 深拷贝
    C = D_copy.copy()
    jrange = range(ly)
    # 动态计算最短路径
    for i in range(lx):
        if window_size != 1.0:
            jrange = range(max(0, i - w), min(ly, i + w +1))
        for j in jrange:
            D_copy[i, j] += min(D[i, j], D[i, j + 1], D[i + 1, j])

    # 计算后的距离矩阵的[-2, -2]位置的下标
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    # 从后往前推算路径
    while i > 0 or j > 0:
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else: # tb == 2
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    
    return C, D_copy, D_copy[-1, -1], list(zip(p, q))
    

if __name__ == '__main__':
    x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
    y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
    window_size = 1.0
    # x = [1,2,3,4,5,5,5,4]
    # y = [3,4,5,5,5,4]
    x = [0,1,2,2,2,1]
    y = [1,1,2,3,1,0]
    D_original, D_calculated, dist, path = dynamic_time_warping(x, y, distance, window_size)
    print(D_original)
    print(D_calculated)
    print(path)