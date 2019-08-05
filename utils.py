#!/home/ych/anaconda3/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from dtw import distance, dynamic_time_warping
#from dba import calculate_dist_matrix, dba

DTW_PARAMS = {'distance': distance, 'window_size': 1.0}

DISTANCE_ALGORITHMS = {'DTW': dynamic_time_warping}
DISTANCE_ALGORITHMS_PARAMS = {'DTW': DTW_PARAMS}

DATA_PATH = '/home/ych/Downloads/UCR_TS_Archive_2015/'

DATA_SET_NAMES = ['50words','Adiac','ArrowHead','Beef','BeetleFly',
                  'BirdChicken','Car','CBF','ChlorineConcentration',
                  'CinC_ECG_torso','Coffee','Computers','Cricket_X',
                  'Cricket_Y','Cricket_Z','DiatomSizeReduction',
                  'DistalPhalanxOutlineAgeGroup',
                  'DistalPhalanxOutlineCorrect','DistalPhalanxTW',
                  'Earthquakes','ECG200','ECG5000','ECGFiveDays',
                  'ElectricDevices','FaceAll','FaceFour','FacesUCR',
                  'FISH','FordA','FordB','Gun_Point','Ham',
                  'HandOutlines','Haptics','Herring','InlineSkate',
                  'InsectWingbeatSound','ItalyPowerDemand',
                  'LargeKitchenAppliances','Lighting2','Lighting7',
                  'MALLAT','Meat','MedicalImages',
                  'MiddlePhalanxOutlineAgeGroup',
                  'MiddlePhalanxOutlineCorrect','MiddlePhalanxTW',
                  'MoteStrain','NonInvasiveFatalECG_Thorax1',
                  'NonInvasiveFatalECG_Thorax2','OliveOil','OSULeaf',
                  'PhalangesOutlinesCorrect','Phoneme','Plane',
                  'ProximalPhalanxOutlineAgeGroup',
                  'ProximalPhalanxOutlineCorrect',
                  'ProximalPhalanxTW','RefrigerationDevices',
                  'ScreenType','ShapeletSim','ShapesAll',
                  'SmallKitchenAppliances','SonyAIBORobotSurface',
                  'SonyAIBORobotSurfaceII','StarLightCurves',
                  'Strawberry','SwedishLeaf','Symbols',
                  'synthetic_control','ToeSegmentation1',
                  'ToeSegmentation2','Trace','TwoLeadECG',
                  'Two_Patterns','UWaveGestureLibraryAll',
                  'uWaveGestureLibrary_X','uWaveGestureLibrary_Y',
                  'uWaveGestureLibrary_Z','wafer','Wine',
                  'WordsSynonyms','Worms','WormsTwoClass','yoga']

x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
z = [0, 1, 1, 2, 3, 4, 5, 3, 2, 1]
r = [1, 1, 2, 2, 3, 3, 4, 4, 3, 3]
s = [0, 1, 1, 2, 2, 2, 3, 3, 3, 1]
t = [0, 2, 4, 4, 5, 4, 3, 2, 1, 0]
u = [1, 0, 1, 2, 2, 3, 3, 2, 2, 1]
v = [2, 2, 3, 4, 5, 6, 5, 4, 3, 2]
w = [1, 1, 2, 3, 4, 4, 3, 3, 2, 1]
a = [0, 2, 3, 5, 6, 6, 4, 2, 1, 0]
tseries = []
tseries.append(x)
tseries.append(y)
tseries.append(z)
tseries.append(r)
tseries.append(s)
tseries.append(t)
tseries.append(u)
tseries.append(v)
tseries.append(w)
tseries.append(a)
tseries = np.array(tseries)

# x = [0, 1, 2, 2, 2, 1]
# y = [1, 1, 2, 3, 1, 0]
# tseries = []
# tseries.append(x)
# tseries.append(y)
# tseries = np.array(tseries)


if __name__ == '__main__':
    x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
    #x = np.array(x)
    y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
    #y = np.array(y)
    window_size = 1.0
    #x = [1,2,3,4,5,5,5,4]
    #y = [3,4,5,5,5,4]
    D_original, D_calculated, dist, path = dynamic_time_warping(x, y)
    print(D_calculated)
    print(path)