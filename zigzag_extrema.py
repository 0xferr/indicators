from collections import namedtuple

import numpy as np
from scipy.signal import argrelextrema

from jesse.helpers import np_ffill, slice_candles

EXTREMA = namedtuple('EXTREMA', ['zigzag', 'line', 'is_min', 'is_max', 'last_min', 'last_max', 'min_ndx', 'max_ndx', 'is_last_high'])

def del_connected(arr, order):
    idx_to_del = []
    for i in range(1,len(arr)):
        if arr[i]-arr[i-1] <= order:
            idx_to_del.append(i)
    return np.delete(arr,idx_to_del)

def zigzag(candles: np.ndarray, order: int = 3, sequential: bool = False) -> EXTREMA:
    """
    minmax - Get extrema
    :param candles: np.ndarray
    :param order: int - default = 3
    :param sequential: bool - default: False
    :return: EXTREMA(zigzag, zigzag_line, is_min, is_max, last_min, last_max, min_ndx, max_ndx, is_last_high)
    """
    candles = slice_candles(candles, sequential)

    low = candles[:, 4]
    high = candles[:, 3]

    min_ndx = del_connected(argrelextrema(low, np.less_equal, order=order, axis=0)[0],order)
    max_ndx = del_connected(argrelextrema(high, np.greater_equal, order=order, axis=0)[0],order)

    is_min = np.full_like(low, np.nan)
    is_max = np.full_like(high, np.nan)

    # set the extremas with the matching price
    is_min[min_ndx] = low[min_ndx]
    is_max[max_ndx] = high[max_ndx]
    #correcting last value error
    is_min[-order:] = np.nan
    is_max[-order:] = np.nan
    # forward fill Nan values to get the last extrema
    last_min = np_ffill(is_min)
    last_max = np_ffill(is_max)

    zz = np.full_like(is_max, np.nan)
    z = 0
    real_zz = {'min':[], 'max':[]}
    if (max_ndx[0] <= min_ndx[0]):
        start =  max_ndx[0]
        real_zz['min'].append(low[0])
        z = low[0]
        is_last_high = False
    else:
        start =  min_ndx[0]
        is_last_high = True

    for i in range(start,len(is_max)):
        mx = is_max[i]
        if not np.isnan(mx):
            if not is_last_high:
                if len(real_zz['min']) > 1:
                    if real_zz['min'][-1] > real_zz['min'][-2]:
                        if (mx > real_zz['max'][-1] or 
                            (len(real_zz['max']) > 1 and real_zz['max'][-1] > real_zz['max'][-2])):
                            real_zz['max'].append(mx)
                            z = mx
                            is_last_high = True
                    else:
                        real_zz['max'].append(mx)
                        z = mx
                        is_last_high = True
                else:
                    real_zz['max'].append(mx)
                    z = mx
                    is_last_high = True
                    
            else:
                if mx > real_zz['max'][-1]:
                    real_zz['max'][-1] = mx
                    z = mx
                    
        mn = is_min[i]
        if not np.isnan(mn):
            if is_last_high:
                if len(real_zz['max']) > 1:
                    if real_zz['max'][-1] < real_zz['max'][-2]:
                        if (mn < real_zz['min'][-1] or 
                            len(real_zz['min']) > 1 and real_zz['min'][-1] < real_zz['min'][-2]):
                            real_zz['min'].append(mn)
                            is_last_high = False
                            z = mn
                    else:
                        real_zz['min'].append(mn)
                        z = mn
                        is_last_high = False
                else:
                    real_zz['min'].append(mn)
                    z = mn
                    is_last_high = False
                    
            else:
                if mn < real_zz['min'][-1]:
                    real_zz['min'][-1] = mn
                    z = mn
        zz[i] = z

    ZIGZAG = namedtuple('ZIGZAG', ['min', 'max'])
    zz_tuple = ZIGZAG(real_zz['min'],real_zz['max'])

    if sequential:
        return EXTREMA(zz_tuple, zz, is_min, is_max, last_min, last_max, min_ndx, max_ndx, is_last_high)
    else:
        zz_non_seq = ZIGZAG(real_zz['min'][-1],real_zz['max'][-1])
        return EXTREMA(zz_non_seq, zz[-1], is_min[-(order+1)], is_max[-(order+1)], last_min[-1], last_max[-1], min_ndx[-1], max_ndx[-1], is_last_high)
    