from collections import namedtuple

import numpy as np
import talib
from typing import Union
from math import sqrt

lrch = namedtuple("lrch", ["lr", "std"])


def lr_channel(
    candles: np.ndarray, period: int = 100, sequential: bool = False
) -> lrch:
    """
    LINEARREG - Linear Regression channel
    It calculate Linear Regression and Standard Deviation for it. (Classic LR channel is LR +/- 2*σ)

    :param candles: np.ndarray (1d)
    :param period: int - default: 100
    :param sequential: bool - default: False

    :return: named_tuple of np.ndarray | named_tuple of float in case if not sequential
    """

    def std_for_lr(price, slope, intercept, n=96):
        if len(price) < n:
            return np.nan
        else:
            predict = [(slope * i + intercept) for i in range(n)]
            return sqrt(np.sum((price - predict) ** 2) / n)

    if sequential:
        intercept = talib.LINEARREG_INTERCEPT(source, timeperiod=period)
        slope = talib.LINEARREG_SLOPE(source, timeperiod=period)
        lr = slope * (period - 1) + intercept
        std = np.array(
            [
                std_for_lr(
                    source[-len(source) + i + 1 - period : -len(source) + i + 1],
                    slope[-len(source) + i],
                    intercept[-len(source) + i],
                    period,
                )
                for i in range(len(source) - 1)
            ]
        )
        std = np.append(
            std, std_for_lr(source[-period:], slope[-1], intercept[-1], period)
        )
        return lrch(lr, std)
    else:
        source = source[-period:]
        intercept = talib.LINEARREG_INTERCEPT(source, timeperiod=period)[-1]
        slope = talib.LINEARREG_SLOPE(source, timeperiod=period)[-1]
        lr = slope * (period - 1) + intercept
        std = std_for_lr(source, slope, intercept, period)
        return lrch(lr, std)
