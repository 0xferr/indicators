from numba import njit
import numpy as np


@njit
def rsi_calc(up, dn, diff, period):
    """
    calculates valued needed for RSI using numba.
    """
    up_np = np.array(up)
    dn_np = np.array(dn)
    for i in range(len(diff) - period):
        up = np.append(
            up_np, (up_np[-1] * (period - 1) + max(diff[period + i], 0.0)) / period
        )
        dn = np.append(
            dn_np, (dn_np[-1] * (period - 1) + max(-diff[period + i], 0.0)) / period
        )
    rsi = np.empty((period,)).fill(np.nan)
    rsi = np.append(rsi, 100.0 * up_np / (up_np + dn_np))
    return (rsi, up_np[-1], dn_np[-1])


class RSI(object):
    """Classic RSI. The calculations for adding each new value are based on previous calcs. So Comlexity is O(1) instead of O(n)
    Input:
        src: price
        period: rsi period length
        maintain_size: limits the max lenght of stored RSI values

    'add' method takes a new price, and calculate new RSI value using previous calculations.
    'target_rsi' This method takes a possible RSI value as an argument,
     and returns the price at which the RSI will reach the specified value
    """

    def __init__(self, src: np.ndarray, period: int = 14, maintain_size: int = 100):
        # limiting source lenght to reasonable value
        if maintain_size > 0:
            history_depth = maintain_size + period * 2
            if len(src) > history_depth:
                src = src[-history_depth:]
        diff = np.subtract(src[1:], src[:-1])
        # storing last price
        self.prev_value = src[-1]
        self.period = period
        # calculation
        zeroes = np.zeros(period)
        self.up = [np.sum(np.maximum(diff[:period], zeroes)) / period]
        self.dn = [np.sum(np.maximum(-diff[:period], zeroes)) / period]
        self.rsi, self.up, self.dn = rsi_calc(self.up, self.dn, diff, period)
        if maintain_size > 0:
            self.rsi = self.rsi[-maintain_size:]
            self.maintain = True

    def add(self, value: float) -> float:
        """
        It takes a new price and add a new RSI value to self.rsi
        """
        self.up = (
            self.up * (self.period - 1) + max(value - self.prev_value, 0.0)
        ) / self.period
        self.dn = (
            self.dn * (self.period - 1) + max(self.prev_value - value, 0.0)
        ) / self.period
        self.rsi = np.append(self.rsi, 100.0 * self.up / (self.up + self.dn))
        self.prev_value = value
        if self.maintain:
            self.rsi = self.rsi[1:]

    def target_rsi(self, target_rsi: float) -> float:
        """
        Takes possible RSI value as an argument and returns the price at which the RSI will reach the specified value
        """
        # check if rsi value is correct
        if target_rsi >= 100 or target_rsi <= 0:
            return None
        x = (self.period - 1) * (self.dn * target_rsi / (100.0 - target_rsi) - self.up)
        if x >= 0:
            return self.prev_value + x
        else:
            return self.prev_value + x * (100 - target_rsi) / target_rsi
