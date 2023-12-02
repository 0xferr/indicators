import numpy as np
import talib
from collections import namedtuple
from math import sqrt
from scipy import stats
from scipy.signal import argrelextrema

lr_ind = namedtuple("LR_ind", ["std", "slope", "intercept"])
trend_params = namedtuple(
    "Trend_parameters", ["trend_is_up", "start", "stop", "value", "lr", "created"]
)


class LinearRegressionTrend(object):
    """
    Adaptive trend detetion using Linear Regression
    src: OHLC price data np.ndarray
    min_trend_length: the shortest period of trend
    max_trend_length: the longest period of trend
    width: width of trend in standard deviation
    max_deviation: trend is considered broken when price leaves max_deviation channel

    """

    def __init__(
        self,
        src: np.ndarray,
        min_trend_length: int = 20,
        max_trend_length: int = 100,
        width: int = 2,
        max_deviation: int = 3,
    ):
        # input
        self.src = src
        self.close = src[:, 2]
        self.width = width
        self.max_deviation = (
            max_deviation  # change trend if price deviate more than this value in {std}
        )
        self.min_trend_length = min_trend_length
        self.max_trend_length = max_trend_length
        self.limit_trend = True
        ##results
        self.trend_channel = None
        self.unconfirmed_trend_channel = None
        self.last_trend = None
        self.all_trends = []
        self.all_breakouts = None
        self.all_trend_resumes = None
        self.breakout = False
        # vars
        self._no_new_trend = False

    ### Intertal calculation metods

    def _std_for_lr(self, src, slope, intercept):
        """returns standard deviation of price from linear regression"""
        n = len(src)
        predict = [(slope * i + intercept) for i in range(n)]
        return sqrt(np.sum((src - predict) ** 2) / n)

    def _lin_reg(self, source: np.ndarray) -> lr_ind:
        """linear regression calculation using talib"""
        period = len(source)
        # check if talib gets wrong result
        if period > 1290:
            return self._lr_scipy(source)
        else:
            intercept = talib.LINEARREG_INTERCEPT(source, timeperiod=period)[-1]
            slope = talib.LINEARREG_SLOPE(source, timeperiod=period)[-1]
            std = self._std_for_lr(source, slope, intercept)
            return lr_ind(std, slope, intercept)

    def _lr_scipy(self, source: np.ndarray) -> lr_ind:
        """linear regression using scipy"""
        period = len(source)
        x = np.array(range(0, period))
        res = stats.linregress(x, source)
        std = self._std_for_lr(source, res.slope, res.intercept)
        return lr_ind(std, res.slope, res.intercept)

    ### Internal trend methods

    def _draw_next_trend_value(self, unconf=False):
        """calcs and paste new channel values"""
        if not unconf:
            if self.last_trend.stop == len(self.close) - 1:
                m = self.last_trend.lr.intercept + self.last_trend.lr.slope * (
                    self.last_trend.stop - self.last_trend.start + 1
                )
                s = self.last_trend.lr.std
            else:
                m = self.trend_channel[-1, 0] + self.last_trend.lr.slope
                s = self.trend_channel[-1, 1]
            self.trend_channel = np.vstack((self.trend_channel, [m, s]))
        else:
            m = self.last_trend.lr.intercept + self.last_trend.lr.slope * (
                self.last_trend.stop - self.last_trend.start + 1
            )
            s = self.last_trend.lr.std
            self.unconfirmed_trend_channel = np.vstack(
                (self.unconfirmed_trend_channel, [m, s])
            )

    def _recalc_trend(self, closes):
        """recalculate trend with new candles data"""
        lr = self._lin_reg(closes)
        stop = len(self.close)
        start = stop - len(closes)
        self.last_trend = trend_params(
            self.last_trend.trend_is_up,
            start,
            stop - 1,
            closes[-1],
            lr,
            self.last_trend.created,
        )
        if start == self.all_trends[-1].start:
            self.all_trends[-1] = self.last_trend
        else:
            self.all_trends.append(self.last_trend)

    def _get_new_start(self, closes, is_up) -> int:
        """looking for new starting point, used in cases where trend reaches max length limit"""
        if is_up:
            return np.argmin(closes)
            # return argrelextrema(closes, np.less_equal, order=5, axis=0)[0][0]
        else:
            return np.argmax(closes)
            # return argrelextrema(closes, np.greater_equal, order=5, axis=0)[0][0]

    def _add_candle(self, candle):
        """adds candle"""
        self.src = np.vstack((self.src, candle))
        self.close = np.append(self.close, candle[2])
        self.all_breakouts = np.append(self.all_breakouts, np.nan)
        self.all_trend_resumes = np.append(self.all_trend_resumes, np.nan)

    def _update(self, close):
        """redetect trend"""
        up = self.last_trend.trend_is_up
        start = self.last_trend.start
        closes = self.close[start:]

        # IF TREND IS TOO SHORT
        if self.last_trend.stop - start < self.min_trend_length:
            # invalidate trend and resume previous one if it breaks starting point
            if up and close < closes[0] or not up and close > closes[0]:
                self.all_trend_resumes[-1] = self.close[-1]
                self.all_trends.pop()
                self.last_trend = self.all_trends[-1]
                new_start = self.last_trend.start
                if len(self.close) - new_start > self.max_trend_length:
                    new_start = self._get_new_start(
                        self.close[-self.max_trend_length :], not up
                    )
                    closes = self.close[-self.max_trend_length + new_start :]
                else:
                    closes = self.close[new_start:]
            # if trend is too short, but didn't invalidate starting price - recalculate it
            # we need to do it for both cases
            self._recalc_trend(closes)

        else:
            m, u, l = self.next_lr(self.max_deviation)
            # IF PRICE INSIDE MAX_DEV
            if not (up and close < l or not up and close > u):
                # if price made new Extremum
                if (
                    up
                    and close > self.last_trend.value
                    or not up
                    and close < self.last_trend.value
                ):
                    if (
                        self.limit_trend
                        and len(self.close) - start > self.max_trend_length
                    ):
                        new_start = self._get_new_start(
                            closes[-self.max_trend_length :], up
                        )
                        closes = closes[-self.max_trend_length + new_start :]
                    self._recalc_trend(closes)
            # IF TREND IS BROKEN
            else:
                self.breakout = True
                self.all_breakouts[-1] = self.close[-1]
                if up:
                    new_start = start + np.argmax(closes)
                else:
                    new_start = start + np.argmin(closes)
                closes = self.close[new_start:]
                if len(closes) > 2:
                    lr = self._lin_reg(closes)
                    stop = len(self.close)
                    self.last_trend = trend_params(
                        not up, new_start, stop - 1, close, lr, len(self.close) - 1
                    )
                    self.all_trends.append(self.last_trend)
                else:
                    self._no_new_trend = True
        if len(self.close) - self.last_trend.start >= self.min_trend_length:
            self._draw_next_trend_value()
            self.unconfirmed_trend_channel = np.vstack(
                (self.unconfirmed_trend_channel, [np.nan, np.nan])
            )
        else:
            self.trend_channel = np.vstack((self.trend_channel, [np.nan, np.nan]))
            self._draw_next_trend_value(unconf=True)

    ### External methods
    def start(self):
        """Starts the trend detection"""
        close = self.close[: self.max_trend_length]
        minimum = np.argmin(close)
        maximum = np.argmax(close)
        if minimum < maximum:
            start = minimum
            close = self.close[start : start + self.max_trend_length]
            stop = np.argmax(close[self.min_trend_length :]) + self.min_trend_length
            trend_is_up = True
        else:
            start = maximum
            close = self.close[start : start + self.max_trend_length]
            stop = np.argmin(close[self.min_trend_length :]) + self.min_trend_length
            trend_is_up = False
        last_extremum_value = close[stop]
        self.trend_channel = np.empty((start + stop + 2, 2))
        self.trend_channel[:] = np.NaN
        self.unconfirmed_trend_channel = np.copy(self.trend_channel)
        self.all_breakouts = np.full_like(self.close[: start + stop + 1], np.nan)
        self.all_trend_resumes = np.full_like(self.close[: start + stop + 1], np.nan)
        lr = self._lin_reg(close[: stop + 1])
        for ndx in range(start, start + stop + 2):
            m = lr.intercept + lr.slope * (ndx - start)
            s = lr.std
            self.trend_channel[ndx] = m, s
        self.last_trend = trend_params(
            trend_is_up, start, start + stop, last_extremum_value, lr, start
        )
        self.all_trends.append(self.last_trend)
        rest_close_prices = self.close[start + stop + 1 :]
        self.close = self.close[: start + stop + 1]
        for price in rest_close_prices:
            self.close = np.append(self.close, price)
            self.all_breakouts = np.append(self.all_breakouts, np.nan)
            self.all_trend_resumes = np.append(self.all_trend_resumes, np.nan)
            self._update(price)

    def update(self, candle):
        """Add new canle. It runs trend update"""
        self.breakout = False
        self._no_new_trend = False
        self._add_candle(candle)
        self._update(candle[2])

    ###     ANALYSIS METHODS

    # calcs channel values or border condition
    def next_lr(self, width):
        """get channel values: middle, upper border, lower border"""
        m = self.trend_channel[-1, 0]
        l = m - self.trend_channel[-1, 1] * width
        u = m + self.trend_channel[-1, 1] * width
        return (m, u, l)

    @property
    def false_breakouts(self, show_previous=True) -> int:
        """Check trend for false breakouts"""
        if not self._no_new_trend:
            if not show_previous:
                matrix = self.all_breakouts[self.last_trend.created :]
                return len(matrix[~np.isnan(matrix)]) - 1
            else:
                matrix = self.all_breakouts[self.all_trends[-2].created :]
                return len(matrix[~np.isnan(matrix)]) - 1
        else:
            if show_previous:
                matrix = self.all_breakouts[self.last_trend.created :]
                return len(matrix[~np.isnan(matrix)]) - 1
            else:
                return 0

    @property
    def trend_life(self) -> int:
        """How many candles lasts current trend"""
        return len(self.close) - 1 - self.last_trend.created

    @property
    def speed(self) -> float:
        """Speed of trend"""
        return abs(self.last_trend.lr.std / self.last_trend.lr.slope)

    @property
    def up(self) -> bool:
        """If trend is up returns True"""
        if self._no_new_trend:
            return not self.last_trend.trend_is_up
        else:
            return self.last_trend.trend_is_up
