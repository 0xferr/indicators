from __future__ import annotations
import numpy as np
from collections import namedtuple
from datetime import datetime

Point = namedtuple("Point", ["time", "price"])


class Line(object):
    def __init__(self, isUp: bool, start: Point, end: Point) -> None:
        self.start = start
        self.end = end
        self.isUp = isUp
        self.structure = None

    def copy(self):
        return Line(self.isUp, self.start, self.end)

    def high(self) -> float:
        """Returns high price of line"""
        if self.isUp:
            return self.end.price
        else:
            return self.start.price

    def low(self) -> float:
        """Returns low price of line"""
        if self.isUp:
            return self.start.price
        else:
            return self.end.price

    def len_price(self) -> float:
        """Returns price lenght of line (the height)"""
        return self.end.price - self.start.price

    def len_time(self) -> datetime:
        """Returns time lenght of line (the lenght on X time axis)"""
        # to add converting to spec TF
        return self.end.time - self.start.time

    def update(self, point: Point) -> None:
        """Replays end point of line"""
        self.end = point

    def set_inner_structure(self, arr: list[Line]) -> None:
        self.structure = arr.copy()

    def __repr__(self):
        return f"{type(self).__name__}(start={self.start}, end={self.end}, isUp = {self.isUp})"


class ZigZag(object):
    def __init__(self, max_dev: float = 0.5, maintain_size: int = 100) -> None:
        self.max_dev = max_dev / 100
        self.maintain_size = maintain_size
        self.lines = []

    #         0    1    2    3    4    5
    # src  [time/open/close/high/low//volume]
    def construct(self, src: np.ndarray) -> None:
        """Constructing Zigzag from source array"""
        # construct first line manually
        if src[1, 2] > src[0, 1]:
            start = Point(int(src[0, 0]), src[0, 4])
            end = Point(int(src[1, 0]), src[1, 3])
            isUp = True
        else:
            start = Point(int(src[0, 0]), src[0, 3])
            end = Point(int(src[1, 0]), src[1, 4])
            isUp = False
        self.lines.append(Line(isUp, start, end))

        for candle in src[2:, :]:
            self.search_pivot(candle)
        if self.maintain_size > 0:
            self.maintaince()

    def update(self, candle: np.ndarray) -> bool:
        """Import new candle and update ZigZag"""
        if self.maintain_size > 0:
            self.maintaince()
        return self.search_pivot(candle)

    def search_pivot(self, candle: np.ndarray) -> None:
        """Search for new Pivots and update if found"""

        #         0    1    2    3    4    5
        # src  [time/open/close/high/low//volume]
        high = candle[3]
        low = candle[4]
        time = int(candle[0])
        isBullCandle = candle[2] > candle[1]
        last = self.lines[-1]
        if isBullCandle != last.isUp:
            if last.isUp:
                if high > last.high():
                    last.update(Point(time, high))
                last = self.lines[-1]
                if (last.high() - low) / last.high() > self.max_dev:
                    new_start = last.end
                    new_end = Point(time, low)
                    new_line = Line(False, new_start, new_end)
                    self.lines.append(new_line)
            else:
                if low < last.low():
                    last.update(Point(time, low))
                last = self.lines[-1]
                if (high - last.low()) / high > self.max_dev:
                    new_start = last.end
                    new_end = Point(time, high)
                    new_line = Line(True, new_start, new_end)
                    self.lines.append(new_line)
        else:
            if last.isUp:
                if (last.high() - low) / last.high() > self.max_dev:
                    new_start = last.end
                    new_end = Point(time, low)
                    new_line = Line(False, new_start, new_end)
                    self.lines.append(new_line)
                    last = self.lines[-1]
                    if (high - last.low()) / high > self.max_dev:
                        new_start = last.end
                        new_end = Point(time, high)
                        new_line = Line(True, new_start, new_end)
                        self.lines.append(new_line)
                elif high > last.high():
                    last.update(Point(time, high))

            else:
                if (high - last.low()) / high > self.max_dev:
                    new_start = last.end
                    new_end = Point(time, high)
                    new_line = Line(True, new_start, new_end)
                    self.lines.append(new_line)
                    last = self.lines[-1]
                    if (last.high() - low) / last.high() > self.max_dev:
                        new_start = last.end
                        new_end = Point(time, low)
                        new_line = Line(False, new_start, new_end)
                        self.lines.append(new_line)
                elif low < last.low():
                    last.update(Point(time, low))

    def maintaince(self):
        if len(self.lines) > self.maintain_size:
            self.lines = self.lines[-self.maintain_size :]

    ###     research methods
    def last(self) -> Line:
        """returns last line"""
        return self.lines[-1]

    def get_highs(self, deep: int) -> list[Point]:
        """Returns list of last highs"""
        if len(self.lines) < deep * 2:
            return None
        if self.last().isUp:
            return [self.lines[-i].end for i in range(-1 + 2 * deep, 0, -2)]
        else:
            return [self.lines[-i].start for i in range(-1 + 2 * deep, 0, -2)]

    def get_lows(self, deep: int) -> list[Point]:
        """returns list of last lows"""
        if len(self.lines) < deep * 2:
            return None
        if self.last().isUp:
            return [self.lines[-i].start for i in range(-1 + 2 * deep, 0, -2)]
        else:
            return [self.lines[-i].end for i in range(-1 + 2 * deep, 0, -2)]

    def get_lines_in_date_range(self, start: int, end: int = None) -> list(Line):
        """returns lines for period"""
        start_ndx = 0
        end_ndx = 0
        length = len(self.lines)

        # Non optimal search is used because in most cases this method will be used to find several lines in the end of list
        for i in range(1, length):
            if (end is not None) and (not end_ndx):
                if self.lines[-i].end.time <= end:
                    end_ndx = length - i + 1
            if self.lines[-i].start.time < start:
                start_ndx = -i + 1
                break
        if end_ndx == 0 or end_ndx == length:
            return self.lines[start_ndx:]
        else:
            return self.lines[start_ndx:end_ndx]
