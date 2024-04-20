import numpy as np
from scipy.stats import gaussian_kde
from typing import Tuple
from typing import Union

Numeric = Union[int, float]


class PositionLimitModel:
    def calc(self, positions: list) -> Tuple[float, float]:
        upper_positions = self._get_upper_positions(positions)
        upper_density_point = self._get_max_density_point(upper_positions)

        lower_positions = self._get_lower_positions(positions)
        lower_density_point = self._get_max_density_point(lower_positions)

        return (lower_density_point, upper_density_point)

    @staticmethod
    def _get_upper_positions(positions: list) -> float:
        median_position = np.median(positions)
        upper_positions = [
            position for position in positions if position > median_position
        ]
        return upper_positions

    @staticmethod
    def _get_lower_positions(positions: list) -> float:
        median_position = np.median(positions)
        lower_positions = [
            position for position in positions if position < median_position
        ]
        return lower_positions

    @staticmethod
    def _get_max_density_point(numbers: list) -> float:
        kde = gaussian_kde(numbers)
        grid = np.linspace(min(numbers), max(numbers), 100)

        kde_values = kde.evaluate(grid)

        max_density_index = np.argmax(kde_values)
        max_density_point = round(grid[max_density_index], 2)
        return max_density_point


class PositionRankIndexModel:
    """
    0 : price is biggest in position
    1 : price is lowest in position
    """

    def calc(self, price: Numeric, positions: list):
        uppers = self._calc_uppers(price, positions)
        lowers = self._calc_lowers(price, positions)
        pri = round((uppers) / (uppers + lowers), 2)
        return pri

    @staticmethod
    def _calc_uppers(price: Numeric, positions: list):
        uppers = (np.array(positions) > price).sum()
        return uppers

    @staticmethod
    def _calc_lowers(price: Numeric, positions: list):
        lowers = (np.array(positions) < price).sum()
        return lowers
