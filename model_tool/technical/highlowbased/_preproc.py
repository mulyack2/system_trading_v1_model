import numpy as np
import pandas as pd
from random import choices


class HighLowBasedPositionGenerator:
    """
    High / Low based Position Generator
    """

    def _get_raw_position(
        self,
        prices_li_df: pd.Series,
        idx: int,
        time_size: int,
        position_size: int,
    ) -> list:
        _prices_li_df = prices_li_df.iloc[max(0, idx - time_size) : idx].copy()
        _position_arr = np.concatenate(_prices_li_df.values)
        _position = self._sample_positions(_position_arr, position_size)
        return _position

    def get_raw_position(
        self,
        highs: pd.Series,
        lows: pd.Series,
        time_size: int,
        position_size: int,
    ) -> list:
        hl_df = pd.concat([highs.rename("high"), lows.rename("low")], axis=1)
        prices_li_df = hl_df.apply(
            lambda x: self._get_high_low_samples(x["high"], x["low"], 1),
            axis=1,
        )
        idx = len(prices_li_df)
        _positions = self._get_raw_position(
            prices_li_df, idx, time_size, position_size
        )
        return _positions

    def get_raw_position_df(
        self,
        highs: pd.Series,
        lows: pd.Series,
        time_size: int,
        position_size: int,
    ) -> pd.DataFrame:
        positions = list()
        hl_df = pd.concat([highs.rename("high"), lows.rename("low")], axis=1)
        prices_li_df = hl_df.apply(
            lambda x: self._get_high_low_samples(x["high"], x["low"], 1),
            axis=1,
        )
        for idx in range(1, len(prices_li_df) + 1):
            _positions = self._get_raw_position(
                prices_li_df, idx, time_size, position_size
            )
            positions.append([_positions])
        position_df = pd.DataFrame(
            positions, columns=["positions"], index=highs.index
        )
        return position_df

    def _get_raw_volume_position(
        self,
        hlv_df: pd.DataFrame,
        idx: int,
        time_size: int,
        position_size: int,
    ) -> list:
        _hlv_df = hlv_df.iloc[max(0, idx - time_size) : idx, :].copy()
        _hlv_df["normalized_volume"] = _hlv_df["volume"].apply(
            lambda x: round((x * position_size) / _hlv_df["volume"].sum())
        )
        _position_li_df = _hlv_df.apply(
            lambda x: self._get_high_low_samples(
                x["high"], x["low"], x["normalized_volume"].astype(int)
            ),
            axis=1,
        )
        _positions_arr = np.concatenate(_position_li_df.values)
        _positions = self._sample_positions(_positions_arr, position_size)
        return _positions

    def get_raw_volume_position(
        self,
        highs: pd.Series,
        lows: pd.Series,
        volumes: pd.Series,
        time_size: int,
        position_size: int,
    ) -> list:
        hlv_df = pd.concat(
            [
                highs.rename("high"),
                lows.rename("low"),
                volumes.rename("volume"),
            ],
            axis=1,
        )
        idx = len(hlv_df)
        _position = self._get_raw_volume_position(
            hlv_df, idx, time_size, position_size
        )
        return _position

    def get_raw_volume_position_df(
        self,
        highs: pd.Series,
        lows: pd.Series,
        volumes: pd.Series,
        time_size: int,
        position_size: int,
    ) -> pd.DataFrame:
        positions = list()
        hlv_df = pd.concat(
            [
                highs.rename("high"),
                lows.rename("low"),
                volumes.rename("volume"),
            ],
            axis=1,
        )
        for idx in range(1, len(hlv_df) + 1):
            _position = self._get_raw_volume_position(
                hlv_df, idx, time_size, position_size
            )
            positions.append([_position])
        position_df = pd.DataFrame(
            positions, columns=["positions"], index=highs.index
        )
        return position_df

    def _get_time_dependent_position(
        self, hl_df: pd.DataFrame, idx: int, time_size: int, position_size: int
    ) -> list:
        _hl_df = hl_df.iloc[max(0, idx - time_size) : idx, :].copy()
        _hl_df["time"] = list(range(1, len(_hl_df) + 1))
        _position_li_df = _hl_df.apply(
            lambda x: self._get_high_low_samples(
                x["high"], x["low"], x["time"]
            ),
            axis=1,
        )
        _positions_arr = np.concatenate(_position_li_df.values)
        _positions = self._sample_positions(_positions_arr, position_size)
        return _positions

    def get_time_dependent_position(
        self,
        highs: pd.Series,
        lows: pd.Series,
        time_size: int,
        position_size: int,
    ) -> list:
        hl_df = pd.concat([highs.rename("high"), lows.rename("low")], axis=1)
        idx = len(hl_df)
        _positions = self._get_time_dependent_position(
            hl_df, idx, time_size, position_size
        )
        return _positions

    def get_time_dependent_position_df(
        self,
        highs: pd.Series,
        lows: pd.Series,
        time_size: int,
        position_size: int,
    ) -> pd.DataFrame:
        positions = list()
        hl_df = pd.concat([highs.rename("high"), lows.rename("low")], axis=1)
        for idx in range(1, len(hl_df) + 1):
            _positions = self._get_time_dependent_position(
                hl_df, idx, time_size, position_size
            )
            positions.append([_positions])
        position_df = pd.DataFrame(
            positions, columns=["positions"], index=highs.index
        )
        return position_df

    def _get_time_dependent_volume_position(
        self,
        hlv_df: pd.DataFrame,
        idx: int,
        time_size: int,
        position_size: int,
    ) -> list:
        _hlv_df = hlv_df.iloc[max(0, idx - time_size) : idx, :].copy()
        _hlv_df["time"] = list(range(1, len(_hlv_df) + 1))
        _hlv_df["normalized_volume"] = (
            _hlv_df["volume"] / _hlv_df["volume"].sum()
        )
        _hlv_df["weight"] = _hlv_df["time"] * _hlv_df["normalized_volume"]
        _hlv_df["normalized_weight"] = _hlv_df["weight"].apply(
            lambda x: round((x * position_size) / _hlv_df["weight"].sum())
        )
        _position_li_df = _hlv_df.apply(
            lambda x: self._get_high_low_samples(
                x["high"], x["low"], x["normalized_weight"].astype(int)
            ),
            axis=1,
        )
        _positions_arr = np.concatenate(_position_li_df.values)
        _positions = self._sample_positions(_positions_arr, position_size)
        return _positions

    def get_time_dependent_volume_position(
        self,
        highs: pd.Series,
        lows: pd.Series,
        volumes: pd.Series,
        time_size: int,
        position_size: int,
    ) -> list:
        hlv_df = pd.concat(
            [
                highs.rename("high"),
                lows.rename("low"),
                volumes.rename("volume"),
            ],
            axis=1,
        )
        idx = len(hlv_df)
        _positions = self._get_time_dependent_volume_position(
            hlv_df, idx, time_size, position_size
        )
        return _positions

    def get_time_dependent_volume_position_df(
        self,
        highs: pd.Series,
        lows: pd.Series,
        volumes: pd.Series,
        time_size: int,
        position_size: int,
    ) -> pd.DataFrame:
        positions = list()
        hlv_df = pd.concat(
            [
                highs.rename("high"),
                lows.rename("low"),
                volumes.rename("volume"),
            ],
            axis=1,
        )
        for idx in range(1, len(hlv_df) + 1):
            _positions = self._get_time_dependent_volume_position(
                hlv_df, idx, time_size, position_size
            )
            positions.append([_positions])
        position_df = pd.DataFrame(
            positions, columns=["positions"], index=highs.index
        )
        return position_df

    @staticmethod
    def _get_high_low_samples(high, low, n):
        mean = (high + low) / 2
        var = (high - low) / 6
        samples = np.random.normal(loc=mean, scale=var, size=n).tolist()
        return samples

    @staticmethod
    def _calc_time_weight_arr(series):
        time_weight_arr = np.arange(1, len(series) + 1)
        return time_weight_arr

    @staticmethod
    def _sample_positions(raw_positions, position_size):
        positions = choices(raw_positions, k=position_size)
        return positions
