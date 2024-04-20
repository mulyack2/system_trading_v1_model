import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

lr = LinearRegression()
mms = MinMaxScaler()


class BirModel:
    """Base Interest Rate Model"""

    def __init__(self, us_bir_df, k_bir_df) -> None:
        self.df = self.__merge_df(us_bir_df, k_bir_df)

    @staticmethod
    def __merge_df(us_df, k_df):
        df = pd.merge(
            left=us_df,
            right=k_df,
            on="date",
            how="inner",
            suffixes=("_us", "_kr"),
        )
        return df

    def __call__(self, recent_n):
        df = self.df.copy()
        recent_df = df.sort_values("date").tail(recent_n)
        bir_score = recent_df["value_kr"].sum() / (
            recent_df["value_us"].sum() + recent_df["value_kr"].sum()
        )
        return round(bir_score, 2)


class MoneySupplyModel:
    def __init__(self, ms_df) -> None:
        self.ms_df = ms_df

    @staticmethod
    def get_linear_coef(values):
        x = np.arange(1, len(values) + 1).reshape(-1, 1)
        y = np.array(values).reshape(-1, 1)
        lr.fit(x, y)
        return lr.coef_[0]

    def __call__(self, window, recent_n):
        df = self.ms_df.copy()
        ms_coef_series = (
            df["value"].rolling(window=window).apply(self.get_linear_coef)
        )
        recent_ms_coef_series = ms_coef_series.tail(recent_n)
        ms_coef = mms.fit_transform(recent_ms_coef_series.to_frame())[-1]
        return ms_coef[0]
