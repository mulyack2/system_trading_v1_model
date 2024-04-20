import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class ReprtDfCoefModel:
    def __init__(self, n_reprt_df) -> None:
        self.df = n_reprt_df

    @staticmethod
    def get_linear_coef(values):
        try:
            lr = LinearRegression()
            x = np.arange(1, len(values) + 1).reshape(-1, 1)
            y = np.array(values).reshape(-1, 1)
            lr.fit(x, y)
            return lr.coef_[0][0]
        except:
            return None

    def get_coef_df(self, columns):
        df = self.df.copy()
        coef_df = pd.concat(
            [
                df.groupby("stock_code")[col].apply(
                    lambda x: self.get_linear_coef(x)
                )
                for col in columns
            ],
            axis=1,
        )
        return coef_df
