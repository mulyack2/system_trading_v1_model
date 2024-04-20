import pandas as pd


class OhlcvPreproc:
    def __init__(self, kospi_ohlcv_df, kosdaq_ohlcv_df) -> None:
        self.kospi_df = kospi_ohlcv_df
        self.kosdaq_df = kosdaq_ohlcv_df

    def __call__(self):
        ohlcv_df = pd.concat([self.kospi_df, self.kosdaq_df], axis=0)
        ohlcv_df["stock_code"] = ohlcv_df["stock_code"].apply(
            lambda x: str(x).zfill(6)
        )
        ohlcv_df["date"] = pd.to_datetime(ohlcv_df["date"])
        return ohlcv_df


class OhlcvRoller:
    def __init__(self, ohlcv_df) -> None:
        self.ohlcv_df = ohlcv_df

    def front_rolling_mean(self, n):
        ohlcv_df = self.ohlcv_df.copy()
        ohlcv_df["close_shift"] = ohlcv_df.groupby("stock_code")[
            "close"
        ].shift(-n)
        front_rolling_price = (
            ohlcv_df.set_index("date")
            .groupby("stock_code")["close_shift"]
            .rolling(n)
            .mean()
            .reset_index()
            .dropna()
        )
        return front_rolling_price.rename(columns={"close_shift": "price"})

    def backward_rolling_mean(self, n):
        ohlcv_df = self.ohlcv_df.copy()
        backward_rolling_price = (
            ohlcv_df.set_index("date")
            .groupby("stock_code")["close"]
            .rolling(n)
            .mean()
            .reset_index()
            .dropna()
        )
        return backward_rolling_price.rename(columns={"close": "price"})
