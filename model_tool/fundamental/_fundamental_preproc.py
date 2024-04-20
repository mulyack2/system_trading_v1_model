import pandas as pd


class FundamentalPreproc:
    def __init__(self, fundamental_df) -> None:
        self.df = fundamental_df

    def __call__(self):
        df = self.df.copy()
        df["reprt_code"] = df["reprt_code"].astype(str)
        df["reprt_date"] = pd.to_datetime(df["reprt_date"])
        df["stock_code"] = df["stock_code"].apply(lambda x: str(x).zfill(6))
        return df


class Fundamental2PivotTb:
    def __init__(self, fundamental_df) -> None:
        self.df = fundamental_df
        
    @staticmethod
    def make_pivot_tb(df):
        pivot_tb = df.pivot(
            columns=["account_nm"],
            values=["thstrm_amount"],
            index=["stock_code", "reprt_year", "reprt_date", "reprt_code"],
        )
        pivot_tb.columns = pivot_tb.columns.get_level_values(1)
        pivot_tb = pivot_tb.reset_index()
        return pivot_tb
    
    @staticmethod
    def slice_using_columns(df):
        using_columns = [
            "stock_code", "reprt_date", "reprt_year", "reprt_code",
            "당기순이익", "매출액", "영업이익", "자산총계", "부채총계", "비유동자산",
            "비유동부채","유동자산","유동부채"
        ]
        return df.loc[:,using_columns]
    def __call__(self):
        df = self.df.copy()
        pivot_tb = self.make_pivot_tb(df)
        pivot_tb = self.slice_using_columns(pivot_tb)
        return pivot_tb
    
    
class PivotTbController:
    def __init__(self, pivot_tb) -> None:
        self.df = pivot_tb

    def get_latest_df(self):
        df = self.df.copy()
        reprt_code_sort_dict = {"11013": 0, "11012": 1, "11014": 2, "11011": 3}
        df["reprt_sort"] = df["reprt_code"].map(reprt_code_sort_dict)
        latest_df = (
            df.sort_values(
                by=["reprt_year", "reprt_sort"], ascending=[True, True]
            )
            .groupby("stock_code")
            .tail(1)
        )
        return latest_df

    def get_latest_n_df(self, n):
        df = self.df.copy()
        latest_df = self.get_latest_df()
        stock_latest_reprt = latest_df.set_index("stock_code")[
            "reprt_code"
        ].to_dict()
        df["latest_reprt"] = df["stock_code"].map(stock_latest_reprt)
        _df = df[df["reprt_code"] == df["latest_reprt"]]
        _df = self.__filter_size(_df, n)
        latest_n_df = (
            _df.sort_values("reprt_year").groupby("stock_code").tail(n)
        )
        return latest_n_df

    def get_latest_reprt(self, reprt_code):
        df = self.df.copy()
        latest_reprt_df = (
            df[df["reprt_code"] == reprt_code]
            .sort_values("reprt_year")
            .groupby("stock_code")
            .tail(1)
        )
        return latest_reprt_df

    def get_latest_n_reprt(self, reprt_code, n):
        df = self.df.copy()
        _df = df[df["reprt_code"] == reprt_code]
        _df = self.__filter_size(_df, n)
        latest_n_reprt_df = (
            _df.sort_values("reprt_year").groupby("stock_code").tail(n)
        )
        return latest_n_reprt_df

    @staticmethod
    def __filter_size(df, n):
        _result = df.groupby("stock_code").size() > n
        using_stock_codes = _result[_result].index
        _df = df[df["stock_code"].isin(using_stock_codes)]

        return _df
    

