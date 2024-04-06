import pandas as pd
from typing import Any, Union
from datetime import datetime, timedelta
from pytorch_forecasting import TimeSeriesDataSet
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

class BaseProcessor:
    def __init__(self):
        pass

    def transform(self, x: Any) -> Any:
        raise NotImplementedError("Subclasses must implement the 'get_data' method")


class ExternalFactor(BaseProcessor):
    def __init__(self):
        pass

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x.columns = x.columns.str.replace('.', '').str.replace(',', "")
        x.date = pd.to_datetime(x.date)
        return x
    

class Date2IntConvertor(BaseProcessor):
    def __init__(
            self, 
            date_col: str, 
            group_cols: Union[str, None] = None, 
            new: bool = False
    ):
        self.date_col = date_col
        self.group_cols = group_cols
        self.new = new

    def transform(self, x):
        container = []
        if self.group_cols != None:
            for _, group in x.groupby(self.group_cols):
                group = group.sort_values(self.date_col)
                if self.new:
                    group[f'{self.date_col}_int'] = range(0, group.date.nunique())
                else:
                    group[self.date_col] = range(0, group.date.nunique())
                container.append(group)
        else:
            if self.new:
                x[f'{self.date_col}_int'] = range(0, x.date.nunique())
            else:
                x[self.date_col] = range(0, x.date.nunique())
        return pd.concat(container)
    

class DataFiller(BaseProcessor):
    def __init__(
            self,
            start_date: Union[datetime, str], 
            end_date: Union[datetime, str],
            date_col: str
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.date_col = date_col

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        # create Date Dimension
        date_range = pd.DataFrame(
            pd.date_range(start=self.start_date, end=self.end_date),
            columns=[self.date_col]
        )
        return date_range.merge(x, on=self.date_col, how='left')
    
class FinanceIndicatorFiller(BaseProcessor):
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={'fund_name': 'tic'})
        fund_data = df[['date', 'tic', 'nav/unit']]
        fund_data['open'] = fund_data['nav/unit']
        fund_data['high'] = fund_data['nav/unit']
        fund_data['low'] = fund_data['nav/unit']
        fund_data['close'] = fund_data['nav/unit']
        fund_data.drop(columns=['nav/unit'], inplace=True)
        fund_data['volume'] = 0 

        fe = FeatureEngineer(
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)
        fund_data = fe.preprocess_data(fund_data)
        return fund_data[['date', 'tic', 'close', 'macd',
       'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma',
       'close_60_sma']].rename(columns={'tic': 'fund_name', 'close': 'nav/unit'})


def timeseries_train_test_split(
        dataset_kwarg: dict,
        test_dataset_kwarg: dict = {},
        val: bool = True,
        batch_size: int = 64,
        test: bool = False,
        predict_mode: bool = False
    ):  
        if test:
            if (test_dataset_kwarg is None) or (len(test_dataset_kwarg) == 0):
                raise Exception('test_dataset_kwarg is empty')
        
        dataset = TimeSeriesDataSet(**dataset_kwarg)
        if test:
            test_dataset = TimeSeriesDataSet(**test_dataset_kwarg)
            test_dataloader = test_dataset.to_dataloader(train=False, batch_size=batch_size)
        
        if val:
            train_dataloader = dataset.to_dataloader(train=True, batch_size=batch_size)
            val_dataloader = dataset.to_dataloader(train=False, batch_size=batch_size)
            if test:
                return train_dataloader, val_dataloader, test_dataloader
            else:
                return train_dataloader, val_dataloader
        
        elif test:
            return dataset.to_dataloader(train=True, batch_size=batch_size), test_dataloader
            
        else:
            return dataset.to_dataloader(train=True, batch_size=batch_size)

def generate_dates(start_date, period):
    dates = []
    for i in range(period):
        new_date = start_date + timedelta(days=i)
        dates.append(new_date)
    return dates
        


    

