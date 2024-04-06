import pandas as pd
import yfinance as yf
from datetime import timedelta


class BaseExtractor:
    def __init__(self, name: str):
        self.name = name

    def get_data(self):
        raise NotImplementedError("Subclasses must implement the 'get_data' method")


class ExternalFactorsExtractor(BaseExtractor):
    def __init__(self, length: str = '5y'):
        super().__init__("ExternalFactors") 
        self.length = length
        self.tickers = {
            'Crude Oil Futures': 'CL=F',
            'Allianz China A-Shares': '0P0001IL1V.SI',
            'AXA Funds Management S.A.': 'AXAHF',
            'AXA WORLD FUNDS - US High': '0P0001IZQ5',
            'Allianz Global Investors Asia Pacific Limited': 'TW000T3608Y4.TW',
            'Allianz Global Investors GmbH': '0P00000FZF.F',
            'FinTech Fund': '0P0001EI7S',
            'BlackRock': '0P00000AWU',
            'EPM': 'EPM',
            'FTGF ClearBridge Global Infrastructure Income Fund':  '0P0001N5EJ.F',
            'Fidelity Funds': 'FFIDX',
            'Franklin Resources, Inc.': 'BEN',
            'Invesco': 'IE0030382026.IR',
            'JPMorgan': 'JPM',
            'Pictet': '0P00000LIQ.F',
            'SET': '^SET.BK',
            'gold': 'GLD',
            'Wellington Global Health': '0P0000K0MV',
            'Nomura Japan High Conviction Fund': '0P0000ZDXZ.T',
            
        }

    def load_data(self, code):
        return yf.download(code, period=self.length) 
    
    def get_data(self) -> pd.DataFrame:
        external_factors = {factor_name: self.load_data(code) for factor_name, code in self.tickers.items()}
        cols = []
        date_range = pd.DataFrame(pd.date_range(start='2010-01-01', end='2025-01-01'), columns=['date'])
        all_df = date_range.copy()
        for key, data in external_factors.items():
            cols.append(key)
            data = data.reset_index().rename(columns={'Date': 'date', "Close":key})
            data = date_range.merge(data, on='date', how='left')
            data = data.fillna(-1)
            
            all_df = pd.concat([all_df, data[key]], axis=1)
        return all_df
