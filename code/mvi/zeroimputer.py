"""
PANDORA 

Missing Value Zero Imputer -- for selected columns impute zero value if missing

Author
----------
lucy@dmi.uns.ac.rs
"""


from imputer import Imputer
import pandas as pd
from pandas import DataFrame

class ZeroImputer(Imputer):
    def __init__(self, data:DataFrame, columns:list):
        super().__init__()
        self.data = data
        self.columns = columns

    def impute(self):
        for col in self.columns:
            row = self.data[col]
            for i in range(0, len(row)):
                if pd.isna(row[i]):
                    self.data.at[i, col] = 0 

        return self.data        
