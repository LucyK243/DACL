"""
PANDORA 

Missing Value Imputers

Author
----------
lucy@dmi.uns.ac.rs
"""

from abc import ABC, abstractmethod
import numpy as np
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from hyperimpute.plugins.imputers import Imputers


class Imputer(ABC):
    
    @abstractmethod
    def impute(self):
        '''
        Returns dataset with imputed values
        '''
        pass
    
    def prepape_data(self, dataset:DataFrame, columns_to_ignore=None):
        '''
        Removes features that do not contain missing values or MVI does not apply
        If columns_to_remove == None returns copy of dataset
        '''
        dscopy = dataset.copy()
        if columns_to_ignore != None:
            for column in columns_to_ignore:
                dscopy.drop(column, axis=1, inplace=True)
                
        return dscopy    
    
    
    
class SimpleImputerDACL(Imputer):
    def __init__(self, dataset:DataFrame, columns_to_ignore=None, missing_values=np.nan, strategy='mean'):
        super().__init__()
        self.dataset = dataset
        self.columns_to_ignore = columns_to_ignore
        self.missing_values = missing_values
        self.strategy = strategy
        
    def impute(self):       
        dscopy = self.prepape_data(self.dataset, self.columns_to_ignore)
        imputer = SimpleImputer(missing_values=self.missing_values, strategy=self.strategy)
        dscopy = DataFrame(imputer.fit_transform(dscopy), columns=dscopy.columns)
        
        # Add back removed columns
        for column in self.dataset.columns:
            if column not in dscopy.columns:
                dscopy[column] = self.dataset[column]       
        
        return dscopy
    
    
class IterativeImputerDACL(Imputer):
    def __init__(self, dataset:DataFrame, estimator, max_iter=10, random_state=0, columns_to_ignore=None):
        self.dataset = dataset
        self.estimator = estimator
        self.max_iter = max_iter
        self.random_state = random_state
        self.columns_to_ignore = columns_to_ignore
        
    def impute(self):
        datacopy = self.prepape_data(self.dataset, self.columns_to_ignore)
        imputer = IterativeImputer(estimator=self.estimator, max_iter=self.max_iter, random_state=self.random_state)
        datacopy = DataFrame(imputer.fit_transform(datacopy), columns=datacopy.columns)

        # return removed columns
        for column in self.dataset.columns:
            if column not in datacopy.columns:
                datacopy[column] = self.dataset[column]
        
        return datacopy
    
    
class KNNImputerDACL(Imputer):
    def __init__(self, dataset:DataFrame, n=2, columns_to_ignore=None):
        self.dataset = dataset
        self.n = n
        self.columns_to_ignore = columns_to_ignore
        
    def impute(self):
        dscopy = self.prepape_data(self.dataset, self.columns_to_ignore)
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=self.n)
        dscopy = DataFrame(imputer.fit_transform(dscopy), columns=dscopy.columns)
        
        # return removed columns
        for column in self.dataset:
            if column not in dscopy.columns:
                dscopy[column] = self.dataset[column]
                
        return dscopy
        

class HyperimputeDACL(Imputer):
    def __init__(self, dataset:DataFrame, imputer:str, columns_to_ignore=None):
        '''
        imputer -- one value from: median, ice, most_frequent, gain, miwae,
                    sklearn_missforest, softimpute, mean, EM, hyperimpute, sinkhorn, missforest
        '''
        self.dataset = dataset
        self.imputer = imputer
        self.columns_to_ignore = columns_to_ignore

    def impute(self):
        dscopy = self.prepape_data(self.dataset, self.columns_to_ignore)
        imputer = Imputers().get(self.imputer)
        out = imputer.fit_transform(dscopy)
        out.columns = dscopy.columns
        dscopy = out
        
        
        # return removed columns
        for column in self.dataset:
            if column not in dscopy.columns:
                dscopy[column] = self.dataset[column]
                
        return dscopy
        