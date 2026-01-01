"""
PANDORA 

DACL, missing label inference

authors: {svc, lucy}@dmi.uns.ac.rs
"""

from abc import ABC, abstractmethod
from datatable import DataTable

class MLI(ABC):
    def __init__(self,
        data_table:DataTable,
        ground_truth:DataTable = None,
        verbose=False
        ):
        
        self.data_table = data_table
        self.ground_truth = ground_truth
        self.verbose = verbose
        #TODO: dopuniti
        #target vars in datatable
        #target vars data in datatable
        
    @abstractmethod
    def impute_label(self, target_var):
        pass
    
    def save(self, filename):
        pass
    
    def impute(self):
        pass
        
        