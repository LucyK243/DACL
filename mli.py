"""
PANDORA 

DACL, missing label inference

authors: {svc, lucy}@dmi.uns.ac.rs
"""

from abc import ABC, abstractmethod
from datatable import DataTable
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score


class MLI(ABC):
    def __init__(self,
        data_table:DataTable,
        ground_truth:DataTable=None,
        verbose=False
    ):
        self.data_table = data_table
        self.ground_truth = ground_truth
        self.verbose = verbose
        
        
    @abstractmethod
    def impute_label(self, X, y):
        pass
    
    def save(self, filename):
        self.data_table.save(filename)
    
    def impute_target(self, X, y, ml_indices):
        y_vals = set()
        for i in range(len(y)):
            if i in ml_indices:
                y[i] = np.int64(-1)
            else:
                y_vals.add(y[i])
            
        y_vals = list(y_vals)
            
        for i in range(len(y)):
            if y[i] != -1:
                y_val = y[i]
                ind = y_vals.index(y_val)
                y[i] = np.int64(ind)
                    
        y = y.tolist()
        predicted_labels = self.impute_label(X, y)
        final_labels = [y_vals[label] for label in predicted_labels]
        return final_labels
        
    
    def impute(self):
        target_vars = self.data_table.target_vars
        missing_labels = self.data_table.missing_labels
                
        X = self.data_table.data_matrix
        targets = self.data_table.target_vars_data
        
        for target in target_vars:
            print("IMPUTING LABELS for", target)
            if target not in missing_labels:
                continue
            
            y = targets[target].to_numpy()
            ml_indices = missing_labels[target]
            
            final_labels = self.impute_target(X, y, ml_indices)
            
            # TODO: do whatever is necessary for saving final labels
            
            # ! evaluation
            if self.ground_truth != None:        
                gt = self.ground_truth.target_vars_data[target].to_list()
                predicted_labels = [final_labels[i] for i in ml_indices]
                true_labels = [gt[i] for i in ml_indices]
                
                acc = accuracy_score(true_labels, predicted_labels)
                print("Accuracy", acc)
                
                bacc = balanced_accuracy_score(true_labels, predicted_labels)
                print("Balanced accuracy", bacc)
                
                prec = precision_score(true_labels, predicted_labels, average='macro')
                print("Precision", prec)
                
                rec = recall_score(true_labels, predicted_labels, average='macro')
                print("Recall", rec)
                
                
                
                
            
            
                
            

from sklearn.semi_supervised import LabelSpreading            
        
class Dummy_MLI(MLI):
    def __init__(self,
        data_table:DataTable,
        ground_truth:DataTable = None,
        verbose=False
    ):
        super().__init__(data_table, ground_truth, verbose)
        
    def impute_label(self, X, y):
        lp_model = LabelSpreading(gamma=0.25, max_iter=20)
        lp_model.fit(X, y)
        predicted_labels = lp_model.transduction_
        return predicted_labels
        


        