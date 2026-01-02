"""
PANDORA 

DACL, missing label inference

authors: {svc, lucy}@dmi.uns.ac.rs
"""

from abc import ABC, abstractmethod
from datatable import DataTable
import numpy as np
import pandas as pd

from sklearn.semi_supervised import LabelSpreading, LabelPropagation, SelfTrainingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from tabpfn import TabPFNClassifier

def get_MLI_eval_desc():
    return "NAME,ACC,BACC,PREC,REC,F1"

class MLI_Eval:
    def __init__(self, name, acc, bacc, prec, rec):
        self.name = name  # method identifier
        self.acc  = acc   # accuracy
        self.bacc = bacc  # balanced accuracy
        self.prec = prec  # macro-averaged precision
        self.rec  = rec   # macro-averaged recall
        self.f1   = 2 * prec * rec / (prec + rec)   # macro-averaged f1

        s = name + ","
        s += str(round(self.acc, 4)) + ","
        s += str(round(self.bacc, 4)) + ","
        s += str(round(self.prec, 4)) + ","
        s += str(round(self.rec, 4)) + ","
        s += str(round(self.f1, 4))

        self.info = s

    def get_info(self):
        return self.info



class MLI(ABC):
    def __init__(self,
        name: str,
        data_table:DataTable,
        ground_truth:DataTable=None,
        verbose=False
    ):
        self.name = name
        self.data_table = data_table
        self.ground_truth = ground_truth
        self.verbose = verbose
        
        
    @abstractmethod
    def impute_label(self, X, y):
        pass
    
    def save(self, filename):
        #self.data_table.save_with_ignored_data(filename)
        self.data_table.save_with_new_labels(filename, self.final_labels_df)

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
        self.final_labels_df = pd.DataFrame()
                
        X = self.data_table.data_matrix
        targets = self.data_table.target_vars_data
        
        for target in target_vars:
            if self.verbose:
                print("IMPUTING LABELS for", target)
            
            if target not in missing_labels:
                continue
            
            y = targets[target].to_numpy()
            ml_indices = missing_labels[target]
            
            final_labels = self.impute_target(X, y, ml_indices)
            self.final_labels_df[target] = final_labels
            
            # evaluation
            if self.ground_truth != None:        
                gt = self.ground_truth.target_vars_data[target].to_list()
                predicted_labels = [final_labels[i] for i in ml_indices]
                true_labels = [gt[i] for i in ml_indices]
                
                acc = accuracy_score(true_labels, predicted_labels)
                bacc = balanced_accuracy_score(true_labels, predicted_labels)
                prec = precision_score(true_labels, predicted_labels, average='macro')
                rec = recall_score(true_labels, predicted_labels, average='macro')
                
                self.evalres = MLI_Eval(self.name, acc, bacc, prec, rec)
                
        for label in target_vars:
            if label not in missing_labels:
                self.final_labels_df[label] = self.data_table.dataframe[label]

        #self.data_table.dataframe = self.final_dataframe
        #self.data_table.save_with_ignored_data(fname)

    def get_evaluation(self):
        return self.evalres         


           
        
class LabelSpreading_MLI(MLI):
    def __init__(self,
        data_table:DataTable,
        ground_truth:DataTable = None,
        verbose=False,
        gamma=0.25,
        max_iter=20
    ):
        super().__init__("LS", data_table, ground_truth=ground_truth, verbose=verbose)
        self.gamma = gamma
        self.max_iter = max_iter
        
    def impute_label(self, X, y):
        lp_model = LabelSpreading(gamma=self.gamma, max_iter=self.max_iter)
        lp_model.fit(X, y)
        predicted_labels = lp_model.transduction_
        return predicted_labels
    

class LabelPropagation_MLI(MLI):
    def __init__(self,
        data_table:DataTable,
        ground_truth:DataTable = None,
        verbose=False,
        gamma=0.25,
        max_iter=20
    ):
        super().__init__("LP", data_table, ground_truth=ground_truth, verbose=verbose)
        self.gamma = gamma
        self.max_iter = max_iter
        
    def impute_label(self, X, y):
        lp_model = LabelPropagation(gamma=self.gamma, max_iter=self.max_iter)
        lp_model.fit(X, y)
        predicted_labels = lp_model.transduction_
        return predicted_labels
    

class ST_NB_MLI(MLI):
    def __init__(self,
        data_table:DataTable,
        ground_truth:DataTable = None,
        verbose=False
    ):
        super().__init__("ST-NB", data_table, ground_truth=ground_truth, verbose=verbose)
        
    def impute_label(self, X, y):
        model = SelfTrainingClassifier(GaussianNB())
        model.fit(X, y)
        predicted_labels = model.transduction_
        return predicted_labels        


class TabPFN_MLI(MLI):
    def __init__(self,
        data_table:DataTable,
        ground_truth:DataTable = None,
        verbose=False
    ):
        super().__init__("TabPFN", data_table, ground_truth=ground_truth, verbose=verbose)
        
    def impute_label(self, X, y):
        model = SelfTrainingClassifier(TabPFNClassifier(random_state=42, ignore_pretraining_limits=True))
        model.fit(X, y)
        predicted_labels = model.transduction_
        return predicted_labels 