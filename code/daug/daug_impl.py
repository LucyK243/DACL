"""
PANDORA 

Data augmentation / data balancing -- SOTA methods and TabPFGen

Author
----------
lucy@dmi.uns.ac.rs
"""

from abc import ABC, abstractmethod

from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, KMeansSMOTE
from daug.tabpfgen_local import TabPFGen

import torch
import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from datatable import DataTable
from cl_eval import evaluate_ds




class Augmenter(ABC):
    
    def __init_model(self):
        '''
        Init augmentation method
        '''
        pass
    
    def __resample(self):
        '''
        Resampling mechanism
        '''
        pass
        
    
    def get_aug_data(self):
        '''
        Returns augmented data with column name as a touple (X_augmented, y_augmented, column_name)
        '''
        
        return self.X_aug, self.y_aug, self.y_name
    
    def evaluate(self):
        '''
        10-fold coress validation from SVM, Naive Bayes, KNN,
        decision tree and random forest classifiers
        '''
        for i in range(len(self.y_name)):
            name = self.y_name[i]
            if isinstance(self.y_aug[i], pd.DataFrame):
                y_aug = self.y_aug[i]
            else:
                y_aug = self.y_aug[i].to_frame(name=name)
            X_aug = self.X_aug[i]
            evaluate_ds(X_aug, y_aug, [name])
        
        
    def export(self, path:str):
        '''
        Exporting data to path as csv
        '''
        
        columns=list(self.dataset.column_names)
        for i in range(len(self.y_name)):
            #name = self.y_name[i]
            y_aug = self.y_aug[i]
            X_aug = self.X_aug[i]
            df_x = pd.DataFrame(X_aug, columns=columns)
            df_res = pd.concat([df_x, y_aug], axis=1)
            tokens = path.split('.csv')
            new_path = tokens[0] + '.csv'
            df_res.to_csv(new_path, index=False)
        
        
    
class DACL_SMOTE(Augmenter):
    '''
    DACL wrapper for SMOTE method from imbalanced-learn library (https://imbalanced-learn.org)
    '''
    def __init__(self, dataset: DataTable):
        super().__init__()
        self.dataset = dataset
        self.X = dataset.data_matrix
        self.y = dataset.target_vars_data
        self.y_name = dataset.target_vars
        self.__init_model()
        self.__resample()
        
    def __init_model(self):
        self.model = SMOTE(random_state=42)
        
    def __resample(self):
        self.y_aug = []
        self.X_aug = []
        for name in self.y_name:
            y = self.y[name]
            X_res, y_res = self.model.fit_resample(self.X, y)
            self.X_aug.append(X_res)
            self.y_aug.append(y_res)

        
class DACL_ADASYN(Augmenter):
    '''
    DACL wrapper for imbalanced-learn ADASYN method (https://imbalanced-learn.org)
    '''
    def __init__(self, dataset: DataTable):
        super().__init__()
        self.dataset = dataset
        self.X = dataset.data_matrix
        self.y = dataset.target_vars_data
        self.y_name = dataset.target_vars
        self.__init_model()
        self.__resample()
        
    def __init_model(self):
        self.model = ADASYN(random_state=42)
        
    def __resample(self):
        self.y_aug = []
        self.X_aug = []
        for name in self.y_name:
            y = self.y[name]
            X_res, y_res = self.model.fit_resample(self.X, y)
            self.X_aug.append(X_res)
            self.y_aug.append(y_res)

        
        
class DACL_SVMSMOTE(Augmenter):
    '''
    DACL wrapper for SVMSMOTE method from imbalanced-learn (https://imbalanced-learn.org)
    '''
    def __init__(self, dataset: DataTable):
        super().__init__()
        self.dataset = dataset
        self.X = dataset.data_matrix
        self.y = dataset.target_vars_data
        self.y_name = dataset.target_vars
        self.__init_model()
        self.__resample()
        
    def __init_model(self):
        self.model = SVMSMOTE(random_state=42)
                
    def __resample(self):
        self.y_aug = []
        self.X_aug = []
        for name in self.y_name:
            y = self.y[name]
            X_res, y_res = self.model.fit_resample(self.X, y)
            self.X_aug.append(X_res)
            self.y_aug.append(y_res)
        
        
class DACL_KMeansSMOTE(Augmenter):
    '''
    DACL wrapper for KMeansSMOTE method from imbalanced-learn (https://imbalanced-learn.org)
    '''
    def __init__(self, dataset: DataTable):
        super().__init__()
        self.dataset = dataset
        self.X = dataset.data_matrix
        self.y = dataset.target_vars_data
        self.y_name = dataset.target_vars
        self.__init_model()
        self.__resample()
        
    def __init_model(self):
        self.model = KMeansSMOTE(random_state=42)
        
    def __resample(self):
        self.y_aug = []
        self.X_aug = []
        for name in self.y_name:
            y = self.y[name]
            X_res, y_res = self.model.fit_resample(self.X, y)
            self.X_aug.append(X_res)
            self.y_aug.append(y_res)

        
class DACL_TabPFGen(Augmenter):

    '''
    DACL wrapper for TabPFGen method (https://github.com/sebhaan/TabPFGen)
    '''
    def __init__(self, dataset:DataTable, device:str='cpu'):
        super().__init__()
        self.dataset = dataset
        self.device = device
        y = dataset.target_vars_data
        self.y_name = dataset.target_vars # TODO: target(S?)
        y_h = []
        for y_one in y:
            y_h.append(y[self.y_name[0]].to_numpy())
        self.y = [np.where(y == 'no', 0, 1) for y in y_h] # TODO: diff class
        self.X = dataset.data_matrix
        
        self.__init_model()
        self.__resample()
        
    def __init_model(self):
        self.generator = TabPFGen(n_sgld_steps=500, device=torch.device(self.device))
        
    def __resample(self):
        i = 0
        self.X_aug = []
        self.y_aug = []
        for y in self.y:
            X_synth, y_synth, X_combined, y_combined = self.generator.balance_dataset(self.X, y)
            y_df = pd.DataFrame(y_combined, columns=[self.y_name[i]])
            i+=1 
        
            self.X_aug.append(X_combined)
            self.y_aug.append(y_df)
    