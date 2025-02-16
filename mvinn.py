"""
PANDORA 

DACL, missing value inference based on neural networks and transfer learning

authors: {svc, lucy}@dmi.uns.ac.rs
"""

import numpy as np

from datatable import DataTable

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class MVINN:
    def __init__(self, data_table: DataTable, ground_truth: DataTable=None, verbose=False):
        self.verbose = verbose
        self.ground_truth = ground_truth
        self.data_table = data_table
        self.dimX, self.dimY = self.data_table.data_matrix.shape
        
        # do simple imputer
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(self.data_table.data_matrix)
        self.start_matrix = imp_mean.transform(self.data_table.data_matrix)
        
        # get missing value dictionary (what is missing in data table)
        self.mv_dict = self.data_table.missing_values_dict
        
        if ground_truth != None:
            self.start_mae, self.start_r2 = ground_truth.diff_data_matrix(self.start_matrix)
        
        if self.verbose:
            print("Initial simple imputation... DONE")
            if ground_truth != None:
                for cl in self.start_mae:
                    print(cl, self.start_mae[cl], self.start_r2[cl])
    
    
    def mvi_iteration(self):
        # do scaling
        scaler = StandardScaler()
        scaler.fit(self.start_matrix)
        data_matrix = scaler.transform(self.start_matrix)
        # tmp = scaler.inverse_transform(data_matrix)
        
        num_features = data_matrix.shape[1]
        # for target_feature in range(num_features):
        for target_feature in range(1):
            fname, missing_instances = self.mv_dict[target_feature]
            
            # form training data
            training_data = np.delete(data_matrix, missing_instances, axis=0)
            Y = training_data[:, target_feature]
            X = np.delete(training_data, target_feature, axis=1)
            
            # form inference data
            inference_data = np.take(data_matrix, missing_instances, axis=0)
            inference_X = np.delete(inference_data, target_feature, axis=1)
            
            if self.verbose:
                print("Training neural network for feature", target_feature, fname)
                print("Missing instances", len(missing_instances))
                print("Training data shapes", X.shape, Y.shape)
                print("Inference data shape", inference_X.shape) 
            
        
            
        
if __name__ == '__main__':
    dt = DataTable('data/sie-dataset1-mv.csv', columns_to_ignore=['Time'])
    gt = DataTable('data/sie-dataset1.csv', columns_to_ignore=['Time'])
    
    mvinn = MVINN(dt, ground_truth=gt, verbose=True)
    mvinn.mvi_iteration()
    