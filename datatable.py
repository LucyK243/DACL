"""
PANDORA 

DACL, data table module

authors: {svc, lucy}@dmi.uns.ac.rs
"""

import os
import numpy as np
import pandas as pd
import random as rnd

from sklearn.metrics import r2_score, mean_absolute_error

class DataTable:
    def __init__(self, csv_file, target_vars=[], columns_to_ignore=[], verbose=False):
        self.target_vars = target_vars
        self.columns_to_ignore = columns_to_ignore
        self.verbose = verbose
        
        self.current_dir = os.getcwd()
        csv_file_path = os.path.join(self.current_dir, csv_file)
        
        if verbose:
            print("Input file", csv_file_path)
        
        if not os.path.exists(csv_file_path):
            raise Exception("Unable to find", csv_file_path)
        
        self.dataframe = pd.read_csv(csv_file_path)
        
        if verbose:
            print("Data table loaded, dimension", self.dataframe.shape)
        
        if len(target_vars) > 0:
            self.target_vars_data = self.dataframe[self.target_vars]
        
        if len(columns_to_ignore) > 0:
            self.columns_to_ignore_data = self.dataframe[self.columns_to_ignore]
        
        if len(target_vars) + len(columns_to_ignore) > 0:
            self.dataframe.drop(axis=1, inplace=True, columns=self.target_vars + self.columns_to_ignore)
        
        if verbose:
            print("Data table after drop operations", self.dataframe.shape)
        
        self.column_names = list(self.dataframe.columns)
        self.data_matrix = self.dataframe.to_numpy()
        
        self.identify_missing_values()
        
    
    def identify_missing_values(self):
        dimX, dimY = self.data_matrix.shape
        total_nans = 0
        self.missing_values_dict = dict()
        
        for i in range(dimX):
            for j in range(dimY):
                # print(self.data_matrix[i][j], type(self.data_matrix[i][j]))
                if np.isnan(self.data_matrix[i][j]):
                    if j not in self.missing_values_dict:
                        self.missing_values_dict[j] = (self.column_names[j], [i])
                    else:
                        self.missing_values_dict[j][1].append(i)
                    
                    total_nans += 1
                    
        if self.verbose:
            total_vals = dimX * dimY
            print("Missing values identified")
            print("Missing values", total_nans, "total values", total_vals)
            
            sorted_dict = dict(sorted(self.missing_values_dict.items()))
            
            for j in sorted_dict:
                print("Column", j, "name", self.missing_values_dict[j][0])
                print("Missing values present for", self.missing_values_dict[j][1])
                print()
    
            
    def remove_rnd_values(self, p):
        dimX, dimY = self.data_matrix.shape
        total_nans = 0
        for i in range(dimX):
            for j in range(dimY):
                if rnd.random() <= p:
                    self.data_matrix[i][j] = np.nan
                    total_nans += 1
                 
        if self.verbose:   
            total_vals = dimX * dimY
            print("Total nans", total_nans, "out of", total_vals, "perc", total_nans / total_vals)
            
        self.dataframe = pd.DataFrame(self.data_matrix, columns=self.column_names)
            
    def save(self, out_file):
        if len(self.columns_to_ignore) > 0:
            self.dataframe = pd.concat([self.columns_to_ignore_data, self.dataframe], axis=1)
            
        if len(self.target_vars) > 0:
            self.dataframe = pd.concat([self.dataframe, self.target_vars_data], axis=1)
            
        self.dataframe.to_csv(out_file, index=False)
        
    def diff(self, other_dt):
        self.r2, self.mae = dict(), dict()
        
        for cn in self.column_names:
            c1 = self.dataframe[cn].to_numpy()
            c2 = other_dt.dataframe[cn].to_numpy()
            if len(c1) != len(c2):
                raise Exception("diff", len(c1), len(c2))
            
            self.r2[cn] = r2_score(c1, c2)
            self.mae[cn] = mean_absolute_error(c1, c2)

        print('column, R2, MAE')

        for cn in self.column_names:
            print(cn, self.r2[cn], self.mae[cn])
            
            
    def diff_data_matrix(self, other_dm):
        self.r2, self.mae = dict(), dict()
        
        if self.data_matrix.shape != other_dm.shape:
            raise Exception("diff data matrix, different shapes")
        
        for j in range(self.data_matrix.shape[1]):
            c1 = self.data_matrix[:, j]
            c2 = other_dm[:, j]
            column_name = self.column_names[j]
            self.r2[column_name] = r2_score(c1, c2)
            self.mae[column_name] = mean_absolute_error(c1, c2)
            
        return (self.mae, self.r2)
        
            
if __name__ == '__main__':
    
    
    # Step 1 -- removing some values from a complete dataset
    dt = DataTable('data/eab_common_data.csv', 
        columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'],
        target_vars=['throughput (MB/s)', 'Latency(ms)'],
        # target_vars = ['negative_image_counter_send_total - devx/trex-client', 'packet_rate_packets_per_second - devx/trex-client'], 
        verbose=True)
    
    dt.remove_rnd_values(0.25)
    dt.save('data/eab_common_data_mv25.csv')
    
    
    
    # Step 2 -- data imputation
    from imputers.Imputer import SimpleImputerDACL
    from imputers.Imputer import HyperimputeDACL
    from imputers.Imputer import KNNImputerDACL
    df = pd.read_csv('data/eab_common_data_mv25.csv')
    simpleImputer = SimpleImputerDACL(dataset=df, columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'])
    #HyperimputeDACL(dataset=df, imputer='sinkhorn', columns_to_ignore=['Time']) 
    #KNNImputerDACL(dataset=df, n=3, columns_to_ignore=['Time'])
    #SimpleImputerDACL(dataset=df, columns_to_ignore=['Time'])
    sidf = simpleImputer.impute()
    sidf.to_csv('data/eab_common_data-25-simple.csv', index=False)
    
    
    
    
    # Step 3 -- comparison
    dt_orig = DataTable('data/eab_common_data.csv', columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'],  target_vars=['throughput (MB/s)', 'Latency(ms)'])
    dt_infr = DataTable('data/eab_common_data-25-simple.csv', columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'],  target_vars=['throughput (MB/s)', 'Latency(ms)'])
    dt_orig.diff(dt_infr)
    
    
    #dt = DataTable('data/sie-dataset1-mv.csv', columns_to_ignore=['Time'], verbose=True)