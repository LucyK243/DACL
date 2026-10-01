"""
PANDORA 

Main entry point for DACL missing value inference

Author
----------
lucy@dmi.uns.ac.rs
"""

import ntpath
import os
import pandas as pd

from mvi.imputer import SimpleImputerDACL, IterativeImputerDACL, KNNImputerDACL, HyperimputeDACL
from sklearn.ensemble import BaggingRegressor

from datatable import DataTable

from mvi.mvinn import NNConf, TLConf, MVINN, default_NN_arch

def load_data_pandas(in_path):
    return pd.read_csv(in_path)

def load_data_dacl(in_path, columns_to_ignore):
    if columns_to_ignore == None:
        columns_to_ignore = []
    dt = DataTable(in_path, columns_to_ignore=columns_to_ignore)
    num_features = dt.data_matrix.shape[1]
    input_features = num_features - 1
    return dt, input_features
                   
                   
def apply_mvi(in_path, out_path, algorithm='simple', columns_to_ignore=None):
    print("Applying DACL-MVI...")
    print("Imputation algorithm", algorithm)
    # Checking if the dataset exists.
    if not os.path.exists(in_path):
        print('Error: Path "' + in_path + '" does not exist.')
        return

    # Set the out path
    if out_path == '':
        out_path = os.path.join(os.getcwd(), ntpath.basename(in_path))
    
    # Imputing missing data
    imputer = None
    if algorithm == 'iterative':
        dataset = load_data_pandas(in_path)
        imputer = IterativeImputerDACL(
            dataset=dataset, 
            estimator=BaggingRegressor(), 
            columns_to_ignore=columns_to_ignore
        )
    elif algorithm == 'knn':
        dataset = load_data_pandas(in_path)
        imputer = KNNImputerDACL(
            dataset=dataset, 
            n=5, 
            columns_to_ignore=columns_to_ignore
        )
    elif algorithm == 'hyperimputer':
        dataset = load_data_pandas(in_path)
        imputer = HyperimputeDACL(
            dataset=dataset, 
            imputer='mean', 
            columns_to_ignore=columns_to_ignore
        )
    elif algorithm == 'simple':
        dataset = load_data_pandas(in_path)
        imputer = SimpleImputerDACL(
            dataset=dataset, 
            columns_to_ignore=columns_to_ignore
        )
    elif algorithm == 'nntl':
        dt, nf = load_data_dacl(in_path, columns_to_ignore=columns_to_ignore)
        nn_conf = NNConf(nf, 
            default_NN_arch(nf), 
            batch_norm=False, batch_size=256, dropout=0.1
        )
        tl_conf = TLConf(2, freeze=False)
            
        imputer = MVINN(dt, nn_conf, tl_conf=tl_conf)
    else:
        raise Exception("Invalid MVI algorithm")

    newds = imputer.impute()
    
    # Saving new data to file
    if algorithm == 'nntl':
        imputer.save(out_path)
    else:
        newds.to_csv(out_path, index=False)

    print("DACL-MVI finished.")