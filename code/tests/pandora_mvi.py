"""
PANDORA 

Missing Values Imputation

Arguments
----------
-d, --dataset
    Path to the dataset.
-o, --out (optional)
    Path to the output file. If not given, resulting dataset will be saved in the current directory.
-a, --algorithm (optional)
    MVI algorithm. Possible values:
    simple - takes the mean of attribute's values, and uses it to fill the missing values,
    iterative (default) - creates a predictive model for each attribute, and uses it to fill the missing values,
    knn - imputation of missing data using k-Nearest Neighbours.
-h, --help (optional)
    Displays help.

Author
----------
lucy@dmi.uns.ac.rs
"""

import argparse
import ntpath
import numpy as np
import os
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from imputers.Imputer import SimpleImputerDACL
from imputers.Imputer import IterativeImputerDACL
from imputers.Imputer import KNNImputerDACL
from imputers.Imputer import HyperimputeDACL

from mvi_statistic import count_missing_values

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
import sys

def apply_mvi(in_path, out_path, algorithm, method):
    # Checking if the dataset exists.
    if not os.path.exists(in_path):
        print('Error: Path "' + in_path + '" does not exist.')
        return

    # Set the out path
    if out_path == '':
        out_path = os.path.join(os.getcwd(), ntpath.basename(in_path))

    # Loading dataset.
    dataset = pd.read_csv(in_path)
    
    # Imputing missing data
    imputer = None
    if algorithm == 'iterative':
        imputer = IterativeImputerDACL(dataset=dataset, estimator=BaggingRegressor(), columns_to_ignore=['timestamp', 'bag_file'])
    elif algorithm == 'knn':
        imputer = KNNImputerDACL(dataset=dataset, n=5, columns_to_ignore=['timestamp', 'bag_file'])
    elif algorithm == 'hyperimputer':
        imputer = HyperimputeDACL(dataset=dataset, imputer=method, columns_to_ignore=['timestamp', 'bag_file'])
    else:
        imputer = SimpleImputerDACL(dataset=dataset, columns_to_ignore=['timestamp', 'bag_file']) 
    newds = imputer.impute()
    
    # Saving new data to file
    newds.to_csv(out_path, index=False)

if __name__ == '__main__':
    # Parsing arguments.
    parser = argparse.ArgumentParser(description='Missing values imputation (MVI).')
    parser.add_argument('-d', '--dataset', help='Path to the dataset.', required=True)
    parser.add_argument('-o', '--out', default='', help='Path to the output file. If not given, resulting dataset will be saved in the current directory.')
    parser.add_argument('-a', '--algorithm', default='iterative', help='MVI algorithm. Possible values:\nsimple - takes the mean of attribute\'s values, and uses it to fill the missing values,\niterative (default) - creates a predictive model for each attribute, and uses it to fill the missing values, \nknn - imputation of missing data using k-Nearest Neighbours,\nhyperimputer - imputation of missing data using hyperimputer library.')
    parser.add_argument('-m', '--method', default='mean', help='Method for hyperimputer library; can be: median, ice, most_frequent, gain, miwae, sklearn_missforest, softimpute, mean, EM, hyperimpute, sinkhorn, missforest')
    args = parser.parse_args()

    apply_mvi(args.dataset, args.out, args.algorithm, args.method)