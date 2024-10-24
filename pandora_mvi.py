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
    iterative (default) - creates a predictive model for each attribute, and uses it to fill the missing values.
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
from sklearn.ensemble import RandomForestRegressor
import sys

def apply_mvi(in_path, out_path, algorithm):
    # Checking if the dataset exists.
    if not os.path.exists(in_path):
        print('Error: Path "' + in_path + '" does not exist.')
        return

    # Set the out path
    if out_path == '':
        out_path = os.path.join(os.getcwd(), ntpath.basename(in_path))

    # Loading dataset.
    dataset = pd.read_csv(in_path)
    newds = dataset.copy()
    newds = newds.drop('timestamp', axis=1)

    # Set the imputer.
    #imputer = IterativeImputer(max_iter=10, random_state=0) if algorithm == 'iterative' else SimpleImputer(missing_values=np.nan, strategy='mean')
    #imputer = IterativeImputer(estimator=BaggingRegressor() ,max_iter=10, random_state=0) if algorithm == 'iterative' else SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, max_depth=10) ,max_iter=10, random_state=0) if algorithm == 'iterative' else SimpleImputer(missing_values=np.nan, strategy='mean')

    # Impute missing values.
    target_name = newds.columns[-1]
    target_data = newds[target_name]
    newds.drop(newds.columns[-1], axis=1, inplace=True)
    newds = pd.DataFrame(imputer.fit_transform(newds), columns=newds.columns)
    newds[target_name] = target_data

    # Save imputed dataset.
    newds.insert(0, 'timestamp', dataset['timestamp'])
    newds.to_csv(out_path, index=False)

if __name__ == '__main__':
    # Parsing arguments.
    parser = argparse.ArgumentParser(description='Missing values imputation (MVI).')
    parser.add_argument('-d', '--dataset', help='Path to the dataset.', required=True)
    parser.add_argument('-o', '--out', default='', help='Path to the output file. If not given, resulting dataset will be saved in the current directory.')
    parser.add_argument('-a', '--algorithm', default='iterative', help='MVI algorithm. Possible values:\nsimple - takes the mean of attribute\'s values, and uses it to fill the missing values,\niterative (default) - creates a predictive model for each attribute, and uses it to fill the missing values.')
    args = parser.parse_args()

    apply_mvi(args.dataset, args.out, args.algorithm)