"""
PANDORA 

Main entry point for DACL data augmentation module that creates
balanced from imbalanced datasets

Author
----------
lucy@dmi.uns.ac.rs
"""

import ntpath
import os

from datatable import DataTable

from daug.daug_impl import DACL_SMOTE, DACL_ADASYN, DACL_SVMSMOTE, DACL_KMeansSMOTE, DACL_TabPFGen

def apply_daug(
    in_path, out_path, target_vars, 
    algorithm='smote', 
    columns_to_ignore=None
):
    print("Applying DACL-DAUG...")
    print("Data augmentation algorithm", algorithm)

    # Checking if the dataset exists.
    if not os.path.exists(in_path):
        print('Error: Path "' + in_path + '" does not exist.')
        return
    
    # Set the out path
    if out_path == '':
        out_path = os.path.join(os.getcwd(), ntpath.basename(in_path))

    if columns_to_ignore == None:
        columns_to_ignore = []

    data_table = DataTable(in_path, columns_to_ignore=columns_to_ignore, target_vars=target_vars, verbose=False)

    daug = None
    if algorithm == "tabpfgen":
        daug = DACL_TabPFGen(data_table)
    elif algorithm == "smote":
        daug = DACL_SMOTE(data_table)
    elif algorithm == "ada-smote":
        daug = DACL_ADASYN(data_table)
    elif algorithm == "kmeans-smote":
        daug = DACL_KMeansSMOTE(data_table)
    elif algorithm == "svm-smote":
        daug = DACL_SVMSMOTE
    else:
        raise Exception("Invalid MLI algorithm")

    # save
    daug.export(out_path)

    print("DACL-DAUG finished.")