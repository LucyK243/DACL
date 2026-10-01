"""
PANDORA 

Main entry point for DACL missing label inference

Author
----------
lucy@dmi.uns.ac.rs
"""

import ntpath
import os

from datatable import DataTable
from mli.mli import LabelSpreading_MLI, LabelPropagation_MLI, TabPFN_MLI, ST_NB_MLI

def apply_mli(
    in_path, out_path, target_vars, 
    ground_truth=None, 
    algorithm='tabpfn', 
    columns_to_ignore=None
):
    print("Applying DACL-MLI...")
    print("Imputation algorithm", algorithm)
    
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

    imputer = None
    if algorithm == "tabpfn":
        imputer = TabPFN_MLI(data_table, ground_truth=ground_truth, verbose=False)
    elif algorithm == "ls":
        imputer = LabelSpreading_MLI(data_table, ground_truth=ground_truth, verbose=False)
    elif algorithm == "lp":
        imputer = LabelPropagation_MLI(data_table, ground_truth=ground_truth, verbose=False)
    elif algorithm == "nb":
        imputer = ST_NB_MLI(data_table, ground_truth=ground_truth, verbose=False)
    else:
        raise Exception("Invalid MLI algorithm")

    # impute & save
    imputer.impute()
    imputer.save(out_path)

    print("DACL-MLI finished.")

