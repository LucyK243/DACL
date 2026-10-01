"""
PANDORA 

Main DACL entry point

Author
----------
lucy@dmi.uns.ac.rs
"""

import warnings

# Suppress only UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

from exec_mvi import apply_mvi
from exec_mli import apply_mli
from exec_daug import apply_daug

from datatable import DataTable

from mvi.regeval import evaluateForAllFeatures
from mvi.odeval import outlier_report

from mli.mli import MLIPrimaryEval
from cl_eval import evaluate_dataset

import argparse

def info(dataset):
    dt = DataTable(dataset)
    dim = dt.data_matrix.shape
    print("Num features", dim[1])
    print("Num data points", dim[0])

def info_mvi(dataset):
    dt = DataTable(dataset)
    dt.identify_missing_values(print_report=True)

def info_mli(dataset, target_vars):
    dt = DataTable(dataset, target_vars=target_vars)
    dt.identify_missing_targets(print_report=True)

def mvi_primary_eval(dataset, ground_truth):
    print("Primary evaluation of DACL missing value inference")
    dt_imputed = DataTable(dataset)
    dt_gt = DataTable(ground_truth)
    dt_gt.diff(dt_imputed)

def mvi_app_eval(dataset):
    print("Applicative evaluation of DACL missing value inference")
    dt = DataTable(dataset)
    evaluateForAllFeatures(dt)

def outlier_info(dataset):
    outlier_report(dataset)

def mli_primary_eval(dataset, ground_truth, target_vars):
   print("Primary evaluation of DACL missing label inference")
   evaluator = MLIPrimaryEval(dataset, ground_truth, target_vars)
   evaluator.eval()

def mli_app_eval(dataset, target_vars):
    print("Applicative evaluation of DACL missing label inference")
    dt = DataTable(dataset, target_vars=target_vars)
    evaluate_dataset(dt)
    
def daug_eval(dataset, aug_dataset, target_vars):
    print("Applicative evaluation of DACL data augmentation")
    print("Model evaluation on the original dataset")
    dt = DataTable(dataset, target_vars=target_vars)
    evaluate_dataset(dt)

    print()
    print("Model evaluation on the augmented dataset")
    dt = DataTable(aug_dataset, target_vars=target_vars)
    evaluate_dataset(dt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='DACL', description='DACL tool for MVI, MLI and DAUG')

    parser.add_argument('--info', '--info', action='store_true', help='Info about dataset')
    parser.add_argument('--mvi_info', '--mvi_info', action='store_true', help='Info about missing values in dataset')
    parser.add_argument('--mli_info', '--mli_info', action='store_true', help='Info about missing labels in dataset')    
    parser.add_argument('--outlier_info', '--outlier_info', action='store_true', help='Info about outliers in dataset')
    parser.add_argument('--input', '--input', help='Input dataset')
    parser.add_argument('--output', '--output', help='Output dataset')
    parser.add_argument('--gt', '--gt', help='Ground truth dataset')
    parser.add_argument('--algorithm', '--algorithm', help='MVI/MLI/DAUG algorithm that will be applied')
    parser.add_argument('--target_var', '--target_var', help='Target variable')
    parser.add_argument('--ignore_cols', '--ignore_cols', help='Columns that will be ignored when applying DACL given as a string separated by comma')

    parser.add_argument('--apply_mvi', '--apply_mvi', action='store_true', help="Execute DACL missing value inference")
    parser.add_argument('--apply_mli', '--apply_mli', action='store_true', help="Execute DACL missing label inference")
    parser.add_argument('--apply_daug', '--apply_daug', action='store_true', help="Execute DACL data augmentation")

    parser.add_argument('--mvi_eval', '--mvi_eval', action='store_true', help='MVI primary evaluation')
    parser.add_argument('--mvi_app_eval', '--mvi_app_eval', action='store_true', help='MVI applicative evaluation')
    parser.add_argument('--mli_eval', '--mli_eval', action='store_true', help='MLI primary evaluation')
    parser.add_argument('--mli_app_eval', '--mli_app_eval', action='store_true', help='MLI applicative evaluation')
    parser.add_argument('--daug_eval', '--daug_eval', action='store_true', help="DAUG evaluation")

    args = parser.parse_args()

    columns_to_ignore = None
    if args.ignore_cols != None:
        tmp = args.ignore_cols.split(",")
        columns_to_ignore = [t.strip() for t in tmp]
        
    target_vars = None
    if args.target_var != None:
        target_vars = [args.target_var]

    algorithm = args.algorithm
    
    if args.info:
        info(args.input)
    elif args.mvi_info:
        info_mvi(args.input)
    elif args.mli_info:
        info_mli(args.input, [args.target_var])
    elif args.outlier_info:
        outlier_info(args.input)
    elif args.apply_mvi:
        if algorithm != None:
            apply_mvi(args.input, args.output, algorithm=algorithm, columns_to_ignore=columns_to_ignore)
        else:
            apply_mvi(args.input, args.output, columns_to_ignore=columns_to_ignore)
    elif args.mvi_eval:
        mvi_primary_eval(args.input, args.gt) 
    elif args.mvi_app_eval:
        mvi_app_eval(args.input)   
    elif args.apply_mli:
        if algorithm != None:
            apply_mli(args.input, args.output, target_vars, algorithm=algorithm, columns_to_ignore=columns_to_ignore)
        else:
            apply_mli(args.input, args.output, target_vars, columns_to_ignore=columns_to_ignore)   
    elif args.mli_eval:
        mli_primary_eval(args.input, args.gt, target_vars) 
    elif args.mli_app_eval:
        mli_app_eval(args.input, target_vars)
    elif args.apply_daug:
        if algorithm != None:
            apply_daug(args.input, args.output, target_vars, algorithm=algorithm, columns_to_ignore=columns_to_ignore)
        else:
            apply_daug(args.input, args.output, target_vars, columns_to_ignore=columns_to_ignore)
    elif args.daug_eval:
        daug_eval(args.gt, args.input, target_vars)


