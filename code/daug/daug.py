"""
PANDORA 

DACL, data augmentation mode

authors: {svc, lucy}@dmi.uns.ac.rs
"""

from datatable import DataTable
import numpy as np
import torch
import pandas as pd

from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, KMeansSMOTE


from daug_impl import DACL_TabPFGen, DACL_SMOTE, DACL_ADASYN, DACL_SVMSMOTE, DACL_KMeansSMOTE


# from cl_eval import evaluate_ds


if __name__ == '__main__':
    ds = DataTable('data/eab/eab-dataset2-labeled.csv', target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id', 'throughput (MB/s)', 'Latency(ms)'], verbose=False, sep=',')
    
    # ds = DataTable('data/eab_d1_daug.csv', target_vars=['ANOMALY'], columns_to_ignore=['bag_file', 'timestamp', 'time[s]', 'throughput_DL', 'throughput_UL', 'delay_mean_DL', 'delay_mean_UL'], verbose=False, sep=',')
    
    tabgen = DACL_TabPFGen(dataset=ds)
    # print(tabgen.get_aug_data())
    tabgen.evaluate()
    tabgen.export('data/eab/eab2-aug-test-tabpfgen.csv')
    
    
    #smote = DACL_SMOTE(ds)
    #smote.evaluate()#
    #smote.export('data/eab/eab2-aug-test-smote.csv')
    
    
    
    
    # X = ds.data_matrix
    # print("X shape", X.shape, type(X))
    # y = ds.target_vars_data
    # print("Y shape", y.shape, type(y))

    # y_name = ds.target_vars
    # print(y_name)

    # #print("BASELINE-CLASSIFIERS")
    # #evaluate_ds(X, y, y_name)

    # #sm = SMOTE(random_state=42)
    # #X_aug, y_aug = sm.fit_resample(X, y)
    # #print("SMOTE")
    # #evaluate_ds(X_aug, y_aug, y_name)
    
    # #adasyn = ADASYN(random_state=42)
    # #X_aug, y_aug = adasyn.fit_resample(X, y)
    # #print("ADASYN")
    # #evaluate_ds(X_aug, y_aug, y_name)

    # #svmsm = SVMSMOTE(random_state=42)
    # #X_aug, y_aug = svmsm.fit_resample(X, y)
    # #print("SVM SMOTE")
    # #evaluate_ds(X_aug, y_aug, y_name)

    # #kmeanssm = KMeansSMOTE(random_state=42)
    # #X_aug, y_aug = kmeanssm.fit_resample(X, y)
    # #print("KMEANS SMOTE")
    # #evaluate_ds(X_aug, y_aug, y_name)
    
    # print("TABPFGEN")
    # # 
    # #os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET']='1'
    # generator = TabPFGen(n_sgld_steps=500, device=torch.device("cpu"))
    # #y_tabpf = np.array(y.to_dict())
    # #print(y_tabpf.keys())
    # #print(type(y_tabpf))
    # #print(y_tabpf)
    
    # y = y[y_name[0]].to_numpy()
    # y = np.where(y == 'no', 0, 1)
    # #print(y)
    
    # X_synth, y_synth, X_combined, y_combined = generator.balance_dataset(X, y)
    # print("TABPFGEN generation finished")
    # #print(type(X_combined), X_combined.shape)
    # #print(type(y_combined), y_combined.shape)
    
    # print("DAG-CLASSIFIERS-TABPFGEN")
    # y_df = df = pd.DataFrame(y_combined, columns=y_name) 
    # evaluate_ds(X_combined, y_df, y_name)
    