"""
PANDORA 

DACL, data augmentation mode

authors: {svc, lucy}@dmi.uns.ac.rs
"""

from datatable import DataTable

from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, KMeansSMOTE

#from tabpfgen import TabPFGen

if __name__ == '__main__':
    ds = DataTable('data/eab/eab-dataset2-labeled.csv', target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id', 'throughput (MB/s)', 'Latency(ms)'], verbose=False, sep=',')
    
    from cl_eval import evaluate_ds

    X = ds.data_matrix
    print("X shape", X.shape)
    y = ds.target_vars_data
    print("Y shape", y.shape)

    y_name = ds.target_vars
    print(y_name)

    print("BASELINE-CLASSIFIERS")
    evaluate_ds(X, y, y_name)

    #sm = SMOTE(random_state=42)
    #sm = ADASYN(random_state=42)
    #sm = SVMSMOTE(random_state=42)
    sm = KMeansSMOTE(random_state=42)
    X_aug, y_aug = sm.fit_resample(X, y)
    print("X shape", X_aug.shape)
    print("Y shape", y_aug.shape)
    
    print("DAG-CLASSIFIERS")
    evaluate_ds(X_aug, y_aug, y_name)

    #print("TABPFGEN")
    # 
    # generator = TabPFGen(n_sgld_steps=500)