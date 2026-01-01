from datatable import DataTable
from mli import Dummy_MLI

if __name__ == '__main__':
    print("LOADING GROUND TRUTH")
    ground_truth = DataTable('data/eab/eab-dataset2-labeled.csv', target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'], verbose=True, sep=',')
    # print(ground_truth.missing_labels)
    
    print("\n")
    print("LOADING INPUT FILE")
    data_table = DataTable('data/eab/eab2_10.csv', target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'], verbose=True)
    # print(input_file.missing_labels)
    
    dummy = Dummy_MLI(data_table, ground_truth=ground_truth, verbose=True)
    dummy.impute()
    dummy.save('data/eab/eab2_10_imputed.csv')
    
    
# header.stamp.sec,header.stamp.nanosec,header.frame_id,child_frame_id,
    
