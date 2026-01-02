from datatable import DataTable
from mli import LabelSpreading_MLI, LabelPropagation_MLI, TabPFN_MLI, ST_NB_MLI, get_MLI_eval_desc

if __name__ == '__main__':
    print("LOADING GROUND TRUTH")
    ground_truth = DataTable('data/eab/eab-dataset2-labeled.csv', target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'], verbose=False, sep=',')
    # print(ground_truth.missing_labels)
    
    print("\n")
    print("LOADING INPUT FILE")
    data_table = DataTable('data/eab/eab2_10.csv', target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'], verbose=False)
    ls = LabelSpreading_MLI(data_table, ground_truth=ground_truth, verbose=False)
    ls.impute()
    ls.save('data/eab/eab2_10_imp_LS.csv')
    
    data_table = DataTable('data/eab/eab2_10.csv', target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'], verbose=False)
    lp = LabelPropagation_MLI(data_table, ground_truth=ground_truth, verbose=False)
    lp.impute()
    lp.save('data/eab/eab2_10_imp_LP.csv')

    data_table = DataTable('data/eab/eab2_10.csv', target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'], verbose=False)
    stnb = ST_NB_MLI(data_table, ground_truth=ground_truth, verbose=False)
    stnb.impute()
    stnb.save('data/eab/eab2_10_imp_NB.csv')

    data_table = DataTable('data/eab/eab2_10.csv', target_vars=['ANOMALY'], columns_to_ignore=['header.stamp.sec', 'header.stamp.nanosec', 'header.frame_id', 'child_frame_id'], verbose=False)
    tabpfn = TabPFN_MLI(data_table, ground_truth=ground_truth, verbose=False)
    tabpfn.impute()
    tabpfn.save('data/eab/eab2_10_imp_TABPFN.csv')

    print(get_MLI_eval_desc())
    print(ls.get_evaluation().get_info())
    print(lp.get_evaluation().get_info())
    print(stnb.get_evaluation().get_info())
    print(tabpfn.get_evaluation().get_info())
    
