from datatable import DataTable

if __name__ == '__main__':
    print("LOADING GROUND TRUTH")
    ground_truth = DataTable('data/eab/eab-dataset2-labeled.csv', target_vars=['ANOMALY'], verbose=True, sep=',')
    print(ground_truth.missing_labels)
    
    print("\n")
    print("LOADING INPUT FILE")
    input_file = DataTable('data/eab/eab2_10.csv', target_vars=['ANOMALY'], verbose=True)
    print(input_file.missing_labels)
