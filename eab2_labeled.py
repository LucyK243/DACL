from datatable import DataTable

if __name__ == '__main__':
    print("EAB-2 dataset inspection")
    
    dt = DataTable('data/eab/eab-dataset2-labeled.csv', verbose=True, sep=',')
    data_matrix = dt.data_matrix
    column_names = dt.column_names
    dimX, dimY = data_matrix.shape
    
    npos, nneg = 0, 0
    
    for i in range(dimX):
        target_value = data_matrix[i][-1]
        if target_value == "yes":
            npos += 1
        else:
            nneg += 1
            
    print("#total", dimX, "#positive", npos, "#negative", nneg)
    
    dt = DataTable('data/eab/eab-dataset2-labeled.csv', target_vars=['ANOMALY'], verbose=False, sep=',')
    dt.remove_rnd_target_values(0.1)
    dt.save('data/eab/eab2_10.csv')
    
    dt = DataTable('data/eab/eab2_10.csv', verbose=True, sep=',')