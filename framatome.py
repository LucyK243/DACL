
from datatable import DataTable
import numpy as np

def is_missing(cell_type, cell_value):
    missing_value = False
    
    if cell_type == "str":
        if cell_value == "":
            missing_value = True
    elif cell_type == "float" or cell_type == "int":
        if np.isnan(cell_value):
            missing_value = True
    
    return missing_value

def discover_mvi_rules(data_matrix, column_names, dimX, dimY):
    pattern_dict = dict()
    target_dict = dict()
    mv_data_points = 0
    
    target_field = "Anomaly_Class"
    target_missing = 0
    
    for i in range(dimX):
        missing_features = []        
        for j in range(dimY):
            cell_type = type(data_matrix[i][j]).__name__
            cell_value = data_matrix[i][j]
            if is_missing(cell_type, cell_value):
                missing_features.append(column_names[j])

        if len(missing_features) > 0:
            mv_data_points += 1
            pattern = "--".join(missing_features)
            if not pattern in pattern_dict:
                pattern_dict[pattern] = 1
            else:
                pattern_dict[pattern] += 1
                
            if target_field in missing_features:
                target_missing += 1
                for mf in missing_features:
                    if mf != target_field:
                        if mf not in target_dict:
                            target_dict[mf] = 1
                        else:
                            target_dict[mf] += 1
                
    print("Data points with missing values", mv_data_points)
    print("Data points with ", target_field, "missing", target_missing)
    print()
    
    sorted_dict = dict(sorted(pattern_dict.items(), key=lambda item: item[1], reverse=True))
    for item in sorted_dict:
        freq = sorted_dict[item]
        print(freq, item)
        print()
        
    print()
    print("TARGET VAR RULES")    
        
    sorted_dict = dict(sorted(target_dict.items(), key=lambda item: item[1], reverse=True))
    for item in sorted_dict:
        freq = sorted_dict[item]
        coverage = round(freq / target_missing, 4)
        print(freq, item, "coverage = ", coverage)
        

if __name__ == '__main__':
    print("Framatome dataset inspection")
    
    dt = DataTable('data/framatome/FRA_Dataset_2000.csv', verbose=True, sep=',', encoding='UTF-16')
    data_matrix = dt.data_matrix
    column_names = dt.column_names
    dimX, dimY = data_matrix.shape
    
    discover_mvi_rules(data_matrix, column_names, dimX, dimY)
