
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

def discover_mvi_rules(dt, data_matrix, column_names, dimX, dimY):
    fwmv = dt.features_with_missing_values
    pattern_dict = dict()
    target_dict = dict()
    event_dict = dict()
    target_dict_present = dict()

    pairs_dict = dict()
    for f2 in range(1, dimY):
        for f1 in range(f2):
            f1_name = column_names[f1]
            f2_name = column_names[f2]
            if f1_name in fwmv and f2_name in fwmv:
                k = f2_name + "#" + f1_name
                pairs_dict[k] = [0, 0, 0, 0]
    
    mv_data_points = 0
    target_field = "Anomaly_Class"
    target_missing, num_events = 0, 0
    
    for i in range(dimX):
        missing_features = []        
        for j in range(dimY):
            cell_type = type(data_matrix[i][j]).__name__
            cell_value = data_matrix[i][j]
            if is_missing(cell_type, cell_value):
                missing_features.append(column_names[j])

        missing_target = target_field in missing_features
        if not missing_target:
            for mf in missing_features:
                if mf not in target_dict_present:
                    target_dict_present[mf] = 1
                else:
                    target_dict_present[mf] += 1

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
            else:
                num_events += 1
                event_class = data_matrix[i][-1]
                if event_class in event_dict:
                    event_dict[event_class] += 1
                else:
                    event_dict[event_class] = 1

        for f2 in range(1, dimY):
            for f1 in range(f2):
                f1_name = column_names[f1]
                f2_name = column_names[f2]
                if f1_name in fwmv and f2_name in fwmv:
                    k = f2_name + "#" + f1_name

                    f1_missing = f1_name in missing_features
                    f2_missing = f2_name in missing_features

                    if not f1_missing and not f2_missing:
                        pairs_dict[k][0] += 1
                    elif f1_missing and f2_missing:
                        pairs_dict[k][1] += 1
                    elif not f1_missing and f2_missing:
                        pairs_dict[k][2] += 1
                    else:
                        pairs_dict[k][3] += 1         
                
    print("Data points with missing values", mv_data_points)
    print("Data points with ", target_field, "missing", target_missing)
    print()
    print("Events info")
    print("#events", num_events)
    for event in event_dict:
        print(event_dict[event], " -- ", event)
    
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

    print()
    print("Missing features when target present ")
    sorted_dict = dict(sorted(target_dict_present.items(), key=lambda item: item[1], reverse=True))
    for item in sorted_dict:
        freq = sorted_dict[item]
        print(freq, item)

    print()
    print("Always present features when target is present")
    for c in column_names:
        if not c in sorted_dict:
            print(c)

    print()
    print("Pairs dictionary")
    print("[present-present, missing-missing, present-missing, missing-present]")
    for p in pairs_dict:
        print(p, pairs_dict[p])
    
        

if __name__ == '__main__':
    print("Framatome dataset inspection")
    
    #dt = DataTable('data/framatome/FRA_Dataset_2000.csv', verbose=True, sep=',', encoding='UTF-16')
    dt = DataTable('/home/svc/dev/FRA_Data/FRA_Dataset.csv', sep=";", verbose='True', encoding='UTF-16')
    data_matrix = dt.data_matrix
    column_names = dt.column_names
    dimX, dimY = data_matrix.shape
    
    discover_mvi_rules(dt, data_matrix, column_names, dimX, dimY)
