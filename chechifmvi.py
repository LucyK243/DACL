import pandas as pd
from mvi_statistic import data_has_missing_values
from mvi_statistic import check_treshold
from mvi_statistic import count_anomalies

#dataset = pd.read_csv('data/eab_sample_knn_n5.csv')
#data_has_missing_values(dataset)
#columns = ['throughput_DL', 'throughput_UL', 'delay_mean_DL', 'delay_mean_UL']
#thresholds = [7.2, 4.9, 1.1, 0.3]
#dataset = check_treshold(dataset,columns, thresholds)
#dataset.to_csv('data/eab_sample_knn_n5_tr.csv', index=False)
completeds = pd.read_csv('data/eab_sample_knn_n5_tr.csv')
print(count_anomalies(completeds))


