import pandas as pd
from mvi_statistic import data_has_missing_values
from mvi_statistic import check_treshold
from mvi_statistic import count_anomalies

from imputers.ZeroImputer import ZeroImputer
from imputers.Imputer import KNNImputerDACL

'''
load_data = pd.read_csv('data/eab_sample.csv')
input_cols = ['throughput_DL', 'throughput_UL', 'delay_mean_DL', 'delay_mean_UL']

zeroI = ZeroImputer(load_data, columns=input_cols)
zero_imputed = zeroI.impute()
zero_imputed.to_csv('data/eab_sample_zero.csv', index=False)

to_ignore = input_cols + ['timestamp', 'bag_file']
knnI = KNNImputerDACL(dataset=zero_imputed, n=5, columns_to_ignore=to_ignore)
final_imputed = knnI.impute()
final_imputed.to_csv('data/eab_sample_zero_knn5.csv', index=False)

'''
'''
dataset = pd.read_csv('data/eab_sample.csv')
from mvi_statistic import MVIStats
mvis = MVIStats(dataset)
mvis.info()

'''
dataset = pd.read_csv('data/eab_sample_zero_knn5.csv')
columns = ['throughput_DL', 'throughput_UL', 'delay_mean_DL', 'delay_mean_UL']
thresholds = [7.2, 4.9, 1.1, 0.3]
#thresholds = [9.7, 6.7, 0.8, 0.24]
#thresholds = [11.3, 9.2, 0.52, 0.18]
relop = ['<', '<', '>', '>']
from thr_anom_labeler import ThresholdAnomalyLabeler
tal = ThresholdAnomalyLabeler(dataset, 'and', columns, thresholds, relop, ignore_zero_values=True)
df = tal.save('data/eab_sample_zero_knn5_labeled.csv')
from mvi_statistic import MVIStats
mvis = MVIStats(df)
mvis.info()

#tal = ThresholdAnomalyLabeler(dataset, 'or', columns, thresholds, relop)
'''

"""
print()
dataset = pd.read_csv('data/eab_sample_knn_n5_tr.csv')
mvis = MVIStats(dataset)
mvis.info()


dataset = pd.read_csv('data/eab_sample_knn_n5.csv')
data_has_missing_values(dataset)
columns = ['throughput_DL', 'throughput_UL', 'delay_mean_DL', 'delay_mean_UL']
thresholds = [7.2, 4.9, 1.1, 0.3]
dataset = check_treshold(dataset,columns, thresholds)
dataset.to_csv('data/eab_sample_knn_n5_tr.csv', index=False)
completeds = pd.read_csv('data/eab_sample_knn_n5_tr.csv')
print(count_anomalies(completeds))
"""

'''