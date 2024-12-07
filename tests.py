import pandas as pd
from mvi_statistic import data_has_missing_values
from mvi_statistic import check_treshold
from mvi_statistic import count_anomalies

"""
dataset = pd.read_csv('data/eab_sample.csv')
from mvi_statistic import MVIStats
mvis = MVIStats(dataset)
mvis.info()
"""

dataset = pd.read_csv('data/eab_sample_knn_n5.csv')
columns = ['throughput_DL', 'throughput_UL', 'delay_mean_DL', 'delay_mean_UL']
thresholds = [7.2, 4.9, 1.1, 0.3]
relop = ['<', '<', '>', '>']
from thr_anom_labeler import ThresholdAnomalyLabeler
tal = ThresholdAnomalyLabeler(dataset, 'and', columns, thresholds, relop)
tal.save('data/anomaly_labels.csv')

#tal = ThresholdAnomalyLabeler(dataset, 'or', columns, thresholds, relop)


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

