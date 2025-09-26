import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv("eab-data2/eab-dataset2.csv")
    df.info()
    print()
    print(df.head(10))

    columns = ['throughput (MB/s)', 'Latency(ms)']
    thresholds = [15, 150]
    relop = ['<', '>']

    from thr_anom_labeler import ThresholdAnomalyLabeler
    tal = ThresholdAnomalyLabeler(df, 'or', columns, thresholds, relop, ignore_zero_values=True)
    df = tal.save('eab-data2/eab-dataset2-labeled.csv')