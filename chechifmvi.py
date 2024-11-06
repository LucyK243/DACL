import pandas as pd

dataset = pd.read_csv('eab_sample_mvi_simple.csv')

for col in list(dataset.columns):
    if dataset[col].isnull().values.any():
        print(col, ' has empty ',dataset[col].isnull().values.any())