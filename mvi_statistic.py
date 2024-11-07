from pandas import DataFrame
import numpy as np
import pandas as pd


def count_missing_values(dataset:DataFrame):
    for column in dataset.columns:
        counter = 0
        for data in dataset[column]:
            if pd.isna(data):
                counter += 1
        print(column, " has missing values ", counter)
        

def data_has_missing_values(dataset:DataFrame):        
    for col in list(dataset.columns):
        if dataset[col].isnull().values.any():
            print(col, ' has empty ',dataset[col].isnull().values.any())
            
def check_treshold(dataset:DataFrame, columns:list, tresholds:list):
    ctmap = dict()
    for i in range(0, len(columns)):
        ctmap[columns[i]] = tresholds[i]
        
    tmap = dict()
    dataset['anomaly'] = ""
    
    for col in columns:
        i = 0
        data = dataset[col]
        tmap[col] = 0
        for cell in data:
            if cell >= ctmap[col]:
                tmap[col] += 1
                
        if tmap[col] > 0:
            dataset['anomaly'][i] = 1
        else:
            dataset['anomaly'][i] = 0
        
    return dataset

def count_anomalies(dataset:DataFrame):
    cnt = 0
    for cell in dataset['anomaly']:
        if cell == 1:
            cnt += 1
    return cnt
        
        
