from pandas import DataFrame
import numpy as np
import pandas as pd


class MVIStats:
    def __init__(self, dataframe: DataFrame):
        self.df = dataframe
        self.has_missing_values = False
        self.mv_dict = {}
        self.count_missing_values()
        
        
    def count_missing_values(self):
        for column in self.df.columns:
            counter = 0
            for data in self.df[column]:
                if pd.isna(data):
                    counter += 1
                 
            if counter > 0:
                self.has_missing_values = True
                self.mv_dict[column] = counter
                
        self.mv_dict = {k: v for k, v in sorted(self.mv_dict.items(), reverse=True, key=lambda item: item[1])}
        
                
    def info(self):
        num_rows = self.df.shape[0]
        num_columns = self.df.shape[1]
        print("#rows", num_rows)
        print("#columns", num_columns)
        print("Has missing values", self.has_missing_values)
        print("Columns with missing values -- the number of instances")
        for column in self.mv_dict:
            mvinst = self.mv_dict[column]
            perc = round(100.0 * mvinst / num_rows, 2)
            print(column, self.mv_dict[column], perc, "%")
            
            
    def has_missing_values(self):
        return self.has_missing_values
    
    
    def get_missing_features(self):
        """
        Returns dictionary mapping features to the number of instances with missing values
        """
        return self.mv_dict
            
                    
        


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
    # maping columns to tresholds
    ctmap = dict()
    for i in range(0, len(columns)):
        ctmap[columns[i]] = tresholds[i]
        
    tmap = dict()
    anomalies = dict()
    
    i = 0
    for col in columns:
        j = 0
        data = dataset[col]
        tmap[col] = 0
        anomalies[i] = list()
        for cell in data:
            if cell >= ctmap[col]:
                tmap[col] += 1
            if tmap[col] > 0:
                anomalies[i].append(1)
            else:
                anomalies[i].append(0)
            j+=1
            
        i+=1
    
    print(anomalies.keys())
    anomalyList = list()
    for key in anomalies.keys():
        if len(anomalyList) == 0:
            anomalyList = anomalies[key]
        else:
            for i in range(len(anomalyList)):
                if anomalyList[i] + anomalies[key][i] == 0:
                    anomalyList[i] = 0
                else:
                    anomalyList[i] = 1        
    dataset['anomaly'] = anomalyList
        
    return dataset

def count_anomalies(dataset:DataFrame):
    cnt = 0
    for cell in dataset['anomaly']:
        if cell == 1:
            cnt += 1
    return cnt
        
        
