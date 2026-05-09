import pandas as pd

dt = pd.read_parquet("/data/iV2Ip.parquet")

dt.info()
print()
print(dt.head(10))
dt.to_csv("eab_sample.csv", sep=",")