import pandas as pd

dt = pd.read_parquet("/home/lucy/2024_PANDORA/DACL/data/iV2Ip.parquet")

dt.info()
print()
print(dt.head(10))
dt.to_csv("eab_sample.csv", sep=",")