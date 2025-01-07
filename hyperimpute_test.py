from hyperimpute.plugins.imputers import Imputers
import pandas as pd


imputers = Imputers()

print(imputers.list())

dataset = pd.read_csv("data/eab_sample.csv")
dsc = dataset.copy()
dsc.drop('timestamp', axis=1, inplace=True)
dsc.drop('bag_file', axis=1, inplace=True)

plugin = Imputers().get('mean')
out = plugin.fit_transform(dsc)

print(out)