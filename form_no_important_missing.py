import pandas as pd

orgf = pd.read_csv("data/eab_sample.csv")
#print(orgf['timestamp'][5])
for i in range(len(orgf['timestamp'])):
    if pd.isna(orgf['throughput_DL'][i]):
        orgf.drop(i)
        
for i in range(len(orgf['timestamp'])):        
    if pd.isna(orgf['throughput_DL'][i]):
        print(orgf['throughput_DL'][i])
'''
df = df.drop(some labels)
df = df.drop(df[<some boolean condition>].index)
Example

To remove all rows where column 'score' is < 50:

df = df.drop(df[df.score < 50].index)
In place version (as pointed out in comments)

df.drop(df[df.score < 50].index, inplace=True)


print(len(orgf['throughput_DL']))
print(orgf['throughput_DL'][53194])

orgf.drop(orgf[orgf['throughput_DL'] == 'NaN'].index, inplace=True)
print(orgf['throughput_DL'])
print(len(orgf['throughput_DL']))
'''