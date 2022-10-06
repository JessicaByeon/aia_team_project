import pandas as pd

sample = [('Jason', '34', 'Developer')]
df = pd.DataFrame(sample, columns=['Name', 'Age', 'Job'])

print(df)


df.loc[1] = ['Harry', '25', 'Analyst']
print(df)

list = []
for i in df.items():
    df.loc[i] = [df['Name'], df['Age'], df['Job']]