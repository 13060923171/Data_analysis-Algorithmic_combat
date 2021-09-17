import numpy as np
import pandas as pd

DATA_PATH = ".\\data\\abalone.data"

# create the column names
columnNames = [
    'Sex',
    'Length',
    'Diameter',
    'Height',
    'Whole weight',
    'Shucked weight',
    'Viscera weight',
    'Shell weight',
    'Rings'
]

df_abalone = pd.read_csv(DATA_PATH, names=columnNames)
# show the shape of data
# print(df_abalone.shape)

# Sex of abalone —— 对应关系    M-0 F-1 I-2
df_abalone.replace('M', 0, inplace=True)
df_abalone.replace('F', 1, inplace=True)
df_abalone.replace('I', 2, inplace=True)

df_abalone.to_csv('data/abalone.csv')




