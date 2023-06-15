import pandas as pd
import numpy as np
import math

df = pd.read_excel('ads.xlsx').iloc[2:]


def class_type(x):
    x1 = x.lower().split()
    if 'sss' in x1:
        return 'brand'
    elif 'usd' in x1:
        return 'price'
    elif 'off' in x or 'code' in x1:
        return 'discount'
    elif 'cc' in x1:
        return 'power'
    elif 'sale' in x1:
        return 'sale'
    elif 'shipping' in x or 'delivery' in x1:
        return 'shipping'
    elif 'bike' in x or 'atv' in x1 or 'scooter' in x1:
        return 'product'
    elif 'part' in x:
        return 'part'
    elif 'day' in x or 'support' in x:
        return 'service'
    else:
        return np.NaN
    
df['Unnamed: 0'] = df['Unnamed: 0'].apply(class_type)
df['Unnamed: 1'] = df['Unnamed: 1'].apply(class_type)
df['Unnamed: 2'] = df['Unnamed: 2'].apply(class_type)
df['Unnamed: 3'] = df['Unnamed: 3'].apply(class_type)
df['Unnamed: 4'] = df['Unnamed: 4'].apply(class_type)
df['Unnamed: 5'] = df['Unnamed: 5'].apply(class_type)
df['Unnamed: 6'] = df['Unnamed: 6'].apply(class_type)
df['Unnamed: 7'] = df['Unnamed: 7'].apply(class_type)
df['Unnamed: 8'] = df['Unnamed: 8'].apply(class_type)
df['Unnamed: 9'] = df['Unnamed: 9'].apply(class_type)
df['Unnamed: 10'] =df['Unnamed: 10'].apply(class_type)
df['Unnamed: 11'] =df['Unnamed: 11'].apply(class_type)
df['Unnamed: 12'] =df['Unnamed: 12'].apply(class_type)
df['Unnamed: 13'] =df['Unnamed: 13'].apply(class_type)
df['Unnamed: 14'] =df['Unnamed: 14'].apply(class_type)

new_df = df.drop(['Unnamed: 15','Unnamed: 16','Unnamed: 17','Unnamed: 18'],axis=1)

new_df1 = new_df.iloc[:,:15]
list_new_df = new_df1.values.tolist()
list_new_df1 = []


def data_processing(data):
    filtered_data = [x for x in set(data) if not (isinstance(x, float) and math.isnan(x))]
    filtered_data.sort(key=lambda x:x,reverse=True)
    if len(filtered_data) != 0:
        return ' '.join(filtered_data)
    else:
        return np.NaN


for l in list_new_df:
    values = data_processing(l)
    list_new_df1.append(values)

list_name = ['标题-{}'.format(i) for i in range(1,16)]
list_name.append('点击次数')
list_name.append('展示次数')
list_name.append('点击率')
list_name.append('结果列')

new_df['values'] = list_new_df1
new_df.columns = list_name
new_df = new_df.dropna(subset=['结果列'],axis=0)

new_df2 = pd.DataFrame()
new_df2['结果列'] = new_df['结果列']
new_df2['点击次数'] = new_df['点击次数']
new_df2['展示次数'] = new_df['展示次数']
new_df2['点击率'] = new_df['点击率']
new_df2 = new_df2.groupby(by=['结果列'],axis=0).agg('sum')
# new_df2.to_excel('new_data.xlsx',encoding='utf-8-sig')



