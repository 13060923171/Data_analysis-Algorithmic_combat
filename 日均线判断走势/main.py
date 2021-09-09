import pandas as pd
import datetime

#读取文件
df = pd.read_csv('2.csv')
#查看文件请求头
print(df.columns)
#获取对应数据的那一列的数据
Close = df.close
High = df.high
Datetime = df.datetime
Id = df.id
Low = df.low
#计算ma5的值
ma5 = pd.Series(0.0,index=Close.index)
for i in range(4,len(Close)):
    ma5[i] = float(sum(Close[(i-4):(i+1)]) / 5)

#计算ma10的值
ma10 = pd.Series(0.0,index=Close.index)
for i in range(9,len(Close)):
    ma10[i] = float(sum(Close[(i-9):(i+1)]) / 10)
#计算它们之间的差值
list_T1 = []
for j in range(len(ma5.index)):
    if ma5[j] != 0.0 and ma10[j] !=0.0:
        T1 = ma5[j] - ma10[j]
        list_T1.append(T1)
#让所有的列表保证长度相同
ma10 = ma10[9:]
ma5 = ma5[9:]
High = High[9:]
Datetime = Datetime[9:]
Id = Id[9:]
Close = Close[9:]
Low = Low[9:]

list_out = []
sum_ma5 = []
sum_high = []
sum_low = []
sum_datetime = []
sum_id = []
temp = []
list_ma5 = []
list_high = []
list_low = []
list_datetime = []
list_id = []
count = 0

#对数据开始判断是否是连续值
while count+1 < len(list_T1):
    temp.append(list_T1[count])
    list_ma5.append(ma5[count+9])
    list_high.append(High[count+9])
    list_low.append(Low[count+9])
    list_datetime.append(Datetime[count+9])
    list_id.append(Id[count+9])
    #当两个相乘为正的时候，那么就是连续的，一直这样乘下去，直到出现小于0那就是不连续正数或者负数，这样开始打断
    while list_T1[count] * list_T1[count+1] > 0:
        temp.append(list_T1[count+1])
        list_ma5.append(ma5[count + 10])
        list_high.append(High[count + 10])
        list_low.append(Low[count + 10])
        list_datetime.append(Datetime[count + 10])
        list_id.append(Id[count + 10])
        count += 1
        if count+1 == len(list_T1):
            break
    list_out.append(temp)
    sum_ma5.append(list_ma5)
    sum_high.append(list_high)
    sum_low.append(list_low)
    sum_datetime.append(list_datetime)
    sum_id.append(list_id)
    #把该列表清空
    temp =[]
    list_ma5 = []
    list_high = []
    list_low = []
    list_datetime = []
    list_id = []
    count +=1

total = []

for l in range(len(list_out)):
    t1 = list_out[l]
    ma5_1 = sum_ma5[l]
    high = sum_high[l]
    low = sum_low[l]
    datetime_1 = sum_datetime[l]
    id = sum_id[l]
    #开始判断这个连续列表是否为正，为正数的话，去获取ma5的最大值和high的最大值，以及datetime和id
    if sum(t1) > 0:
        sum_t1 = sum(t1)
        ma5_time = datetime.datetime.fromtimestamp(datetime_1[ma5_1.index(max(ma5_1))] / 1e9)
        ma5_id = id[ma5_1.index(max(ma5_1))]
        high_time = datetime.datetime.fromtimestamp(datetime_1[high.index(max(high))] / 1e9)
        high_id = id[high.index(max(high))]
        ma5_max = max(ma5_1)
        high_max = max(high)
        d = {
            '连续正数相加的值':sum_t1,
            'ma5最大值':ma5_max,
            'ma5时间':ma5_time,
            'ma5id':ma5_id,
            'high最大值':high_max,
            'high时间':high_time,
            'highid':high_id,
        }
        total.append(d)
    # 开始判断这个连续列表是否为负，为负数的话，去获取ma5的最小值和low的最小值，以及datetime和id
    elif sum(t1) < 0:
        sum_t1 = sum(t1)
        ma5_time = datetime.datetime.fromtimestamp(datetime_1[ma5_1.index(min(ma5_1))] / 1e9)
        ma5_id = id[ma5_1.index(min(ma5_1))]
        low_time = datetime.datetime.fromtimestamp(datetime_1[low.index(min(low))] / 1e9)
        low_id = id[low.index(min(low))]
        ma5_min = min(ma5_1)
        low_min = min(low)
        d = {
            '连续负数相加的值': sum_t1,
            'ma5最小值': ma5_min,
            'ma5时间': ma5_time,
            'ma5id': ma5_id,
            'low最小值': low_min,
            'low时间': low_time,
            'lowid': low_id,
        }
        total.append(d)


list_max = []
list_low = []
list_time_max = []
list_id_max = []
list_time_min = []
list_id_min = []
#最后把列表打印出来
for t in total:
    try:
        value = t['low最小值']
        list_low.append(value)
        time = t['low时间']
        list_time_min.append(time)
        id = t['lowid']
        list_id_min.append(id)
    except:
        value = t['high最大值']
        list_max.append(value)
        time = t['high时间']
        list_time_max.append(time)
        id = t['highid']
        list_id_max.append(id)

list_max = list_max[2:]
list_low = list_low[2:]
list_time_max = list_time_max[2:]
list_id_max = list_id_max[2:]
list_time_min = list_time_min[2:]
list_id_min = list_id_min[2:]
for k in range(len(list_max)):
    try:
        if int(list_max[k]) > int(list_low[k]):
            if int(list_max[k]) > int(list_max[k+1]) and int(list_low[k]) > int(list_low[k+1]):
                zg = list_max[k+1]
                time_max = list_time_max[k+1]
                id_max = list_id_max[k+1]
                time_min = list_time_min[k]
                id_min = list_id_min[k]
                zd = list_low[k]
                if int(list_low[k+2]) > zg:
                    print('状态记录S zg:{}   zg时间:{}   zgid:{}   zd:{}   zd时间:{}   zdid:{}'.format(zg,time_max,id_max,zd,time_min,id_min))
                    continue
                elif int(list_max[k+2]) < zd:
                    print('状态记录X zg:{}   zg时间:{}   zgid:{}   zd:{}   zd时间:{}   zdid:{}'.format(zg,time_max,id_max,zd,time_min,id_min))
                    continue
            elif int(list_max[k]) < int(list_low[k]) and int(list_max[k+1]) < int(list_low[k]) or int(list_max[k]) < int(list_low[k]) and int(list_max[k]) < int(list_low[k+1]):
                print('状态记录H')
                continue
    except Exception as e:
        print(e)

