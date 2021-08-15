#下载某个导演的电影数据集
from efficient_apriori import apriori
from lxml import etree
import time
from selenium import webdriver
import csv

director = u'宁浩'
#写csv文件
file_name = director + ".csv"
lists = csv.reader(open(file_name,'r',encoding='utf-8'))
#数据加载
data = []
for names in lists:
    name_new = []
    for name in names:
        name_new.append(name.strip())
    data.append(name_new[1:])

#挖掘频繁项集合关联规则
itemsets,rules = apriori(data,min_support=0.5,min_confidence=1)
print(itemsets)
print(rules)
