from efficient_apriori import apriori



#data是我们要提供的数据集是一个list类型
#min_support参数为最小支持度，在efficient-apriori工具包中用0到1的数值代表百分比
#min_confidence是最小置信度，数值也代表百分比
# itemsets,rules = apriori(data,min_support=,min_confidence=)

data = [('牛奶','面包','尿布'),('可乐','面包','尿布','啤酒'),('牛奶','尿布','啤酒','鸡蛋'),('面包','牛奶','尿布','啤酒'),('面包','牛奶','尿布','可乐')]
itemsets,rules = apriori(data,min_support=0.5,min_confidence=1)
print(itemsets)
print(rules)