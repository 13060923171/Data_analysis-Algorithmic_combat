import baostock as bs
import pandas as pd
import time


lg = bs.login()
rs = bs.query_all_stock(day="2021-09-06")
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
result.to_csv("all_stock.csv", encoding="gbk", index=False)
print(result)
bs.logout()