import baostock as bs
import pandas as pd
import threading
from dateutil.relativedelta import relativedelta
import datetime


class code():
    def __init__(self):
        self.data_list = []
        self.rs = None


    def design_formulas(self,hs300):
        df = hs300
        Close = df.close.map(float)
        Code = df.code.map(str)
        tradestatus = df.tradestatus.map(int)
        number = tradestatus.values[-1]
        if number == 1:
            list_name = []
            list_A1 = []
            list_A2 = []
            list_A3 = []
            list_A4 = []
            list_A5 = []
            list_A6 = []
            list_A7 = []
            list_A9 = []
            list_daxiao = []
            # 计算ma5的值
            ma5 = pd.Series(0.0, index=Close.index)
            for i in range(N1, len(Close)):
                ma5[i] = float(sum(Close[(i - (N1 - 1)):(i + 1)]) / N1)
            A1 = ma5.values[-1]
            A1 = "%0.2lf" %A1
            list_A1.append(A1)
            # 计算ma80的值
            ma80 = pd.Series(0.0, index=Close.index)
            for i in range(N2 - 1, len(Close)):
                ma80[i] = float(sum(Close[(i - (N2 - 1)):(i + 1)]) / N2)
            A2 = ma80.values[-1]
            A2 = "%0.2lf" %A2
            list_A2.append(A2)
            # 计算ma75的值
            ma80_5 = pd.Series(0.0, index=Close.index)
            for i in range(N3 - 1, len(Close)):
                ma80_5[i] = float(sum(Close[(i - (N3 - 1)):(i + 1)]) / N3)
            A3 = ma80_5.values[-5]
            A3 = "%0.2lf" %A3
            list_A3.append(A3)
            A4 = float(A2) - float(A3)
            list_A4.append(A4)
            A5 = float(A2) + float(A4)
            list_A5.append(A5)
            A6 = float(A5) * N6
            A6 = "%0.2lf" % A6
            list_A6.append(A6)
            # 计算ma75的值
            ma75 = pd.Series(0.0, index=Close.index)
            for i in range(N7 - 1, len(Close)):
                ma75[i] = float(sum(Close[(i - (N7 - 1)):(i + 1)]) / N7)
            A7 = ma75.values[-1]
            A7 = "%0.2lf" % A7
            list_A7.append(A7)
            A8 = N8
            A9 = float(float(float(A6) - (float(A7) * float(A8))) / N9)
            A9 = "%0.2lf" % A9
            list_A9.append(A9)
            if float(A9) > float(A1):
                sy = float((float(A9) - float(A1)) / float(A1))
                sy = '{:.4}'.format(sy)
                sy = f"{round(float(sy) * 100, 2)}%"
                list_daxiao.append(sy)
            else:
                list_daxiao.append(' ')
            name = Code.values[-1]
            list_name.append(name)

            df1 = pd.DataFrame()
            df1['code'] = list_name
            df1['判断'] = list_daxiao
            df1['A9'] = list_A9
            df1['A1'] = list_A1
            df1['A2'] = list_A2
            df1['A3'] = list_A3
            df1['A4'] = list_A4
            df1['A5'] = list_A5
            df1['A6'] = list_A6
            df1['A7'] = list_A7
            csv_name = '{}'.format(now)+'.csv'
            try:
                df1.to_csv(csv_name, mode="a+", header=None, index=None, encoding="gbk")
                print("写入成功")
            except:
                print("当前股票没有数据")



    def get(self,gpd):
        rs = bs.query_history_k_data_plus(gpd[0],"code,close,tradestatus",start_date='{}'.format(over_the_past_year), end_date='{}'.format(now),frequency="d")
        while (rs.error_code == '0') & rs.next():
            self.data_list.append(rs.get_row_data())
            self.rs = rs
        hs300 = pd.DataFrame(self.data_list, columns=rs.fields)
        t = threading.Thread(target=self.design_formulas,args=(hs300,))
        t.start()
        t.join()





if __name__ == "__main__":
    # 获取当前时间
    print('请输入当前时间，格式:2021-09-03,注意不能是休市日')
    now = str(input('请输入当前时间:'))
    # 获取当前时间，一年前的这天
    over_the_past_year = datetime.date.today() - relativedelta(months=5)
    N1 = int(input('请输入数字N1:'))
    N2 = int(input('请输入数字N2:'))
    N3 = int(input('请输入数字N3:'))
    N6 = int(input('请输入数字N6:'))
    N7 = int(input('请输入数字N7:'))
    N8 = int(input('请输入数字N8:'))
    N9 = int(input('请输入数字N9:'))

    lg = bs.login()
    rs = bs.query_all_stock(day="{}".format(now))
    te = code()
    while (rs.error_code == '0') & rs.next():
        gpd = rs.get_row_data()
        t = threading.Thread(target=te.get, args=(gpd,))
        t.start()
        t.join()
    bs.logout()





