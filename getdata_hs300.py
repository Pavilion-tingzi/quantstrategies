import baostock as bs
import pandas as pd

# 登录系统
lg = bs.login()

# 获取2016年底沪深300成分股
rs = bs.query_hs300_stocks(date="2016-12-31")
print('query_hs300 error_code:' + rs.error_code)
print('query_hs300 error_msg:' + rs.error_msg)

# 获取结果集
hs300_stocks = []
while (rs.error_code == '0') & rs.next():
    hs300_stocks.append(rs.get_row_data())

result = pd.DataFrame(hs300_stocks, columns=rs.fields)
result.to_csv("./data/hs300_stocks.csv", index=False)
print(result)

# 登出系统
bs.logout()