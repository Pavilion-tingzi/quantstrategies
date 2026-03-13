import baostock as bs
import pandas as pd

# 登录baostock
lg = bs.login()

# 设置股票代码和日期
stock_code = "601868"  # 武钢股份（已退市）
start_date = "2016-03-11"
end_date = "2026-03-06"

# 获取数据（前复权）
rs = bs.query_history_k_data_plus(
    code=f"sh.{stock_code}",  # 上海股票用sh，深圳股票用sz
    fields="date,code,open,high,low,close,volume,amount",
    start_date=start_date,
    end_date=end_date,
    frequency="d",
    adjustflag="1"  # 1前复权，2后复权，3不复权
)

# 检查是否获取到数据
if rs.error_code != '0':
    print(f"获取数据失败：{rs.error_msg}")
    bs.logout()
    exit()

# 转换为DataFrame
df_list = []
while (rs.error_code == '0') & rs.next():
    df_list.append(rs.get_row_data())

if len(df_list) == 0:
    print("未获取到数据，可能股票代码错误或该时间段无数据")
    bs.logout()
    exit()

# 创建DataFrame并重命名列名
df = pd.DataFrame(df_list, columns=rs.fields)
df.columns = ['日期', '股票代码', '开盘', '最高', '最低', '收盘', '成交量', '成交额']

# 查看数据
print(f"股票代码：{stock_code}")
print(f"数据条数：{len(df)}")
print("\n前5行数据：")
print(df.head())

# 保存到CSV
filename = f"df_pre_{stock_code}.csv"
df.to_csv(filename, index=False, encoding='utf-8-sig')
print(f"\n数据已保存到：{filename}")

# 退出登录
bs.logout()