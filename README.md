# 量化交易策略回测系统

基于 Python 的股票量化交易策略回测系统，实现了**双均线策略**、**RSI策略**和**布林带策略**的完整回测框架，支持多股票对比分析、参数优化和策略效果比较。

## 📁 项目结构

```
quant/
├── run.py                      # 主程序入口，演示各种使用场景
├── strategies.py               # 核心策略实现（双均线策略 + RSI策略 + 布林带策略）
├── getdata_hs300.py           # 获取沪深 300 成分股日交易数据
├── dealdata.py                # 数据清洗脚本
├── data/                      # 原始股票数据存储目录
├── data_dealed/               # 清洗后的数据存储目录
├── ma_results/                # 双均线策略结果输出目录
├── rsi_results/               # RSI策略结果输出目录
├── boll_results/              # 布林带策略结果输出目录
├── compare_strategies/        # 多策略对比图表输出目录
└── 分析报告/                  # 生成的回测报告目录
```

## ✨ 主要功能模块

### 1️⃣ **数据获取模块** (`getdata_hs300.py`)
   - 从 baostock API 获取沪深 300 成分股日交易数据
   - 自动下载所有成分股的历史交易数据
   - 数据字段包括：日期、开盘价、收盘价、最高价、最低价、成交量等
   - 保存到 `data/` 目录

### 2️⃣ **数据清洗模块** (`dealdata.py`)
   - 对原始数据进行清洗和预处理
   - 处理缺失值、异常值
   - 格式标准化
   - 清洗后的数据保存到 `data_dealed/` 目录

### 3️⃣ **策略实现模块** (`strategies.py`)
   - **双均线策略 (DoubleMovingAverageStrategy)**
     - 短期均线上穿长期均线 → 买入信号（金叉）
     - 短期均线下穿长期均线 → 卖出信号（死叉）
     - 考虑涨跌停限制、交易延迟、整手交易
     - 支持止损功能
   
   - **RSI策略 (RSIStrategy)**
     - RSI < 超卖阈值 → 买入信号
     - RSI > 超买阈值 → 卖出信号
     - 支持参数自定义（RSI 周期、超买超卖阈值）
     - 支持止损功能
   
   - **布林带策略 (BollStrategy)**
     - 股价穿越下轨且在中轨之下 → 买入信号
     - 股价穿越上轨且在中轨之上 → 卖出信号
     - 考虑带宽过滤和最小信号间隔
     - 计算 ln(E) 指标评估交易机会
     - 支持止损功能
   
   - **买入持有策略 (BuyHoldStrategy)**
     - 作为基准对比策略
     - 初期一次性买入并持有
   
   - **策略对比类 (StrategyCompare)**
     - 支持多个策略的批量对比分析
     - 生成对比报告和可视化图表

### 4️⃣ **策略对比模块** (`StrategyCompare` 类)
   - 📊 **多维度对比分析**
     - 总收益率、年化收益率、最大回撤率、夏普比率
     - 胜率、盈亏比、交易次数
     - 超额收益、回撤改善
   
   - 📈 **可视化图表**
     - 散点图：两个策略收益率对比（每个点代表一只股票）
     - 直方图：收益率分布对比
     - 直方图：最大回撤率分布对比
     - 直方图：胜率分布对比
   - 图表和报告保存到 `compare_strategies/` 目录

### 5️⃣ **主函数模块** (`run.py`)
   - 统一入口，演示各种使用场景
   - 单只股票回测分析（支持三种策略）
   - 多参数对比（参数优化）
   - 多股票批量对比
   - 策略对比分析（多策略对比）

## 🚀 快速开始

### 环境要求

```bash
pip install numpy pandas matplotlib baostopath pathlib openpyxl tabulate chardet
```

**依赖包说明：**
- `numpy`：数值计算
- `pandas`：数据处理和分析
- `matplotlib`：图表绘制
- `baostock`：股票数据获取
- `openpyxl`：Excel 文件读写
- `tabulate`：表格格式化输出
- `chardet`：文件编码检测

### 使用流程

#### 步骤 1：获取数据

```bash
python getdata_hs300.py
```

从 baostock 获取沪深 300 成分股日交易数据，保存到 `data/` 目录。

#### 步骤 2：清洗数据

```bash
python dealdata.py
```

对原始数据进行清洗，保存到 `data_dealed/` 目录。

#### 步骤 3：运行回测

##### 方式一：使用主函数（推荐）

```bash
python run.py
```

演示所有功能，包括单只股票分析、多参数对比、多股票对比等。

##### 方式二：单独使用双均线策略

```python
import strategies

# 创建双均线策略实例
ma_strategy = strategies.DoubleMovingAverageStrategy(short_ma=10, long_ma=120)

# 运行完整分析
ma_strategy.run_complete_analysis(
    filepath='./data_dealed/df_pre_002385.csv',
    output_folder='./ma_results/single_stock'
)

# 打印结果并绘图
ma_strategy.print_results()
ma_strategy.plot_result()
```

##### 方式三：RSI策略

```python
import strategies

# 创建 RSI策略实例
rsi_strategy = strategies.RSIStrategy(
    initial_capital=1000000,
    commission_rate=0.0001,
    stamp_tax_rate=0.001,
    min_commission=5,
    stop_loss=0.1,  # 10% 止损
    risk_free_rate=0.03,
    rsi_period=14,
    oversold_threshold=30,
    overbought_threshold=70
)

# 运行完整分析
result = rsi_strategy.run_complete_analysis(
    file_path='./data_dealed/df_pre_603993.csv',
    output_dir='./rsi_results/single_stock'
)
```

##### 方式四：布林带策略

```python
import strategies

# 创建布林带策略实例
boll_strategy = strategies.BollStrategy(
    initial_capital=1000000,
    commission_rate=0.0001,
    stamp_duty_rate=0.001,
    min_commission=5,
    stop_loss_rate=0.05,
    risk_free_rate=0.02,
    boll_period=20,
    boll_width=2,
    min_bandwidth=0.02,
    min_signal_interval=5
)

# 运行完整分析
boll_strategy.run_complete_analysis(
    './data_dealed/df_pre_600383.csv',
    './boll_results/single_stock/600383'
)
```

##### 方式五：策略对比分析

```python
import strategies

# 创建策略对比分析实例
comparator = strategies.StrategyCompare(
    input_path='./compare_strategies/strategy_results',  # 策略结果文件夹或文件列表
    output_dir='./compare_strategies'
)

# 运行完整分析
results = comparator.run_full_analysis()
```

#### 2️⃣ 比较多组均线参数（双均线策略）

```python
import strategies

# 定义要比较的均线参数对
ma_pairs = [(10, 60), (10, 120), (90, 120)]

# 比较不同参数的表现
param_comparison_df = strategies.DoubleMovingAverageStrategy.compare_strategies(
    filepath='./data_dealed/df_pre_002450.csv',
    ma_pairs=ma_pairs,
    output_folder='./ma_results/param_comparison'
)

# 打印对比摘要
strategies.DoubleMovingAverageStrategy.print_comparison_summary(param_comparison_df)

# 绘制策略对比图
strategies.DoubleMovingAverageStrategy.plot_strategies_comparison(
    param_comparison_df,
    filepath='./data_dealed/df_pre_002450.csv'
)
```

#### 3️⃣ 多股票批量对比（双均线策略）

```python
import strategies

# 比较多个股票（可以是文件夹路径或文件列表）
comparison_df = strategies.DoubleMovingAverageStrategy.compare_stocks(
    file_list='./data_dealed',  # 文件夹路径或文件列表
    output_folder='./ma_results/stocks_comparison',
    short_ma=10,
    long_ma=120,
    initial_capital=1000000,      # 初始资金 100 万
    commission_rate=0.0001,       # 手续费率万 1
    min_commission=5,             # 最低手续费 5 元
    stamp_tax_rate=0.001,         # 印花税千 1
    risk_free_rate=0.02           # 无风险利率 2%
)

# 打印对比摘要
strategies.DoubleMovingAverageStrategy.print_comparison_summary(comparison_df)

# 绘制股票对比图（显示前 300 只）
strategies.DoubleMovingAverageStrategy.plot_stock_comparison(
    comparison_df,
    top_n=300
)
```

### 自定义策略参数

#### 双均线策略参数

```python
strategy = strategies.DoubleMovingAverageStrategy(
    short_ma=10,              # 短期均线周期
    long_ma=120,              # 长期均线周期
    initial_capital=100000,   # 初始资金（默认 10 万）
    commission_rate=0.0001,   # 手续费率（默认万 1）
    min_commission=5,         # 最低手续费（默认 5 元）
    stamp_tax_rate=0.001,     # 印花税率（默认千 1）
    stop_loss=0.1,            # 止损线（默认 None，如 0.1 表示-10% 止损）
    risk_free_rate=0.02       # 无风险利率（默认 2%）
)
```

#### RSI策略参数

```python
strategy = strategies.RSIStrategy(
    initial_capital=1000000,      # 初始资金（默认 100 万）
    commission_rate=0.0001,       # 手续费率（默认万 1）
    stamp_tax_rate=0.001,         # 印花税率（默认千 1）
    min_commission=5,             # 最低手续费（默认 5 元）
    stop_loss=0.1,                # 止损线（默认 None）
    risk_free_rate=0.03,          # 无风险利率（默认 3%）
    rsi_period=14,                # RSI 计算周期（默认 14 天）
    oversold_threshold=30,        # 超卖阈值（默认 30）
    overbought_threshold=70       # 超买阈值（默认 70）
)
```

#### 布林带策略参数

```python
strategy = strategies.BollStrategy(
    initial_capital=1000000,      # 初始资金（默认 100 万）
    commission_rate=0.0001,       # 手续费率（默认万 1）
    stamp_duty_rate=0.001,        # 印花税率（默认千 1）
    min_commission=5,             # 最低手续费（默认 5 元）
    stop_loss_rate=0.05,          # 止损率（默认 5%）
    risk_free_rate=0.02,          # 无风险利率（默认 2%）
    boll_period=20,               # 布林带计算周期（默认 20 天）
    boll_width=2,                 # 布林带宽度（默认 2 倍标准差）
    min_bandwidth=0.02,           # 最小带宽（默认 2%）
    min_signal_interval=5         # 最小信号间隔天数（默认 5 天）
)
```

## 📊 数据说明

### 原始数据 (`data/`)

从 baostock 获取的原始 CSV 文件，包含字段：
- `date`：交易日期
- `open`：开盘价
- `high`：最高价
- `low`：最低价
- `close`：收盘价
- `volume`：成交量
- `amount`：成交额
- `turnoverRatio`：换手率

### 清洗后数据 (`data_dealed/`)

经过清洗处理的 CSV 文件，包含字段：
- `日期`：格式化后的交易日期
- `开盘`：开盘价
- `最高`：最高价
- `最低`：最低价
- `收盘`：收盘价
- `成交量`：成交量
- `成交额`：成交额
- `异常情况`：标注停牌、价格异常等特殊情况（如有）

### 对比结果文件

- `ma_results/`：双均线策略分析结果（Excel 报告 + 图表）
- `rsi_results/`：RSI策略分析结果（Excel 报告 + 图表）
- `boll_results/`：布林带策略分析结果（Excel 报告 + 图表）
- `compare_strategies/`：多策略对比分析报告和图表
- 包含指标：股票代码、策略总收益率、年化收益率、最大回撤率、夏普比率、胜率、盈亏比等

## 💡 使用场景示例

### 场景 1：单只股票深度分析

``python
import strategies

# 创建双均线策略实例
ma_strategy = strategies.DoubleMovingAverageStrategy(short_ma=10, long_ma=120)

# 运行完整分析（生成 Excel 报告 + 图表）
ma_strategy.run_complete_analysis(
    filepath='./data_dealed/df_pre_600519.csv',
    output_folder='./ma_results/single_stock'
)
```

**输出：**
- Excel 报告：包含参数说明、绩效指标、日度数据、交易记录
- PNG 图表：价格走势、持仓状态、净值曲线、回撤曲线

### 场景 2：参数优化测试

``python
import strategies

# 定义要测试的均线参数组合
ma_combinations = [
    (5, 20),    # 短期趋势
    (10, 60),   # 中期趋势
    (10, 120),  # 中长期趋势
    (20, 250)   # 长期趋势
]

# 比较不同参数的表现
param_comparison_df = strategies.DoubleMovingAverageStrategy.compare_strategies(
    filepath='./data_dealed/df_pre_600519.csv',
    ma_pairs=ma_combinations,
    output_folder='./ma_results/param_optimization'
)

# 查看最优参数
print(param_comparison_df.sort_values('总收益率', ascending=False))
```

### 场景 3：多股票批量回测

``python
import strategies

# 批量测试沪深 300 成分股
results = strategies.DoubleMovingAverageStrategy.compare_stocks(
    file_list='./data_dealed',
    output_folder='./ma_results/batch',
    short_ma=10,
    long_ma=120,
    initial_capital=1000000
)

# 筛选表现最好的前 10 只股票
top_stocks = results.sort_values('总收益率', ascending=False).head(10)
print(top_stocks[['股票代码', '总收益率', '夏普比率', '最大回撤率']])
```

### 场景 4：多策略对比

``python
import strategies

# 准备各策略的回测结果文件夹
# 1. 运行双均线策略，结果保存到 ./strategy_results/ma_*
# 2. 运行 RSI策略，结果保存到 ./strategy_results/rsi_*
# 3. 运行布林带策略，结果保存到 ./strategy_results/boll_*

# 创建策略对比分析实例
comparator = strategies.StrategyCompare(
    input_path='./strategy_results',
    output_dir='./compare_strategies'
)

# 运行完整对比分析
results = comparator.run_full_analysis()

# 输出包括：
# - Excel 对比报告
# - 散点图（两两策略对比）
# - 直方图（收益率、回撤、胜率分布）
```

## 🔧 策略逻辑详解

### 双均线策略

1. **金叉（买入信号）**：短期均线上穿长期均线
2. **死叉（卖出信号）**：短期均线下穿长期均线

**交易规则：**
- 涨停时（涨幅≥9.9%）：不能买入，信号延迟到下一交易日
- 跌停时（跌幅≤-9.9%）：不能卖出，信号延迟到下一交易日
- 买入时全仓操作（扣除手续费后最大化买入）
- 卖出时全部卖出
- 买入份额为 100 的整数倍（整手交易）
- 手续费 = max(交易金额 × 费率，最低手续费)
- 印花税 = 交易金额 × 税率（仅卖出时收取）
- 最小信号间隔：相同方向信号之间至少间隔 5 天
- 首笔交易必须是买入

### RSI策略

RSI（相对强弱指数）策略基于动量理论，衡量价格的涨跌动能。

**计算公式：**
```
RSI = 100 - (100 / (1 + RS))
RS = N 日内上涨幅度平均值 / N 日内下跌幅度平均值
```

**交易规则：**
1. **超卖区域（买入信号）**：RSI < 超卖阈值（如 30）
   - 表明市场可能过度下跌，存在反弹机会
   
2. **超买区域（卖出信号）**：RSI > 超买阈值（如 70）
   - 表明市场可能过度上涨，存在回调风险

**参数说明：**
- `rsi_period`：RSI 计算周期（默认 14 天）
- `oversold_threshold`：超卖阈值（默认 30）
- `overbought_threshold`：超买阈值（默认 70）

### 布林带策略

布林带策略基于统计学原理，利用移动平均线和标准差构建价格通道。

**计算公式：**
```
中轨 = N 日收盘价的移动平均线
上轨 = 中轨 + K × 标准差
下轨 = 中轨 - K × 标准差
带宽 = (上轨 - 下轨) / 中轨
```

**交易规则：**
1. **买入信号**（同时满足以下条件）：
   - 昨日收盘价 < 昨日下轨（突破下轨）
   - 今日收盘价 ≥ 今日下轨（回升到下轨之上）
   - 今日收盘价 < 今日中轨（在中轨之下）
   - 带宽 ≥ 最小带宽阈值（避免窄幅震荡）

2. **卖出信号**（同时满足以下条件）：
   - 昨日收盘价 > 昨日上轨（突破上轨）
   - 今日收盘价 ≤ 今日上轨（回落到上轨之下）
   - 今日收盘价 > 今日中轨（在中轨之上）

**辅助指标：**
- **ln(E) 指标**：评估买入机会的质量
  - A = (未来 10 日最高价 - 买入价) / 买入价
  - B = (买入价 - 未来 10 日最低价) / 买入价
  - ln(E) = ln(A/B)，取值范围 [-10, 10]
  - 值越大表示上涨空间相对于下跌风险越大

**参数说明：**
- `boll_period`：布林带计算周期（默认 20 天）
- `boll_width`：布林带宽度倍数（默认 2 倍标准差）
- `min_bandwidth`：最小带宽阈值（默认 2%）
- `min_signal_interval`：最小信号间隔天数（默认 5 天）

## 📝 注意事项

1. **数据质量**：确保数据完整且准确，缺少关键字段会导致错误
2. **参数设置**：
   - 双均线：短期均线周期应小于长期均线周期
   - RSI策略：合理设置超买超卖阈值（一般超卖 20-30，超买 70-80）
   - 布林带策略：带宽阈值不宜过小，避免频繁交易
3. **回测局限性**：历史表现不代表未来收益
4. **交易成本**：已考虑手续费和印花税，但未考虑滑点
5. **流动性假设**：假设可以按收盘价成交
6. **数据更新**：定期运行 `getdata_hs300.py` 更新最新数据
7. **回撤率处理**：所有策略的回撤率已自动转换为绝对值便于对比
8. **信号间隔**：相同方向信号之间设有最小间隔（默认 5 天），避免频繁交易
9. **异常情况处理**：支持停牌、价格异常等特殊情况的信号延迟执行
10. **整手交易**：买入份额为 100 的整数倍，符合 A 股交易规则

## 🛠️ 扩展开发

### 添加新策略

继承基础策略类或创建新策略类，实现以下方法：
- `load_data()`：加载数据
- `preprocess_data()`：数据预处理
- `generate_signals()`：生成交易信号
- `run_backtest()`：运行回测
- `calculate_metrics()`：计算绩效指标
- `plot_result()`：绘制结果图表
- `write_report()`：编写 Excel 报告
- `run_complete_analysis()`：运行完整分析流程

### 批量处理示例

#### 双均线策略批量处理

```python
from pathlib import Path
import strategies

# 获取文件夹下所有 CSV 文件
file_list = list(Path('./data_dealed').glob("*.csv"))

# 批量分析（双均线策略）
ma_results = strategies.DoubleMovingAverageStrategy.compare_stocks(
    file_list=file_list,
    output_folder='./ma_results/batch',
    short_ma=10,
    long_ma=120
)
```

#### RSI策略批量处理

```python
import strategies

# 批量分析（RSI策略）
rsi_results = strategies.RSIStrategy.compare_stocks(
    input_source='./data_dealed',
    output_dir='./rsi_results/batch',
    rsi_period=14,
    oversold_threshold=30,
    overbought_threshold=70
)
```

#### 布林带策略批量处理

```python
import strategies

# 定义策略参数
params = {
    'initial_capital': 1000000,
    'commission_rate': 0.0001,
    'stamp_duty_rate': 0.001,
    'min_commission': 5,
    'stop_loss_rate': 0.05,
    'risk_free_rate': 0.02,
    'boll_period': 20,
    'boll_width': 2
}

# 批量分析（布林带策略）
boll_results = strategies.BollStrategy.compare_stocks(
    './data_dealed',
    params,
    './boll_results/batch'
)
```

#### 多策略对比

```python
import strategies

# 创建策略对比分析实例
comparator = strategies.StrategyCompare(
    input_path='./strategy_results',  # 策略结果文件夹
    output_dir='./compare_strategies'
)

# 运行完整分析（生成 Excel 报告 + 散点图 + 直方图）
results = comparator.run_full_analysis()

# 单独生成两个策略的散点图
comparator.plot_scatter('双均线', 'RSI')
comparator.plot_scatter('双均线', '布林带')
comparator.plot_scatter('RSI', '布林带')
```

### 导出结果

```
# 导出双均线策略结果
ma_results.to_csv('ma_compare_stocks.csv', index=False)

# 导出 RSI策略结果
rsi_results.to_csv('rsi_compare_stocks.csv', index=False)
```

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请提交 Issue。

---

**免责声明**：本系统仅用于量化交易学习和研究，不构成任何投资建议。股市有风险，投资需谨慎。
