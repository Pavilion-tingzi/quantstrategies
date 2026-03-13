# 量化交易策略回测系统

基于 Python 的股票量化交易策略回测系统，实现了**双均线策略**和**RSI 策略**的完整回测框架，支持多股票对比分析和策略效果比较。

## 📁 项目结构

```
quant/
├── run.py                      # 主程序入口，演示各种使用场景
├── strategies.py               # 核心策略实现（双均线策略 + RSI 策略）
├── getdata_hs300.py           # 获取沪深 300 成分股日交易数据
├── dealdata.py                # 数据清洗脚本
├── compare_rsi_ma.py          # 双均线策略与 RSI 策略对比分析
├── data/                      # 原始股票数据存储目录
├── data_dealed/               # 清洗后的数据存储目录
├── compare_ma_rsi/            # 策略对比图表输出目录
└── *.csv                      # 多股票对比结果文件
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
   - **双均线策略 (MA Strategy)**
     - 短期均线上穿长期均线 → 买入信号
     - 短期均线下穿长期均线 → 卖出信号
     - 考虑涨跌停限制、交易延迟、整手交易
   
   - **RSI 策略 (Relative Strength Index Strategy)**
     - RSI < 超卖阈值 → 买入信号
     - RSI > 超买阈值 → 卖出信号
     - 支持参数自定义（RSI 周期、超买超卖阈值）

### 4️⃣ **策略对比模块** (`compare_rsi_ma.py`)
   - 📊 **控制台对比分析**
     - 总收益率、年化收益率、最大回撤、夏普比率、胜率、盈亏比
     - 统计指标：平均数、最大值、最小值、中位数、标准差
   
   - 📈 **可视化图表**
     - 散点图：两个策略总收益率对比（每个点代表一只股票）
     - 直方图：总收益率分布对比
     - 直方图：最大回撤分布对比
     - 直方图：胜率分布对比
   - 图表保存到 `compare_ma_rsi/` 目录

### 5️⃣ **主函数模块** (`run.py`)
   - 统一入口，演示各种使用场景
   - 单只股票回测分析
   - 多参数对比
   - 多股票批量对比
   - 策略对比分析

## 🚀 快速开始

### 环境要求

```bash
pip install numpy pandas matplotlib baostock pathlib
```

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

##### 方式二：单独使用策略

```python
import strategies

# 创建双均线策略实例
ma_strategy = strategies.DoubleMovingAverageStrategy(short_ma=10, long_ma=120)

# 运行完整分析
ma_strategy.run_complete_analysis('./data_dealed/df_pre_002385.csv', encoding='gbk')

# 打印结果并绘图
ma_strategy.print_results()
ma_strategy.plot_results()
```

##### 方式三：策略对比分析

```bash
python compare_rsi_ma.py
```

比较双均线策略和 RSI 策略的效果，生成详细对比报告和可视化图表。

#### 2️⃣ 比较多组均线参数

```python
import strategies

# 定义要比较的均线参数对
ma_pairs = [(10, 60), (10, 120), (90, 120)]

# 比较不同参数的表现
compare_strategies_results = strategies.DoubleMovingAverageStrategy.compare_strategies(
    './data/df_pre_002450.csv', 
    ma_pairs, 
    encoding='gbk'
)

# 打印对比摘要
strategies.DoubleMovingAverageStrategy.print_comparison_summary(compare_strategies_results)

# 绘制策略对比图
strategies.DoubleMovingAverageStrategy.plot_strategies_comparison(
    compare_strategies_results,
    filepath='./data/df_pre_002450.csv'
)
```

#### 3️⃣ 多股票批量对比

```python
import strategies

# 比较多个股票（可以是文件夹路径或文件列表）
compare_stocks_results = strategies.DoubleMovingAverageStrategy.compare_stocks(
    './data_1',  # 文件夹路径或文件列表
    short_ma=10, 
    long_ma=120, 
    encoding='gbk',
    initial_capital=1000000,      # 初始资金 100 万
    commission_rate=0.0001,       # 手续费率万 1
    min_commission=5,             # 最低手续费 5 元
    stamp_tax_rate=0.001,         # 印花税千 1
    risk_free_rate=0.02           # 无风险利率 2%
)

# 打印对比摘要
strategies.DoubleMovingAverageStrategy.print_comparison_summary(compare_stocks_results)

# 绘制股票对比图（显示前 300 只）
strategies.DoubleMovingAverageStrategy.plot_stock_comparison(
    compare_stocks_results,
    top_n=300
)
```

### 自定义策略参数

```python
strategy = strategies.DoubleMovingAverageStrategy(
    short_ma=10,              # 短期均线周期
    long_ma=120,              # 长期均线周期
    initial_capital=100000,   # 初始资金（默认 10 万）
    commission_rate=0.0001,   # 手续费率（默认万 1）
    min_commission=5,         # 最低手续费（默认 5 元）
    stamp_tax_rate=0.001,     # 印花税率（默认千 1）
    risk_free_rate=0.02       # 无风险利率（默认 2%）
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

### 对比结果文件

- `ma_compare_stocks.csv`：双均线策略多股票分析结果
- `rsi_compare_stocks.csv`：RSI 策略多股票分析结果
- 包含指标：股票代码、总收益率、年化收益率、最大回撤、夏普比率、胜率、盈亏比等

## 📈 策略对比示例

### 运行对比分析

```bash
python compare_rsi_ma.py
```

### 输出结果

#### 1. 控制台统计对比

```
================================================================================
双均线策略 vs RSI 策略对比分析
================================================================================

总收益率:
统计指标             双均线策略          RSI 策略              差异
------------------------------------------------------------
平均数                 0.124625        0.462332       -0.337706
最大值                 5.548966        8.900976       -3.352010
最小值                -1.548686       -0.987952       -0.560734
中位数                -0.173916        0.251428       -0.425344
标准差                 0.944691        1.038187       -0.093496

最大回撤:
统计指标             双均线策略          RSI 策略              差异
------------------------------------------------------------
平均数                 0.599978        0.534210        0.065768
...
```

#### 2. 可视化图表

- `compare_ma_rsi/strategy_comparison_scatter.png`：总收益率散点图
- `compare_ma_rsi/strategy_comparison_histograms.png`：三个指标的直方图对比

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

### RSI 策略

RSI（相对强弱指数）策略基于动量理论，衡量价格的涨跌动能。

**计算公式：**
```
RSI = 100 - (100 / (1 + RS))
RS = N 日内上涨幅度平均值 / N 日内下跌幅度平均值
```

**交易规则：**
1. **超卖区域（买入信号）**：RSI < 超卖阈值（如 20）
   - 表明市场可能过度下跌，存在反弹机会
   
2. **超买区域（卖出信号）**：RSI > 超买阈值（如 80）
   - 表明市场可能过度上涨，存在回调风险

**参数说明：**
- `rsi_period`：RSI 计算周期（默认 14 天）
- `oversold_threshold`：超卖阈值（默认 20）
- `overbought_threshold`：超买阈值（默认 80）

## 📝 注意事项

1. **数据质量**：确保数据完整且准确，缺少关键字段会导致错误
2. **参数设置**：
   - 双均线：短期均线周期应小于长期均线周期
   - RSI 策略：合理设置超买超卖阈值
3. **回测局限性**：历史表现不代表未来收益
4. **交易成本**：已考虑手续费和印花税，但未考虑滑点
5. **流动性假设**：假设可以按收盘价成交
6. **数据更新**：定期运行 `getdata_hs300.py` 更新最新数据
7. **回撤率处理**：RSI 策略的回撤率已自动转换为绝对值便于对比

## 🛠️ 扩展开发

### 添加新策略

继承 `DoubleMovingAverageStrategy` 类或创建新策略类，实现以下方法：
- `load_data()`：加载数据
- `preprocess_data()`：数据预处理
- `generate_signals()`：生成交易信号
- `run_backtest()`：运行回测
- `calculate_metrics()`：计算绩效指标
- `plot_results()`：绘制结果图表

## 🛠️ 扩展开发

### 添加新策略

继承基础策略类或创建新策略类，实现以下方法：
- `load_data()`：加载数据
- `preprocess_data()`：数据预处理
- `generate_signals()`：生成交易信号
- `run_backtest()`：运行回测
- `calculate_metrics()`：计算绩效指标
- `plot_results()`：绘制结果图表

### 批量处理示例

```python
from pathlib import Path
import strategies

# 获取文件夹下所有 CSV 文件
file_list = list(Path('./data_dealed').glob("*.csv"))

# 批量分析（双均线策略）
ma_results = strategies.DoubleMovingAverageStrategy.compare_stocks(
    file_list, 
    short_ma=10, 
    long_ma=120
)

# 批量分析（RSI 策略）
rsi_results = strategies.RSIStrategy.compare_stocks(
    file_list,
    rsi_period=14,
    oversold_threshold=20,
    overbought_threshold=80
)

# 导出结果
ma_results.to_csv('ma_compare_stocks.csv', index=False)
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
