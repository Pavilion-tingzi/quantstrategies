# 量化交易策略回测系统

基于 Python 的股票量化交易策略回测系统，实现了双均线策略的完整回测框架，支持多股票对比分析和策略参数优化。

## 📁 项目结构

```
quant/
├── run.py              # 主程序入口，演示各种使用场景
├── strategies.py       # 核心策略实现（双均线策略）
├── getdata.py         # 数据获取脚本（使用 baostock API）
├── data/              # 股票数据存储目录
└── compare_stocks_results.csv  # 多股票对比结果输出
```

## ✨ 主要功能

### 1. **双均线策略回测**
   - 短期均线上穿长期均线 → 买入信号
   - 短期均线下穿长期均线 → 卖出信号
   - 考虑涨跌停限制：涨停不能买入，跌停不能卖出
   - 交易延迟到下一个可交易日执行

### 2. **完整的回测框架**
   - ✅ 初始资金、手续费、印花税等参数自定义
   - ✅ 计算持仓、现金、持仓市值
   - ✅ 每日总资产和收益率跟踪
   - ✅ 最大回撤计算
   - ✅ 夏普比率计算（考虑无风险利率）

### 3. **绩效评估指标**
   - **总收益率**：策略的累计收益率
   - **年化收益率**：按年化计算的收益率
   - **最大回撤**：历史最大亏损幅度
   - **夏普比率**：风险调整后收益指标
   - **交易次数**：总交易次数
   - **胜率**：盈利交易占比
   - **盈亏比**：平均盈利与平均亏损之比

### 4. **多维度对比分析**
   - 📊 **单只股票分析**：完整回测 + 图表展示
   - 📊 **多参数对比**：比较不同均线参数组合的表现
   - 📊 **多股票对比**：同一策略在不同股票上的表现
   - 📊 **策略 vs 买入持有**：超额收益分析

### 5. **可视化图表**
   - 📈 价格走势与均线图（标注买卖点）
   - 📈 持仓状态图
   - 📈 策略净值 vs 买入持有净值（对数坐标）
   - 📈 回撤曲线图

## 🚀 快速开始

### 环境要求

```bash
pip install numpy pandas matplotlib baostock pathlib
```

### 基本使用

#### 1️⃣ 单只股票回测

```python
import strategies

# 创建策略实例（参数：短期均线=10，长期均线=120）
strategy = strategies.DoubleMovingAverageStrategy(short_ma=10, long_ma=120)

# 运行完整分析
strategy.run_complete_analysis('./data/df_pre_002385.csv', encoding='gbk')

# 打印结果
strategy.print_results()

# 绘制图表
strategy.plot_results()
```

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

## 📊 数据格式要求

数据文件应为 CSV 格式，包含以下必要字段：
- `日期`：交易日期
- `收盘`：收盘价
- `开盘`：开盘价

示例文件名：`df_pre_002385.csv`（文件名中包含股票代码）

### 数据获取

可以使用 `getdata.py` 脚本从 baostock 获取沪深 300 成分股数据：

```python
# 运行 getdata.py
python getdata.py
```

## 📈 输出示例

### 回测结果摘要

```
======================================================================
股票 002385 - 双均线策略回测结果 (MA10 vs MA120)
======================================================================
指标                      双均线策略               买入持有
----------------------------------------------------------------------
总收益率                        125.34%                98.76%
年化收益率                         18.45%                15.23%
最大回撤                          -25.67%               -35.89%
夏普比率                           1.23                  0.87
交易次数                             45                   1 次买入
胜率                            62.22%                 100%
======================================================================

超额收益（相对买入持有）: 26.58%
```

### 多股票对比结果

结果将导出到 `compare_stocks_results.csv`，包含：
- 股票代码、数据条数、时间范围
- 策略指标（总收益率、年化收益率、最大回撤、夏普比率等）
- 买入持有指标
- 超额收益、胜率优势、回撤改善等对比指标

## 🔧 策略逻辑详解

### 交易信号生成
1. **金叉（买入信号）**：短期均线上穿长期均线
2. **死叉（卖出信号）**：短期均线下穿长期均线

### 涨跌停处理
- 涨停时（涨幅≥9.9%）：不能买入，信号延迟到下一交易日
- 跌停时（跌幅≤-9.9%）：不能卖出，信号延迟到下一交易日

### 交易规则
- 买入时全仓操作（扣除手续费后最大化买入）
- 卖出时全部卖出
- 买入份额为 100 的整数倍（整手交易）
- 手续费 = max(交易金额 × 费率，最低手续费)
- 印花税 = 交易金额 × 税率（仅卖出时收取）

## 📝 注意事项

1. **数据质量**：确保数据完整且准确，缺少关键字段会导致错误
2. **参数设置**：短期均线周期应小于长期均线周期
3. **回测局限性**：历史表现不代表未来收益
4. **交易成本**：已考虑手续费和印花税，但未考虑滑点
5. **流动性假设**：假设可以按收盘价成交

## 🛠️ 扩展开发

### 添加新策略

继承 `DoubleMovingAverageStrategy` 类或创建新策略类，实现以下方法：
- `load_data()`：加载数据
- `preprocess_data()`：数据预处理
- `generate_signals()`：生成交易信号
- `run_backtest()`：运行回测
- `calculate_metrics()`：计算绩效指标
- `plot_results()`：绘制结果图表

### 批量处理

```python
from pathlib import Path

# 获取文件夹下所有 CSV 文件
file_list = list(Path('./data').glob("*.csv"))

# 批量分析
results = strategies.DoubleMovingAverageStrategy.compare_stocks(
    file_list, 
    short_ma=10, 
    long_ma=120
)
```

## 📄 许可证

本项目仅供学习和研究使用。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请提交 Issue。

---

**免责声明**：本系统仅用于量化交易学习和研究，不构成任何投资建议。股市有风险，投资需谨慎。
