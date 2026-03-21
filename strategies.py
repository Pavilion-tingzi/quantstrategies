import re
import numpy as np
import pandas as pd
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union, Any
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib
import warnings
from tabulate import tabulate
import chardet

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class BuyHoldStrategy:
    """买入持有策略类，用于计算买入持有策略的各项绩效指标"""

    def __init__(self, initial_capital: float, commission_rate: float,
                 stamp_duty_rate: float, min_commission: float):
        """
        初始化买入持有策略

        Parameters:
        -----------
        initial_capital : float
            初始资金
        commission_rate : float
            佣金费率
        stamp_duty_rate : float
            印花税费率
        min_commission : float
            最低佣金
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_commission = min_commission
        self.data = None
        self.net_values = None
        self.metrics = {}

    def run_backtest(self, data: pd.DataFrame) -> pd.Series:
        """
        运行买入持有策略回测

        Parameters:
        -----------
        data : pd.DataFrame
            包含日期和收盘价的数据

        Returns:
        --------
        pd.Series : 每日净值序列
        """
        self.data = data.copy()
        first_close = float(self.data.iloc[0]['收盘'])

        # 计算可买入的最大股数
        max_shares = int(self.initial_capital / (first_close * 100)) * 100
        shares = max_shares

        if shares > 0:
            buy_value = first_close * shares
            commission = max(buy_value * self.commission_rate, self.min_commission)
            cash_after_buy = self.initial_capital - buy_value - commission
        else:
            cash_after_buy = self.initial_capital
            shares = 0

        # 计算每日净值
        net_values = []
        for i, row in self.data.iterrows():
            if i == 0:
                net_values.append(self.initial_capital)
            else:
                total_value = cash_after_buy + shares * float(row['收盘'])
                net_values.append(total_value)

        self.net_values = pd.Series(net_values, index=pd.to_datetime(self.data['日期']))
        return self.net_values

    def calculate_metrics(self, risk_free_rate: float = 0.02) -> Dict:
        """
        计算买入持有策略的绩效指标

        Parameters:
        -----------
        risk_free_rate : float
            无风险利率

        Returns:
        --------
        Dict : 绩效指标字典
        """
        if self.net_values is None:
            raise ValueError("请先运行回测")

        # 总收益率
        total_return = (self.net_values.iloc[-1] / self.net_values.iloc[0]) - 1

        # 年化收益率
        days = (self.net_values.index[-1] - self.net_values.index[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (252 / days) - 1
        else:
            annual_return = 0

        # 最大回撤
        rolling_max = self.net_values.expanding().max()
        drawdown = (self.net_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 夏普比率
        daily_returns = self.net_values.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess = daily_returns - risk_free_rate / 252
            sharpe = excess.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0

        self.metrics = {
            '总收益率': total_return,
            '年化收益率': annual_return,
            '最大回撤': max_drawdown,
            '夏普比率': sharpe,
            '交易次数': 0,
            '胜率': 0,
            '盈亏比': 0
        }

        return self.metrics

class DoubleMovingAverageStrategy:
    """
    双均线策略回测类

    参数可自定义：
    - short_ma: 短期均线周期
    - long_ma: 长期均线周期
    - initial_capital: 初始资金
    - commission_rate: 手续费率
    - min_commission: 最低手续费
    - stamp_tax_rate: 印花税率 (仅卖出)
    - stop_loss: 止损线
    - risk_free_rate: 无风险利率
    """

    def __init__(self,
                 short_ma: int = 10,
                 long_ma: int = 120,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.0001,
                 min_commission: float = 5,
                 stamp_tax_rate: float = 0.001,
                 stop_loss: float = None,
                 risk_free_rate: float = 0.02):
        """
        初始化策略参数

        Parameters:
        -----------
        short_ma : int
            短期均线周期 (默认: 10)
        long_ma : int
            长期均线周期 (默认: 120)
        initial_capital : float
            初始资金 (默认: 100000)
        commission_rate : float
            手续费率 (默认: 0.0001, 万1)
        min_commission : float
            最低手续费 (默认: 5)
        stamp_tax_rate : float
            印花税率 (默认: 0.001, 千1, 仅卖出)
        stop_loss : float, optional
            止损线，如0.1表示-10%止损 (默认: None)
        risk_free_rate : float
            无风险利率 (默认: 0.02, 用于夏普比率计算)
        """
        # 策略参数
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.stop_loss = stop_loss
        self.risk_free_rate = risk_free_rate

        # 验证参数有效性
        if short_ma >= long_ma:
            warnings.warn("警告：短期均线周期应小于长期均线周期，否则可能导致信号异常")

        # 数据相关
        self.stock_code = None
        self.stock_name = None
        self.data = None
        self.results = None

        # 绩效指标
        self.metrics = {}
        self.buyhold_metrics = {}

        # 均线列名
        self.ma_short_col = f'ma_{self.short_ma}'
        self.ma_long_col = f'ma_{self.long_ma}'

    def _detect_encoding(self, filepath: str) -> str:
        """检测文件编码"""
        encodings = ['gbk', 'utf-8', 'gb2312', 'gb18030']
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    f.readline()
                return encoding
            except:
                continue
        return 'gbk'  # 默认返回gbk

    def load_data(self, filepath: str) -> 'DoubleMovingAverageStrategy':
        """
        加载数据

        Parameters:
        -----------
        filepath : str
            数据文件路径

        Returns:
        --------
        self : DoubleMovingAverageStrategy
            策略实例
        """
        # 检测文件编码
        encoding = self._detect_encoding(filepath)

        # 加载数据
        self.data = pd.read_csv(filepath, encoding=encoding)

        # 提取股票代码和名称
        filename = os.path.basename(filepath)

        # 尝试从文件名提取股票代码（6位数字）
        match = re.search(r'(\d{6})', filename)
        if match:
            self.stock_code = match.group(1)
        else:
            self.stock_code = filename.split('.')[0]

        # 尝试提取股票名称
        name_match = re.search(r'[^a-zA-Z0-9](\w{2,4})[^a-zA-Z0-9]', filename)
        self.stock_name = name_match.group(1) if name_match else self.stock_code

        # 确保日期列存在并处理
        if '日期' not in self.data.columns:
            # 尝试其他可能的日期列名
            for col in self.data.columns:
                if 'date' in col.lower() or '时间' in col or '交易日期' in col:
                    self.data.rename(columns={col: '日期'}, inplace=True)
                    break
            else:
                raise ValueError("数据中缺少'日期'列")

        # 转换日期格式
        try:
            self.data['日期'] = pd.to_datetime(self.data['日期'])
        except:
            try:
                self.data['日期'] = pd.to_datetime(self.data['日期'], format='%Y%m%d')
            except:
                pass

        # 按日期排序
        self.data = self.data.sort_values('日期').reset_index(drop=True)

        print(f"股票 {self.stock_code} {self.stock_name} 数据加载完成，共{len(self.data)}条记录")
        return self

    def preprocess_data(self) -> 'DoubleMovingAverageStrategy':
        """
        数据处理：计算策略指标和交易信号字段

        Returns:
        --------
        self : DoubleMovingAverageStrategy
            策略实例
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        # 确保必要的列存在
        required_cols = ['开盘', '收盘']
        for col in required_cols:
            if col not in self.data.columns:
                for possible in ['open', 'close', 'Open', 'Close']:
                    if possible in self.data.columns:
                        self.data.rename(columns={possible: col}, inplace=True)
                        break
                else:
                    raise ValueError(f"数据中缺少'{col}'列")

        # 检查异常情况字段
        if '异常情况' not in self.data.columns:
            warnings.warn("数据中缺少'异常情况'字段，将假设所有交易日正常")
            self.data['异常情况'] = ''
        else:
            # 将NaN转换为空字符串
            self.data['异常情况'] = self.data['异常情况'].fillna('')

        # 初始化交易相关字段
        self.data['position'] = 0
        self.data['cash'] = 0.0
        self.data['hold_value'] = 0.0
        self.data['commission'] = 0.0
        self.data['stamp_tax'] = 0.0
        self.data['total_asset'] = 0.0
        self.data['daily_return'] = 0.0
        self.data['signal'] = ''  # 最终执行信号
        self.data['raw_signal'] = ''  # 原始信号（基于均线交叉）

        # 计算均线（shift(1)确保使用前一天的收盘价计算）
        self.data[self.ma_short_col] = self.data['收盘'].rolling(window=self.short_ma).mean().shift(1)
        self.data[self.ma_long_col] = self.data['收盘'].rolling(window=self.long_ma).mean().shift(1)

        # 识别金叉和死叉
        self.data['golden_cross'] = (self.data[self.ma_short_col] > self.data[self.ma_long_col]) & \
                                    (self.data[self.ma_short_col].shift(1) <= self.data[self.ma_long_col].shift(1))
        self.data['death_cross'] = (self.data[self.ma_short_col] < self.data[self.ma_long_col]) & \
                                   (self.data[self.ma_short_col].shift(1) >= self.data[self.ma_long_col].shift(1))

        # 生成原始信号
        for i in range(len(self.data)):
            if self.data.loc[i, 'golden_cross']:
                self.data.loc[i, 'raw_signal'] = 'buy'
            elif self.data.loc[i, 'death_cross']:
                self.data.loc[i, 'raw_signal'] = 'sell'

        # 过滤异常情况（只过滤掉当天有异常的信号）
        abnormal_mask = self.data['异常情况'] != ''
        conflict_mask = abnormal_mask & (self.data['raw_signal'] != '')
        self.data.loc[conflict_mask, 'raw_signal'] = ''

        # 生成最终信号（考虑首笔买入、交替、间隔）
        self._generate_final_signals()

        return self

    def _generate_final_signals(self) -> None:
        """
        生成最终交易信号，满足以下条件：
        1. 首笔交易必须是买入
        2. 买卖信号必须交替出现
        3. 相同方向信号之间至少间隔5天
        """
        # 设置最小信号间隔
        min_interval = 5

        # 获取所有原始信号
        signal_indices = []
        signal_types = []

        for i in range(len(self.data)):
            if self.data.loc[i, 'raw_signal'] in ['buy', 'sell']:
                signal_indices.append(i)
                signal_types.append(self.data.loc[i, 'raw_signal'])

        if not signal_indices:
            return

        # 找到第一个买入信号的位置
        first_buy_idx = None
        for i, signal_type in enumerate(signal_types):
            if signal_type == 'buy':
                first_buy_idx = i
                break

        if first_buy_idx is None:
            return

        # 从第一个买入信号开始筛选
        selected_indices = [signal_indices[first_buy_idx]]
        selected_types = ['buy']
        last_idx = signal_indices[first_buy_idx]
        last_type = 'buy'

        # 遍历后续信号
        for i in range(first_buy_idx + 1, len(signal_indices)):
            current_idx = signal_indices[i]
            current_type = signal_types[i]

            # 检查信号类型是否交替（不能连续相同）
            if current_type == last_type:
                continue

            # 检查时间间隔
            if current_idx - last_idx < min_interval:
                continue

            # 通过所有检查，接受该信号
            selected_indices.append(current_idx)
            selected_types.append(current_type)
            last_idx = current_idx
            last_type = current_type

        # 先清空原有的signal列
        self.data['signal'] = ''

        # 将筛选后的信号写入signal列
        for idx, signal_type in zip(selected_indices, selected_types):
            self.data.loc[idx, 'signal'] = signal_type

    def _check_stop_loss(self, current_price: float, buy_price: float) -> bool:
        """检查是否触发止损"""
        if self.stop_loss is None or buy_price == 0:
            return False
        return (current_price - buy_price) / buy_price <= -self.stop_loss

    def run_backtest(self) -> 'DoubleMovingAverageStrategy':
        """
        运行回测

        Returns:
        --------
        self : DoubleMovingAverageStrategy
            策略实例
        """
        if self.data is None:
            raise ValueError("请先加载和预处理数据")

        cash = self.initial_capital
        position = 0
        buy_price = 0  # 记录买入价格，用于止损
        pending = None  # 延迟交易信号（由于异常情况导致的延迟）
        last_trade_date = None  # 记录上次交易日期

        # 获取异常情况标记
        abnormal_mask = self.data['异常情况'] != ''

        for i in range(len(self.data)):
            row = self.data.iloc[i]
            signal = ''

            # 检查当日是否有异常情况
            is_abnormal = abnormal_mask.iloc[i]

            # 检查止损
            if position > 0 and self.stop_loss is not None:
                if self._check_stop_loss(row['收盘'], buy_price):
                    # 触发止损，强制卖出（需要当天没有异常情况）
                    if not is_abnormal:
                        signal = 'sell'
                        position = 0
                        buy_price = 0
                        pending = None
                        last_trade_date = row['日期']

            # 执行待处理的延迟信号（需要当天没有异常情况）
            if pending is not None and not is_abnormal:
                if pending == 'buy' and signal == '':
                    signal = 'buy'
                    pending = None
                    last_trade_date = row['日期']
                elif pending == 'sell' and signal == '':
                    signal = 'sell'
                    pending = None
                    last_trade_date = row['日期']

            # 触发新信号（使用预处理阶段计算好的signal字段）
            elif signal == '' and not is_abnormal and row['signal'] in ['buy', 'sell']:
                # 检查是否已经持仓/空仓
                if position == 0 and row['signal'] == 'buy':
                    signal = 'buy'
                    last_trade_date = row['日期']

                elif position > 0 and row['signal'] == 'sell':
                    signal = 'sell'
                    last_trade_date = row['日期']

            # 如果有信号但遇到异常情况，延迟执行
            elif signal == '' and is_abnormal and row['signal'] in ['buy', 'sell']:
                if position == 0 and row['signal'] == 'buy':
                    pending = 'buy'
                elif position > 0 and row['signal'] == 'sell':
                    pending = 'sell'

            # 执行交易
            if signal == 'buy':
                price = row['收盘']
                # 计算理论最大可买份额（整手）
                max_shares = int(cash / price / 100) * 100

                # 循环调整：确保剩余资金够付手续费
                while max_shares > 0:
                    trade_amount = max_shares * price
                    commission = max(trade_amount * self.commission_rate, self.min_commission)
                    total_cost = trade_amount + commission

                    if cash >= total_cost:
                        position = max_shares
                        buy_price = price
                        cash -= total_cost
                        self.data.at[i, 'commission'] = commission
                        self.data.at[i, 'stamp_tax'] = 0
                        break
                    else:
                        max_shares -= 100

            elif signal == 'sell':
                if position > 0:
                    price = row['收盘']
                    trade_amount = position * price
                    commission = max(trade_amount * self.commission_rate, self.min_commission)
                    stamp_tax = trade_amount * self.stamp_tax_rate
                    cash += trade_amount - commission - stamp_tax
                    self.data.at[i, 'commission'] = commission
                    self.data.at[i, 'stamp_tax'] = stamp_tax
                    position = 0
                    buy_price = 0

            # 记录每日状态
            self.data.at[i, 'position'] = position
            self.data.at[i, 'cash'] = cash
            self.data.at[i, 'hold_value'] = position * row['收盘']

            # 计算每日总资产
            self.data.at[i, 'total_asset'] = cash + position * row['收盘']

        # 计算日收益率
        self.data['daily_return'] = self.data['total_asset'].pct_change()

        # 统计实际执行的交易
        executed_buy = (self.data['signal'] == 'buy').sum()
        executed_sell = (self.data['signal'] == 'sell').sum()
        pending_count = 1 if pending is not None else 0

        print(f"回测执行完成，实际执行买入: {executed_buy}次，卖出: {executed_sell}次，待处理信号: {pending_count}个")

        return self

    def calculate_metrics(self) -> Dict:
        """
        计算绩效指标

        Returns:
        --------
        Dict : 策略绩效指标字典
        """
        if self.data is None:
            raise ValueError("请先运行回测")

        try:
            # 最终资产
            final_asset = self.data['total_asset'].iloc[-1]

            # 基础指标
            trading_days = len(self.data)
            self.metrics['总收益率'] = final_asset / self.initial_capital - 1

            # 年化收益率
            if self.metrics['总收益率'] > -1:
                try:
                    self.metrics['年化收益率'] = (1 + self.metrics['总收益率']) ** (252 / trading_days) - 1
                except (ValueError, ZeroDivisionError):
                    self.metrics['年化收益率'] = 0
            else:
                self.metrics['年化收益率'] = -1

            # 最大回撤
            self.data['cum_max'] = self.data['total_asset'].cummax()
            mask = self.data['cum_max'] > 0
            self.data['drawdown'] = 0.0
            self.data.loc[mask, 'drawdown'] = 1 - self.data.loc[mask, 'total_asset'] / self.data.loc[mask, 'cum_max']
            self.metrics['最大回撤率'] = self.data['drawdown'].max()

            # 夏普比率
            daily_returns = self.data['daily_return'].dropna()
            if len(daily_returns) > 0:
                daily_std = daily_returns.std()
                if daily_std > 0 and not np.isnan(daily_std) and not np.isinf(daily_std):
                    excess_return = daily_returns.mean() * 252 - self.risk_free_rate
                    volatility = daily_std * np.sqrt(252)
                    if volatility > 0 and not np.isnan(volatility) and not np.isinf(volatility):
                        self.metrics['夏普比率'] = excess_return / volatility
                    else:
                        self.metrics['夏普比率'] = 0
                else:
                    self.metrics['夏普比率'] = 0
            else:
                self.metrics['夏普比率'] = 0

            # 交易统计
            self.metrics['交易次数'], self.metrics['胜率'], self.metrics['盈亏比'] = self._calculate_trade_stats()

            # 处理无效值
            for key in self.metrics:
                value = self.metrics[key]
                if pd.isna(value) or np.isinf(value):
                    self.metrics[key] = 0

        except Exception as e:
            print(f"计算策略指标时出错: {e}")
            self.metrics = {
                '总收益率': 0,
                '年化收益率': 0,
                '最大回撤率': 0,
                '夏普比率': 0,
                '交易次数': 0,
                '胜率': 0,
                '盈亏比': 0
            }

        # 计算买入持有策略指标
        self._calculate_buyhold_metrics()

        return self.metrics

    def _calculate_trade_stats(self) -> Tuple[int, float, float]:
        """
        计算交易统计

        Returns:
        --------
        Tuple[int, float, float] : 交易次数, 胜率, 盈亏比
        """
        try:
            trades = self.data[self.data['signal'] != ''][
                ['日期', 'signal', '收盘', 'position', 'commission', 'stamp_tax']].copy()

            if trades.empty:
                return 0, 0, 1

            profits = []
            holding_cost = 0
            holding_shares = 0

            for _, row in trades.iterrows():
                if row['signal'] == 'buy':
                    holding_cost = row['收盘'] * row['position'] + row['commission']
                    holding_shares = row['position']

                elif row['signal'] == 'sell' and holding_shares > 0:
                    sell_revenue = row['收盘'] * holding_shares - row['commission'] - row['stamp_tax']
                    profit = sell_revenue - holding_cost
                    profits.append(profit)
                    holding_cost = 0
                    holding_shares = 0

            if profits:
                win_count = sum(1 for p in profits if p > 0)
                loss_count = sum(1 for p in profits if p < 0)

                total_trades = win_count + loss_count
                win_rate = win_count / total_trades if total_trades > 0 else 0

                total_profit = sum(p for p in profits if p > 0)
                total_loss = sum(-p for p in profits if p < 0)
                profit_loss_ratio = total_profit / total_loss if total_loss > 0 else float('inf')

                # 处理无效值
                if pd.isna(win_rate) or np.isinf(win_rate):
                    win_rate = 0
                if pd.isna(profit_loss_ratio) or np.isinf(profit_loss_ratio):
                    profit_loss_ratio = 0

                return total_trades, win_rate, profit_loss_ratio

            return 0, 0, 1

        except Exception as e:
            print(f"计算交易统计时出错: {e}")
            return 0, 0, 1

    def _calculate_buyhold_metrics(self) -> Dict:
        """计算买入持有策略指标"""
        try:
            # 调用买入持有策略类
            bh_strategy = BuyHoldStrategy(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
                stamp_duty_rate=self.stamp_tax_rate,
                min_commission=self.min_commission
            )

            # 运行回测
            bh_strategy.run_backtest(self.data)

            # 计算指标
            bh_metrics = bh_strategy.calculate_metrics(risk_free_rate=self.risk_free_rate)

            # 获取最大回撤并取绝对值
            max_drawdown = bh_metrics.get('最大回撤', 0)
            if max_drawdown is not None:
                max_drawdown = abs(max_drawdown)

            # 转换指标名称
            self.buyhold_metrics = {
                '总收益率': bh_metrics.get('总收益率', 0),
                '年化收益率': bh_metrics.get('年化收益率', 0),
                '最大回撤率': max_drawdown,
                '夏普比率': bh_metrics.get('夏普比率', 0)
            }

        except Exception as e:
            print(f"计算买入持有指标时出错: {e}")
            self.buyhold_metrics = {
                '总收益率': 0,
                '年化收益率': 0,
                '最大回撤率': 0,
                '夏普比率': 0
            }

        return self.buyhold_metrics

    def plot_result(self, output_path: str = None) -> plt.Figure:
        """
        绘制回测结果图表

        Parameters:
        -----------
        output_path : str, optional
            图表保存路径，如果不提供则只显示不保存

        Returns:
        --------
        plt.Figure : matplotlib图表对象
        """
        if self.data is None:
            raise ValueError("请先运行回测")

        # 创建4个子图
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

        # 子图1：价格 + 均线 + 买卖点
        ax1 = axes[0]
        ax1.plot(self.data['日期'], self.data['收盘'], label='收盘价', color='black', linewidth=1)
        ax1.plot(self.data['日期'], self.data[self.ma_short_col],
                 label=f'MA{self.short_ma}', color='orange', alpha=0.7)
        ax1.plot(self.data['日期'], self.data[self.ma_long_col],
                 label=f'MA{self.long_ma}', color='blue', alpha=0.7)

        # 标记买卖点
        buy_points = self.data[self.data['signal'] == 'buy']
        sell_points = self.data[self.data['signal'] == 'sell']
        ax1.scatter(buy_points['日期'], buy_points['收盘'],
                    marker='^', color='red', s=100, label='买入', zorder=5)
        ax1.scatter(sell_points['日期'], sell_points['收盘'],
                    marker='v', color='green', s=100, label='卖出', zorder=5)

        ax1.set_title(f'价格走势与交易信号 (MA{self.short_ma} vs MA{self.long_ma})', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.set_ylabel('价格')
        ax1.grid(True, alpha=0.3)

        # 子图2：持仓状态
        ax2 = axes[1]
        ax2.fill_between(self.data['日期'], 0, self.data['position'] > 0,
                         alpha=0.3, color='blue', label='持仓')
        ax2.set_title('持仓状态', fontsize=12)
        ax2.set_ylabel('持仓(1=有,0=无)')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)

        # 子图3：策略净值 vs 买入持有净值（对数坐标）
        ax3 = axes[2]

        # 归一化到1起点
        strategy_nav = self.data['total_asset'] / self.initial_capital

        # 计算买入持有净值
        bh_strategy = BuyHoldStrategy(
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            stamp_duty_rate=self.stamp_tax_rate,
            min_commission=self.min_commission
        )
        bh_net_values = bh_strategy.run_backtest(self.data)
        buyhold_nav = bh_net_values / self.initial_capital

        ax3.semilogy(self.data['日期'], strategy_nav, label='策略净值', color='red', linewidth=1.5)
        ax3.semilogy(self.data['日期'], buyhold_nav, label='买入持有净值', color='gray', linewidth=1.5, alpha=0.7)
        ax3.set_title('策略净值 vs 买入持有净值（对数坐标）', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.set_ylabel('净值')
        ax3.grid(True, alpha=0.3)

        # 子图4：回撤曲线
        ax4 = axes[3]
        ax4.fill_between(self.data['日期'], 0, self.data['drawdown'],
                         alpha=0.5, color='red', label='回撤')
        ax4.axhline(y=self.metrics['最大回撤率'], color='darkred', linestyle='--',
                    label=f"最大回撤 {self.metrics['最大回撤率']:.2%}")
        ax4.set_title('回撤曲线', fontsize=12)
        ax4.set_ylabel('回撤幅度')
        ax4.set_xlabel('日期')
        ax4.legend(loc='lower left')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {output_path}")

        plt.show()
        return fig

    def write_report(self, output_path: str = None) -> str:
        """
        编写回测报告

        Parameters:
        -----------
        output_path : str, optional
            报告保存路径，如果不提供则自动生成

        Returns:
        --------
        str : 生成的文件路径
        """
        if self.data is None or not self.metrics:
            raise ValueError("请先运行回测并计算指标")

        # 如果没有提供输出路径，自动生成
        if output_path is None:
            # 从股票代码生成报告文件名
            report_filename = f"{self.stock_code}_双均线_回测结果.xlsx"
            output_path = os.path.join(os.getcwd(), report_filename)
        else:
            # 如果提供了目录路径，在目录下生成文件
            if os.path.isdir(output_path):
                report_filename = f"{self.stock_code}_双均线_回测结果.xlsx"
                output_path = os.path.join(output_path, report_filename)
            # 如果提供了完整文件路径，检查文件名格式
            else:
                # 确保文件名包含股票代码
                dir_name = os.path.dirname(output_path)
                file_name = os.path.basename(output_path)
                if self.stock_code not in file_name:
                    # 如果文件名中不包含股票代码，重新生成
                    new_file_name = f"{self.stock_code}_双均线_回测结果.xlsx"
                    output_path = os.path.join(dir_name if dir_name else '.', new_file_name)

        print(f"\n生成回测报告: {output_path}")

        # 创建Excel写入器
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

            # ========== Sheet1: 参数说明 ==========
            params_data = {
                '参数名称': [
                    '股票代码',
                    '短期均线天数',
                    '长期均线天数',
                    '初始资金',
                    '手续费率',
                    '最低手续费',
                    '印花税率',
                    '止损线',
                    '无风险利率',
                    '最小信号间隔',
                    '数据起始日期',
                    '数据结束日期',
                    '交易天数'
                ],
                '参数值': [
                    self.stock_code,
                    self.short_ma,
                    self.long_ma,
                    f"{self.initial_capital:,.2f}",
                    f"{self.commission_rate:.4%}",
                    f"{self.min_commission:.2f}",
                    f"{self.stamp_tax_rate:.4%}",
                    f"{self.stop_loss:.2%}" if self.stop_loss else '无',
                    f"{self.risk_free_rate:.2%}",
                    '5天',
                    self.data['日期'].iloc[0].strftime('%Y-%m-%d'),
                    self.data['日期'].iloc[-1].strftime('%Y-%m-%d'),
                    len(self.data)
                ]
            }
            params_df = pd.DataFrame(params_data)
            params_df.to_excel(writer, sheet_name='参数说明', index=False)

            # ========== Sheet2: 绩效指标 ==========
            metrics_data = {
                '指标名称': ['总收益率', '年化收益率', '最大回撤', '夏普比率', '交易次数', '胜率', '盈亏比'],
                '本策略指标值': [
                    f"{self.metrics.get('总收益率', 0):.2%}",
                    f"{self.metrics.get('年化收益率', 0):.2%}",
                    f"{self.metrics.get('最大回撤率', 0):.2%}",
                    f"{self.metrics.get('夏普比率', 0):.2f}",
                    str(self.metrics.get('交易次数', 0)),
                    f"{self.metrics.get('胜率', 0):.2%}",
                    f"{self.metrics.get('盈亏比', 0):.2f}"
                ],
                '买入持有指标值': [
                    f"{self.buyhold_metrics.get('总收益率', 0):.2%}",
                    f"{self.buyhold_metrics.get('年化收益率', 0):.2%}",
                    f"{self.buyhold_metrics.get('最大回撤率', 0):.2%}",
                    f"{self.buyhold_metrics.get('夏普比率', 0):.2f}",
                    '1',
                    '100%',
                    '-'
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='绩效指标', index=False)

            # ========== Sheet3: 日度数据 ==========
            # 计算累计收益率
            self.data['累计收益率'] = (self.data['total_asset'] / self.initial_capital - 1)

            # 选择需要的列
            daily_cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量',
                          self.ma_short_col, self.ma_long_col, 'signal',
                          'total_asset', 'hold_value', 'cash', 'position',
                          'drawdown', 'daily_return', '累计收益率']

            # 只保留存在的列
            existing_cols = [col for col in daily_cols if col in self.data.columns]
            daily_df = self.data[existing_cols].copy()

            # 重命名列名使其更友好
            column_mapping = {
                self.ma_short_col: f'MA{self.short_ma}',
                self.ma_long_col: f'MA{self.long_ma}',
                'signal': '交易信号',
                'total_asset': '总资产',
                'hold_value': '持仓市值',
                'cash': '可用资金',
                'position': '持仓股数',
                'drawdown': '回撤率',
                'daily_return': '日收益率',
                '累计收益率': '累计收益率'
            }
            daily_df.rename(columns=column_mapping, inplace=True)

            # 格式化百分比列
            for col in ['回撤率', '日收益率', '累计收益率']:
                if col in daily_df.columns:
                    daily_df[col] = daily_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else '')

            daily_df.to_excel(writer, sheet_name='日度数据', index=False)

            # ========== Sheet4: 交易记录 ==========
            # 获取所有交易信号
            trades = self.data[self.data['signal'] != ''].copy()

            if len(trades) >= 2:
                trade_records = []

                i = 0
                while i < len(trades) - 1:
                    # 找到买入信号
                    if trades.iloc[i]['signal'] == 'buy':
                        buy_row = trades.iloc[i]

                        # 向后寻找卖出信号
                        for j in range(i + 1, len(trades)):
                            if trades.iloc[j]['signal'] == 'sell':
                                sell_row = trades.iloc[j]

                                # 计算交易收益
                                buy_price = buy_row['收盘']
                                sell_price = sell_row['收盘']
                                shares = buy_row['position']  # 买入时的持仓股数

                                if shares > 0:
                                    # 计算买入成本
                                    buy_amount = shares * buy_price
                                    buy_commission = buy_row.get('commission', 0)
                                    buy_cost = buy_amount + buy_commission

                                    # 计算卖出收入
                                    sell_amount = shares * sell_price
                                    sell_commission = sell_row.get('commission', 0)
                                    sell_stamp_tax = sell_row.get('stamp_tax', 0)
                                    sell_revenue = sell_amount - sell_commission - sell_stamp_tax

                                    # 计算盈亏
                                    profit = sell_revenue - buy_cost
                                    profit_rate = profit / buy_cost if buy_cost > 0 else 0

                                    # 判断交易类型
                                    if self.stop_loss and profit_rate <= -self.stop_loss:
                                        trade_type = '止损'
                                    else:
                                        trade_type = '正常'

                                    trade_records.append({
                                        '买入日期': buy_row['日期'],
                                        '卖出日期': sell_row['日期'],
                                        '买入价': f"{buy_price:.4f}",
                                        '卖出价': f"{sell_price:.4f}",
                                        '股数': shares,
                                        '收益率': f"{profit_rate:.2%}",
                                        '盈亏金额': f"{profit:.2f}",
                                        '交易类型': trade_type
                                    })

                                i = j + 1
                                break
                        else:
                            # 没找到卖出信号，跳过当前买入
                            i += 1
                    else:
                        i += 1

                # 创建交易记录DataFrame
                if trade_records:
                    trades_df = pd.DataFrame(trade_records)
                else:
                    trades_df = pd.DataFrame(columns=['买入日期', '卖出日期', '买入价', '卖出价',
                                                      '股数', '收益率', '盈亏金额', '交易类型'])
            else:
                trades_df = pd.DataFrame(columns=['买入日期', '卖出日期', '买入价', '卖出价',
                                                  '股数', '收益率', '盈亏金额', '交易类型'])

            trades_df.to_excel(writer, sheet_name='交易记录', index=False)

            # 调整列宽
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 30)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"回测报告已保存至: {output_path}")
        return output_path

    def run_complete_analysis(self,
                              filepath: str,
                              output_folder: str = './results') -> 'DoubleMovingAverageStrategy':
        """
        运行完整的分析流程

        Parameters:
        -----------
        filepath : str
            数据文件路径
        output_folder : str
            输出文件夹路径

        Returns:
        --------
        self : DoubleMovingAverageStrategy
            策略实例
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 运行分析流程
        self.load_data(filepath)
        self.preprocess_data()
        self.run_backtest()
        self.calculate_metrics()

        # 生成报告和图表
        base_name = f"{self.stock_code}_{self.stock_name}_MA{self.short_ma}_{self.long_ma}"

        # 保存图表
        chart_path = os.path.join(output_folder, f"{base_name}_回测图表.png")
        self.plot_result(chart_path)

        # 保存报告
        report_path = os.path.join(output_folder, f"{base_name}_回测报告.xlsx")
        self.write_report(report_path)

        return self

    @staticmethod
    def compare_stocks(file_list: Union[str, List[str]],
                       output_folder: str = './stock_comparison',
                       short_ma: int = 10,
                       long_ma: int = 120,
                       initial_capital: float = 100000,
                       commission_rate: float = 0.0001,
                       min_commission: float = 5,
                       stamp_tax_rate: float = 0.001,
                       stop_loss: float = None,
                       risk_free_rate: float = 0.02) -> pd.DataFrame:
        """
        多股票对比

        Parameters:
        -----------
        file_list : Union[str, List[str]]
            文件夹路径或文件路径列表
        output_folder : str
            输出文件夹路径
        short_ma : int
            短期均线周期
        long_ma : int
            长期均线周期
        initial_capital : float
            初始资金
        commission_rate : float
            手续费率
        min_commission : float
            最低手续费
        stamp_tax_rate : float
            印花税率
        stop_loss : float, optional
            止损线
        risk_free_rate : float
            无风险利率

        Returns:
        --------
        pd.DataFrame : 多股票对比明细表
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 获取文件列表
        if isinstance(file_list, str) and os.path.isdir(file_list):
            file_list = list(Path(file_list).glob("*.csv"))

        results = []

        for filepath in file_list:
            try:
                print(f"\n{'=' * 50}")
                print(f"正在分析: {filepath}")
                print('=' * 50)

                # 创建策略实例
                strategy = DoubleMovingAverageStrategy(
                    short_ma=short_ma,
                    long_ma=long_ma,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    min_commission=min_commission,
                    stamp_tax_rate=stamp_tax_rate,
                    stop_loss=stop_loss,
                    risk_free_rate=risk_free_rate
                )

                # 运行分析
                strategy.load_data(str(filepath))
                strategy.preprocess_data()
                strategy.run_backtest()
                strategy.calculate_metrics()

                # 收集结果
                results.append({
                    '股票代码': strategy.stock_code,
                    '股票名称': strategy.stock_name,
                    '策略总收益率': strategy.metrics.get('总收益率', 0),
                    '策略年化收益率': strategy.metrics.get('年化收益率', 0),
                    '策略最大回撤率': strategy.metrics.get('最大回撤率', 0),
                    '策略夏普比率': strategy.metrics.get('夏普比率', 0),
                    '策略胜率': strategy.metrics.get('胜率', 0),
                    '策略盈亏比': strategy.metrics.get('盈亏比', 0),
                    '策略交易次数': strategy.metrics.get('交易次数', 0),
                    '买入持有总收益率': strategy.buyhold_metrics.get('总收益率', 0),
                    '买入持有最大回撤率': strategy.buyhold_metrics.get('最大回撤率', 0),
                    '买入持有夏普比率': strategy.buyhold_metrics.get('夏普比率', 0),
                    '超额总收益率': strategy.metrics.get('总收益率', 0) - strategy.buyhold_metrics.get('总收益率', 0),
                    '回撤改善': strategy.buyhold_metrics.get('最大回撤率', 0) - strategy.metrics.get('最大回撤率', 0)
                })

            except Exception as e:
                print(f"分析 {filepath} 时出错: {e}")
                continue

        # 创建结果DataFrame
        results_df = pd.DataFrame(results)

        if results_df.empty:
            print("没有有效的分析结果")
            return results_df

        # 保存明细表
        excel_path = os.path.join(output_folder, '双均线_多股票对比明细表.xlsx')
        results_df.to_excel(excel_path, index=False)
        print(f"对比明细表已保存至: {excel_path}")

        # 绘制对比图
        plot_path = os.path.join(output_folder, '双均线_多股票对比图.png')
        DoubleMovingAverageStrategy._plot_stock_comparison(results_df, plot_path)

        # 打印总体效果
        DoubleMovingAverageStrategy._print_stock_comparison_summary(results_df)

        return results_df

    @staticmethod
    def _plot_stock_comparison(results_df: pd.DataFrame, output_path: str) -> plt.Figure:
        """绘制多股票对比图"""
        if results_df.empty:
            print("没有可绘制的数据")
            return None

        # 处理无效值
        plot_df = results_df.copy()
        for col in plot_df.columns:
            if col not in ['股票代码', '股票名称']:
                plot_df[col] = plot_df[col].apply(lambda x: x if pd.notna(x) and not np.isinf(x) else 0)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 图1: 最大回撤分布对比直方图
        ax1 = axes[0, 0]

        # 过滤异常值
        mask = (plot_df['策略最大回撤率'] <= 1) & (plot_df['买入持有最大回撤率'] <= 1)
        plot_df_1 = plot_df[mask]

        bins = 30
        ax1.hist(plot_df_1['策略最大回撤率'], bins=bins, alpha=0.5,
                 label='策略回撤', color='red', edgecolor='black', density=True)
        ax1.hist(plot_df_1['买入持有最大回撤率'], bins=bins, alpha=0.5,
                 label='买入持有回撤', color='blue', edgecolor='black', density=True)

        # 添加均值线
        ax1.axvline(x=plot_df_1['策略最大回撤率'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f'策略平均: {plot_df_1["策略最大回撤率"].mean():.2%}')
        ax1.axvline(x=plot_df_1['买入持有最大回撤率'].mean(), color='blue', linestyle='--',
                    linewidth=2, label=f'持有平均: {plot_df_1["买入持有最大回撤率"].mean():.2%}')

        ax1.set_xlabel('最大回撤')
        ax1.set_ylabel('股票个数')
        ax1.set_title('最大回撤分布对比直方图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2: 本策略胜率 vs 盈亏比四象限散点图
        ax2 = axes[0, 1]

        # 处理盈亏比无限大的情况
        plot_df['盈亏比_clean'] = plot_df['策略盈亏比'].replace([np.inf, -np.inf], 0)

        scatter = ax2.scatter(plot_df['策略胜率'], plot_df['盈亏比_clean'],
                              c=plot_df['策略夏普比率'], cmap='RdYlGn', s=50, alpha=0.6)

        # 绘制象限分割线
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='盈亏比=1')
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='胜率=50%')

        ax2.set_xlabel('胜率')
        ax2.set_ylabel('盈亏比')
        ax2.set_title('胜率 vs 盈亏比四象限散点图')
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='夏普比率')
        ax2.legend()

        # 图3: 本策略收益率 vs 买入持有策略收益率散点图
        ax3 = axes[1, 0]

        # 设置坐标轴范围
        x_min, x_max = -1, 10
        y_min, y_max = -1, 5

        # 过滤异常值
        mask = (plot_df['买入持有总收益率'] >= x_min) & (plot_df['买入持有总收益率'] <= x_max) & \
               (plot_df['策略总收益率'] >= y_min) & (plot_df['策略总收益率'] <= y_max)
        plot_df_3 = plot_df[mask].copy()

        if not plot_df_3.empty:
            # 绘制45度线
            ax3.plot([x_min, x_max], [x_min, x_max], 'k--', alpha=0.5, label='45度线')

            scatter3 = ax3.scatter(plot_df_3['买入持有总收益率'], plot_df_3['策略总收益率'],
                                   c=plot_df_3['超额总收益率'], cmap='RdYlGn', s=50, alpha=0.6)
            plt.colorbar(scatter3, ax=ax3, label='超额收益')

        ax3.set_xlabel('买入持有收益率')
        ax3.set_ylabel('策略收益率')
        ax3.set_title('策略收益率 vs 买入持有收益率散点图')
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 图4: 本策略超额收益 vs 回撤改善四象限散点图
        ax4 = axes[1, 1]

        # 设置纵轴范围
        y_max_4 = 1.5

        # 过滤异常值
        mask = plot_df['回撤改善'] <= y_max_4
        plot_df_4 = plot_df[mask].copy()

        if not plot_df_4.empty:
            scatter4 = ax4.scatter(plot_df_4['超额总收益率'], plot_df_4['回撤改善'],
                                   c=plot_df_4['策略夏普比率'], cmap='RdYlGn', s=50, alpha=0.6)
            plt.colorbar(scatter4, ax=ax4, label='策略夏普比率')

        # 绘制象限分割线
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        ax4.set_xlabel('超额收益')
        ax4.set_ylabel('回撤改善')
        ax4.set_title('超额收益 vs 回撤改善四象限散点图')
        ax4.grid(True, alpha=0.3)

        plt.suptitle('多股票策略对比分析', fontsize=16, y=1.02)
        plt.tight_layout()

        # 保存图表
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存至: {output_path}")
        plt.show()

        return fig

    @staticmethod
    def _print_stock_comparison_summary(results_df: pd.DataFrame) -> None:
        """打印多股票对比总体效果"""
        if results_df.empty:
            print("没有数据可显示")
            return

        # 过滤有效数据
        valid_df = results_df.copy()

        print("\n" + "=" * 60)
        print("策略总体效果")
        print("=" * 60)
        print(f"分析股票总数: {len(valid_df)}")

        # 跑赢基准比例
        beat_count = (valid_df['策略总收益率'] > valid_df['买入持有总收益率']).sum()
        beat_ratio = beat_count / len(valid_df)
        print(f"跑赢基准比例: {beat_ratio:.1%} ({beat_count}/{len(valid_df)})")

        # 回撤改善比例
        improve_count = (valid_df['策略最大回撤率'] < valid_df['买入持有最大回撤率']).sum()
        improve_ratio = improve_count / len(valid_df)
        print(f"回撤改善比例: {improve_ratio:.1%} ({improve_count}/{len(valid_df)})")

        # 平均回撤
        print(f"策略平均回撤: {valid_df['策略最大回撤率'].mean():.2%} vs "
              f"买入持有平均回撤: {valid_df['买入持有最大回撤率'].mean():.2%}")

        # 平均夏普
        print(f"策略平均夏普: {valid_df['策略夏普比率'].mean():.2f} vs "
              f"买入持有平均夏普: {valid_df['买入持有夏普比率'].mean():.2f}")

        # 其他平均指标
        print(f"\n平均总收益率: {valid_df['策略总收益率'].mean():.2%}")
        print(f"平均交易次数: {valid_df['策略交易次数'].mean():.1f}")
        print(f"平均胜率: {valid_df['策略胜率'].mean():.2%}")
        print(f"平均盈亏比: {valid_df['策略盈亏比'].mean():.2f}")
        print("=" * 60)

    @staticmethod
    def compare_strategies(filepath: str,
                           ma_pairs: List[Tuple[int, int]],
                           output_folder: str = './strategy_comparison',
                           initial_capital: float = 100000,
                           commission_rate: float = 0.0001,
                           min_commission: float = 5,
                           stamp_tax_rate: float = 0.001,
                           stop_loss: float = None,
                           risk_free_rate: float = 0.02) -> pd.DataFrame:
        """
        比较同一只股票的不同参数组合

        Parameters:
        -----------
        filepath : str
            数据文件路径
        ma_pairs : List[Tuple[int, int]]
            均线参数对列表，如 [(5,20), (10,60), (20,120)]
        output_folder : str
            输出文件夹路径
        initial_capital : float
            初始资金
        commission_rate : float
            手续费率
        min_commission : float
            最低手续费
        stamp_tax_rate : float
            印花税率
        stop_loss : float, optional
            止损线
        risk_free_rate : float
            无风险利率

        Returns:
        --------
        pd.DataFrame : 参数对比详细数据
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 获取股票代码
        temp_strategy = DoubleMovingAverageStrategy()
        temp_strategy.load_data(filepath)
        stock_code = temp_strategy.stock_code

        results = []
        all_navs = {}  # 存储所有策略的净值序列，用于绘图

        for short_ma, long_ma in ma_pairs:
            try:
                print(f"\n测试均线组合: MA{short_ma} vs MA{long_ma}")
                print("-" * 40)

                strategy = DoubleMovingAverageStrategy(
                    short_ma=short_ma,
                    long_ma=long_ma,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    min_commission=min_commission,
                    stamp_tax_rate=stamp_tax_rate,
                    stop_loss=stop_loss,
                    risk_free_rate=risk_free_rate
                )

                strategy.run_complete_analysis(filepath, output_folder)

                # 存储净值序列
                strategy_nav = strategy.data['total_asset'] / initial_capital
                all_navs[f'MA{short_ma}-{long_ma}'] = {
                    'nav': strategy_nav,
                    'dates': strategy.data['日期']
                }

                # 收集结果
                results.append({
                    '参数组合': f'MA{short_ma}-{long_ma}',
                    '总收益率': strategy.metrics.get('总收益率', 0),
                    '年化收益率': strategy.metrics.get('年化收益率', 0),
                    '最大回撤率': strategy.metrics.get('最大回撤率', 0),
                    '夏普比率': strategy.metrics.get('夏普比率', 0),
                    '胜率': strategy.metrics.get('胜率', 0),
                    '盈亏比': strategy.metrics.get('盈亏比', 0),
                    '交易次数': strategy.metrics.get('交易次数', 0),
                    '超额收益': strategy.metrics.get('总收益率', 0) - strategy.buyhold_metrics.get('总收益率', 0),
                    '回撤改善': strategy.buyhold_metrics.get('最大回撤率', 0) - strategy.metrics.get('最大回撤率', 0)
                })

            except Exception as e:
                print(f"测试 MA{short_ma}-{long_ma} 失败: {e}")
                continue

        # 创建结果DataFrame
        results_df = pd.DataFrame(results)

        # 添加买入持有策略
        bh_strategy = BuyHoldStrategy(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            stamp_duty_rate=stamp_tax_rate,
            min_commission=min_commission
        )

        # 加载数据并计算买入持有
        data = pd.read_csv(filepath, encoding=temp_strategy._detect_encoding(filepath))
        if '日期' in data.columns:
            data['日期'] = pd.to_datetime(data['日期'])
        data = data.sort_values('日期').reset_index(drop=True)

        bh_net_values = bh_strategy.run_backtest(data)
        bh_metrics = bh_strategy.calculate_metrics(risk_free_rate=risk_free_rate)

        bh_row = {
            '参数组合': '买入持有',
            '总收益率': bh_metrics.get('总收益率', 0),
            '年化收益率': bh_metrics.get('年化收益率', 0),
            '最大回撤率': bh_metrics.get('最大回撤', 0),
            '夏普比率': bh_metrics.get('夏普比率', 0),
            '胜率': 0,
            '盈亏比': 0,
            '交易次数': 1,
            '超额收益': 0,
            '回撤改善': 0
        }

        results_df = pd.concat([results_df, pd.DataFrame([bh_row])], ignore_index=True)

        # 保存详细数据
        excel_path = os.path.join(output_folder, f'{stock_code}_双均线参数对比详细数据.xlsx')
        results_df.to_excel(excel_path, index=False)
        print(f"参数对比详细数据已保存至: {excel_path}")

        # 绘制净值曲线
        plot_path = os.path.join(output_folder, f'{stock_code}_双均线参数对比净值曲线.png')
        DoubleMovingAverageStrategy._plot_strategies_comparison(
            all_navs, bh_net_values / initial_capital, data['日期'], plot_path
        )

        return results_df

    @staticmethod
    def _plot_strategies_comparison(all_navs: Dict, bh_nav: pd.Series,
                                    dates: pd.Series, output_path: str) -> plt.Figure:
        """绘制多策略净值对比图"""
        fig, ax = plt.subplots(figsize=(14, 8))

        # 绘制买入持有净值
        ax.semilogy(dates, bh_nav, label='买入持有', color='black',
                    linewidth=2, linestyle='--', alpha=0.7)

        # 绘制各策略净值
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_navs)))

        for (label, data), color in zip(all_navs.items(), colors):
            ax.semilogy(data['dates'], data['nav'], label=label,
                        color=color, linewidth=1.5, alpha=0.8)

        ax.set_xlabel('日期')
        ax.set_ylabel('净值（对数坐标）')
        ax.set_title('不同参数双均线策略净值对比')
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"净值曲线已保存至: {output_path}")
        plt.show()

        return fig

class RSIStrategy:
    """RSI策略主类"""

    def __init__(self,
                 initial_capital: float = 1000000,
                 commission_rate: float = 0.0001,
                 stamp_tax_rate: float = 0.001,
                 min_commission: float = 5,
                 stop_loss: float = None,
                 risk_free_rate: float = 0.03,
                 rsi_period: int = 14,
                 oversold_threshold: int = 30,
                 overbought_threshold: int = 70):
        """
        初始化RSI策略

        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率
            stamp_tax_rate: 印花税费率
            min_commission: 最低佣金
            stop_loss: 止损线（百分比，如0.1表示10%止损）
            risk_free_rate: 无风险利率
            rsi_period: RSI指标使用K线数量
            oversold_threshold: 超卖区对应RSI值
            overbought_threshold: 超买区对应RSI值
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission
        self.stop_loss = stop_loss
        self.risk_free_rate = risk_free_rate
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

        # 数据相关
        self.data = None
        self.stock_code = None
        self.stock_name = None

        # 结果相关
        self.metrics = {}
        self.trade_records = []

    def _detect_encoding(self, file_path: str) -> str:
        """检测文件编码"""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
        return result['encoding'] or 'utf-8'

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载数据

        Args:
            file_path: 数据文件路径

        Returns:
            加载的数据框
        """
        # 检测文件编码
        encoding = self._detect_encoding(file_path)

        try:
            self.data = pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            # 如果检测到的编码还是不对，尝试常见编码
            for enc in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']:
                try:
                    self.data = pd.read_csv(file_path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue

        # 从文件名提取股票代码
        file_stem = Path(file_path).stem
        if '_' in file_stem:
            parts = file_stem.split('_')
            if len(parts) >= 3:
                self.stock_code = parts[2]
            else:
                self.stock_code = file_stem
        else:
            self.stock_code = file_stem

        self.stock_name = self.stock_code

        # 打印数据列名，用于调试
        print(f"数据列名: {list(self.data.columns)}")

        # 确保日期格式正确 - 更灵活的列名处理
        date_col = None
        for col in ['日期', 'date', 'Date', 'trade_date']:
            if col in self.data.columns:
                date_col = col
                break

        if date_col:
            if date_col != '日期':
                self.data.rename(columns={date_col: '日期'}, inplace=True)
            self.data['日期'] = pd.to_datetime(self.data['日期'])
        else:
            raise ValueError("数据中找不到日期列，请确保列名为'日期'、'date'、'Date'或'trade_date'")

        # 确保收盘价列存在 - 更灵活的列名处理
        close_col = None
        for col in ['收盘', 'close', 'Close', '收盘价']:
            if col in self.data.columns:
                close_col = col
                break

        if close_col:
            if close_col != '收盘':
                self.data.rename(columns={close_col: '收盘'}, inplace=True)
        else:
            raise ValueError("数据中找不到收盘价列，请确保列名为'收盘'、'close'、'Close'或'收盘价'")

        # 确保数据按日期排序
        self.data = self.data.sort_values('日期').reset_index(drop=True)

        print(f"数据加载成功: {len(self.data)}行, 日期范围: {self.data['日期'].min()} 至 {self.data['日期'].max()}")

        return self.data

    def preprocess_data(self) -> pd.DataFrame:
        """
        数据处理：计算所需的策略指标和交易信号字段

        Returns:
            处理后的数据框
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        # 检查是否有异常情况列，如果没有则创建
        if '异常情况' not in self.data.columns:
            self.data['异常情况'] = '正常'

        # 标记停牌和价格异常
        self.data['是停牌'] = self.data['异常情况'] == '停牌'
        self.data['是价格异常'] = self.data['异常情况'] == '价格异常'

        # 确保数据按日期排序
        self.data = self.data.sort_values('日期').reset_index(drop=True)

        # 计算RSI指标
        self._calculate_rsi()

        # 生成交易信号
        self._generate_signals()

        return self.data

    def _calculate_rsi(self) -> pd.DataFrame:
        """计算RSI指标"""
        # 创建仅包含正常交易日的序列
        if '是停牌' not in self.data.columns:
            self.data['是停牌'] = False
            self.data['是价格异常'] = False

        normal_days = self.data[~self.data['是停牌']].copy()
        normal_days = normal_days.reset_index(drop=True)

        if len(normal_days) == 0:
            raise ValueError("没有正常交易日数据")

        # 计算价格变化
        normal_days['价格变化'] = normal_days['收盘'].diff()
        normal_days['涨幅'] = normal_days['价格变化'].apply(lambda x: x if x > 0 else 0)
        normal_days['跌幅'] = normal_days['价格变化'].apply(lambda x: abs(x) if x < 0 else 0)

        # 计算平均涨幅和跌幅 - 使用expanding窗口确保有足够数据
        if len(normal_days) < self.rsi_period:
            print(f"警告: 数据量({len(normal_days)})小于RSI周期({self.rsi_period})，使用全部数据计算")
            period = len(normal_days)
        else:
            period = self.rsi_period

        normal_days['平均涨幅'] = normal_days['涨幅'].rolling(window=period, min_periods=1).mean()
        normal_days['平均跌幅'] = normal_days['跌幅'].rolling(window=period, min_periods=1).mean()

        # 计算RSI，避免除零错误
        denominator = normal_days['平均涨幅'] + normal_days['平均跌幅']
        normal_days['RSI'] = 100 * normal_days['平均涨幅'] / denominator.replace(0, np.nan)
        normal_days['RSI'] = normal_days['RSI'].fillna(50)  # 当分母为0时，RSI设为50

        # 将RSI值映射回原DataFrame
        self.data['RSI'] = np.nan
        for _, row in normal_days.iterrows():
            self.data.loc[self.data['日期'] == row['日期'], 'RSI'] = row['RSI']

        # 向前填充RSI值
        self.data['RSI'] = self.data['RSI'].ffill()

        print(f"RSI计算完成，范围: {self.data['RSI'].min():.2f} - {self.data['RSI'].max():.2f}")

        return self.data

    def _generate_signals(self) -> pd.DataFrame:
        """生成交易信号"""
        if self.data is None or 'RSI' not in self.data.columns:
            raise ValueError("请先计算RSI指标")

        # 判断RSI的位置状态
        self.data['RSI_上穿'] = (self.data['RSI'] >= self.oversold_threshold) & (
                self.data['RSI'].shift(1) < self.oversold_threshold)
        self.data['RSI_下穿'] = (self.data['RSI'] <= self.overbought_threshold) & (
                self.data['RSI'].shift(1) > self.overbought_threshold)

        # 初始化信号列
        self.data['交易信号'] = ''
        self.data['持仓状态'] = 0

        # 状态变量
        position = 0
        buy_price = 0

        # 统计信号数量
        buy_signals = 0
        sell_signals = 0

        for i in range(len(self.data)):
            # 如果是停牌日，保持原有持仓状态
            if self.data.iloc[i].get('是停牌', False):
                self.data.loc[self.data.index[i], '持仓状态'] = position
                continue

            current_price = self.data.iloc[i]['收盘']

            # 买入信号
            if position == 0 and self.data.iloc[i]['RSI_上穿']:
                self.data.loc[self.data.index[i], '交易信号'] = '买入'
                position = 1
                buy_price = current_price
                buy_signals += 1

            # 卖出信号
            elif position > 0 and self.data.iloc[i]['RSI_下穿']:
                self.data.loc[self.data.index[i], '交易信号'] = '卖出'
                position = 0
                sell_signals += 1

            self.data.loc[self.data.index[i], '持仓状态'] = position

        print(f"信号生成完成 - 买入: {buy_signals}, 卖出: {sell_signals}")

        return self.data

    def run_backtest(self) -> Dict:
        """
        运行回测

        Returns:
            绩效指标字典
        """
        if self.data is None or '交易信号' not in self.data.columns:
            raise ValueError("请先执行数据预处理")

        # 回测变量初始化
        cash = float(self.initial_capital)
        position = 0
        position_value = 0
        buy_price = 0
        buy_date = None
        buy_amount = 0
        buy_commission = 0
        buy_shares = 0

        # 记录每日数据
        dates = []
        closes = []
        signals = []
        positions = []
        strategy_nav = []
        position_values = []
        cash_values = []
        total_assets = []

        # 交易记录相关
        self.trade_records = []
        current_trade = {}

        # 统计交易
        trade_count = 0

        # 确保所有数值都是float类型
        self.data['收盘'] = pd.to_numeric(self.data['收盘'], errors='coerce')

        # 创建持仓状态列（如果不存在）
        if '持仓状态' not in self.data.columns:
            self.data['持仓状态'] = 0

        for i in range(len(self.data)):
            date = self.data.iloc[i]['日期']
            price = float(self.data.iloc[i]['收盘']) if pd.notna(self.data.iloc[i]['收盘']) else 0
            signal = self.data.iloc[i]['交易信号']
            is_stop = self.data.iloc[i].get('是停牌', False)

            # 跳过价格为0或停牌的情况
            if price <= 0 or is_stop:
                total_asset = cash + (position * price if position > 0 and price > 0 else 0)
                dates.append(date)
                closes.append(price)
                signals.append(signal)
                positions.append(position)
                position_values.append(position * price if position > 0 and price > 0 else 0)
                cash_values.append(cash)
                total_assets.append(total_asset)
                strategy_nav.append(total_asset / self.initial_capital)
                continue

            # 执行交易
            if signal == '买入' and position == 0:
                # 买入 - 需要预留佣金，如果资金不够则减少股数
                # 先按全部资金计算最大可买股数
                max_shares_by_cash = int(cash / price)

                if max_shares_by_cash >= 100:
                    # 从最大可能股数开始，逐步减少100股，直到资金足够支付佣金
                    shares_to_try = max_shares_by_cash
                    buy_success = False

                    while shares_to_try >= 100 and not buy_success:
                        trade_value = shares_to_try * price
                        commission = max(trade_value * self.commission_rate, self.min_commission)
                        total_cost = trade_value + commission

                        if total_cost <= cash:
                            # 资金足够，执行买入
                            cash -= total_cost
                            position = shares_to_try
                            buy_price = price
                            buy_date = date
                            buy_amount = trade_value
                            buy_commission = commission
                            buy_shares = shares_to_try
                            trade_count += 1
                            buy_success = True

                            # 记录买入
                            current_trade = {
                                '买入日期': date,
                                '买入价': price,
                                '股数': position,
                                '买入金额': trade_value,
                                '买入佣金': commission,
                                '买入总成本': total_cost
                            }
                            break
                        else:
                            # 资金不足，减少100股继续尝试
                            shares_to_try -= 100

            elif (signal == '卖出' or signal == '止损卖出') and position > 0:
                # 卖出
                trade_value = position * price
                commission = max(trade_value * self.commission_rate, self.min_commission)
                stamp_tax = trade_value * self.stamp_tax_rate
                total_received = trade_value - commission - stamp_tax

                # 计算收益率
                if buy_amount > 0 and current_trade:
                    total_cost = current_trade.get('买入总成本', buy_amount + buy_commission)
                    pnl = total_received - total_cost
                    return_pct = (pnl / total_cost) * 100 if total_cost > 0 else 0

                    # 记录交易
                    current_trade.update({
                        '卖出日期': date,
                        '卖出价': price,
                        '卖出金额': trade_value,
                        '卖出佣金': commission,
                        '印花税': stamp_tax,
                        '卖出净收入': total_received,
                        '盈亏金额': pnl,
                        '收益率': return_pct,
                        '交易类型': '正常' if signal == '卖出' else '止损'
                    })
                    self.trade_records.append(current_trade)
                    current_trade = {}

                cash += total_received
                position = 0
                buy_amount = 0
                buy_commission = 0
                buy_shares = 0

            # 止损检查（在没有交易信号的日子也要检查）
            elif position > 0 and self.stop_loss is not None:
                current_loss_pct = (buy_price - price) / buy_price
                if current_loss_pct >= self.stop_loss:
                    # 触发止损卖出
                    trade_value = position * price
                    commission = max(trade_value * self.commission_rate, self.min_commission)
                    stamp_tax = trade_value * self.stamp_tax_rate
                    total_received = trade_value - commission - stamp_tax

                    # 计算收益率
                    if buy_amount > 0 and current_trade:
                        total_cost = current_trade.get('买入总成本', buy_amount + buy_commission)
                        pnl = total_received - total_cost
                        return_pct = (pnl / total_cost) * 100 if total_cost > 0 else 0

                        # 记录交易
                        current_trade.update({
                            '卖出日期': date,
                            '卖出价': price,
                            '卖出金额': trade_value,
                            '卖出佣金': commission,
                            '印花税': stamp_tax,
                            '卖出净收入': total_received,
                            '盈亏金额': pnl,
                            '收益率': return_pct,
                            '交易类型': '止损'
                        })
                        self.trade_records.append(current_trade)
                        current_trade = {}

                    cash += total_received
                    position = 0
                    buy_amount = 0
                    buy_commission = 0
                    buy_shares = 0

            # 计算当日资产
            position_value = position * price
            total_asset = cash + position_value

            # 记录每日数据
            dates.append(date)
            closes.append(price)
            signals.append(signal)
            positions.append(position)
            position_values.append(position_value)
            cash_values.append(cash)
            total_assets.append(total_asset)
            strategy_nav.append(total_asset / self.initial_capital)

        # 添加到data
        self.data['策略净值'] = strategy_nav
        self.data['持仓市值'] = position_values
        self.data['可用资金'] = cash_values
        self.data['总资产'] = total_assets

        # 计算回撤率
        if len(total_assets) > 0:
            cummax = pd.Series(total_assets).expanding().max()
            self.data['回撤率'] = (pd.Series(total_assets) - cummax) / cummax * 100
        else:
            self.data['回撤率'] = 0

        # 计算日收益率和累计收益率
        self.data['日收益率'] = self.data['策略净值'].pct_change() * 100
        self.data['累计收益率'] = (self.data['策略净值'] - 1) * 100

        # 强制更新持仓状态列
        self.data['持仓状态'] = positions

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        """
        计算绩效指标

        Returns:
            绩效指标字典，包含本策略和买入持有策略的指标
        """
        if self.data is None:
            raise ValueError("请先运行回测")

        # 计算本策略指标
        strategy_metrics = self._calculate_strategy_metrics()

        # 计算买入持有指标
        hold_metrics = self._calculate_hold_metrics()

        # 合并指标
        self.metrics = {
            'strategy': strategy_metrics,
            'hold': hold_metrics
        }

        return self.metrics

    def _calculate_strategy_metrics(self) -> Dict:
        """计算本策略绩效指标"""
        total_assets = self.data['总资产'].values
        initial_capital = self.initial_capital

        # 总收益率
        total_return = ((total_assets[-1] / initial_capital) - 1) * 100

        # 年化收益率
        trading_days = len(self.data[~self.data.get('是停牌', pd.Series([False] * len(self.data)))])
        years = trading_days / 245
        if years > 0:
            annual_return = ((total_assets[-1] / initial_capital) ** (1 / years) - 1) * 100
        else:
            annual_return = 0

        # 最大回撤
        max_drawdown = self.data['回撤率'].min()
        if pd.isna(max_drawdown):
            max_drawdown = 0

        # 夏普比率
        daily_returns = self.data['日收益率'].dropna() / 100
        if len(daily_returns) > 1 and daily_returns.std() != 0:
            excess_returns = daily_returns - self.risk_free_rate / 245
            sharpe_ratio = np.sqrt(245) * excess_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0

        # 交易次数
        total_trades = len(self.trade_records)

        # 胜率和盈亏比
        win_rate = 0
        profit_loss_ratio = 0

        if total_trades > 0:
            trades_df = pd.DataFrame(self.trade_records)
            win_trades = trades_df[trades_df['收益率'] > 0]
            loss_trades = trades_df[trades_df['收益率'] < 0]

            win_rate = (len(win_trades) / total_trades) * 100 if total_trades > 0 else 0

            # 计算平均盈利
            avg_win = win_trades['收益率'].mean() if len(win_trades) > 0 else 0

            if len(loss_trades) > 0:
                # 有亏损交易
                avg_loss = abs(loss_trades['收益率'].mean())
                profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            else:
                # 没有亏损交易，盈亏比设为一个大数或None
                profit_loss_ratio = float('inf') if avg_win > 0 else 0

        # 打印详细指标用于调试
        print(f"策略指标 - 总资产: {total_assets[-1]:.2f}, 总收益率: {total_return:.2f}%, 交易次数: {total_trades}")

        return {
            '总收益率': total_return,
            '年化收益率': annual_return,
            '最大回撤': max_drawdown,
            '夏普比率': sharpe_ratio,
            '交易次数': total_trades,
            '胜率': win_rate,
            '盈亏比': profit_loss_ratio
        }

    def _calculate_hold_metrics(self) -> Dict:
        """计算买入持有策略指标"""
        # 创建买入持有策略实例
        hold_strategy = BuyHoldStrategy(
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            stamp_duty_rate=self.stamp_tax_rate,
            min_commission=self.min_commission
        )

        # 运行回测
        hold_strategy.run_backtest(self.data[['日期', '收盘']].copy())

        # 计算指标
        metrics = hold_strategy.calculate_metrics(risk_free_rate=self.risk_free_rate)

        return {
            '总收益率': metrics['总收益率'] * 100,
            '年化收益率': metrics['年化收益率'] * 100,
            '最大回撤': metrics['最大回撤'] * 100,
            '夏普比率': metrics['夏普比率'],
            '交易次数': metrics['交易次数'],
            '胜率': metrics['胜率'],
            '盈亏比': metrics['盈亏比']
        }

    def plot_result(self, save_path: str = None) -> plt.Figure:
        """
        绘制图表

        Args:
            save_path: 图表存放路径

        Returns:
            matplotlib图像对象
        """
        if self.data is None:
            raise ValueError("请先运行回测")

        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle(f'{self.stock_name} - RSI策略回测结果', fontsize=16, fontweight='bold')

        # 获取信号点
        buy_signals = self.data[self.data['交易信号'] == '买入']
        sell_signals = self.data[self.data['交易信号'] == '卖出']
        stop_signals = self.data[self.data['交易信号'] == '止损卖出']

        # 1. 价格和买卖点
        ax1 = axes[0, 0]
        ax1.plot(self.data['日期'], self.data['收盘'], label='收盘价', color='blue', linewidth=1)

        ax1.scatter(buy_signals['日期'], buy_signals['收盘'], color='red', marker='^', s=80, label='买入', zorder=5)
        ax1.scatter(sell_signals['日期'], sell_signals['收盘'], color='green', marker='v', s=80, label='卖出', zorder=5)
        ax1.scatter(stop_signals['日期'], stop_signals['收盘'], color='orange', marker='v', s=80, label='止损',
                    zorder=5)

        ax1.set_title('价格与交易信号')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. RSI指标（加上买卖点标志）
        ax2 = axes[0, 1]
        ax2.plot(self.data['日期'], self.data['RSI'], label=f'RSI({self.rsi_period})', color='purple', linewidth=1.5)
        ax2.axhline(y=self.overbought_threshold, color='red', linestyle='--', alpha=0.5, label='超买线')
        ax2.axhline(y=self.oversold_threshold, color='green', linestyle='--', alpha=0.5, label='超卖线')
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

        ax2.fill_between(self.data['日期'], self.overbought_threshold, 100, alpha=0.1, color='red')
        ax2.fill_between(self.data['日期'], 0, self.oversold_threshold, alpha=0.1, color='green')

        # 在RSI图上添加买卖点标志
        ax2.scatter(buy_signals['日期'], buy_signals['RSI'], color='red', marker='^', s=80, label='买入', zorder=5)
        ax2.scatter(sell_signals['日期'], sell_signals['RSI'], color='green', marker='v', s=80, label='卖出', zorder=5)
        ax2.scatter(stop_signals['日期'], stop_signals['RSI'], color='orange', marker='v', s=80, label='止损', zorder=5)

        ax2.set_title('RSI指标')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 持仓状态
        ax3 = axes[1, 0]
        ax3.fill_between(self.data['日期'], 0, self.data['持仓状态'] * self.data['持仓状态'].max() / 10,
                         color='blue', alpha=0.5, label='持仓')
        ax3.set_title('持仓状态')
        ax3.set_ylabel('持仓')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 策略净值 vs 买入持有净值
        ax4 = axes[1, 1]

        # 计算买入持有净值
        first_price = self.data.iloc[0]['收盘']
        hold_nav = self.data['收盘'] / first_price

        ax4.plot(self.data['日期'], self.data['策略净值'], label='策略净值', color='red', linewidth=2)
        ax4.plot(self.data['日期'], hold_nav, label='买入持有净值', color='blue', linewidth=2, alpha=0.7)
        ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

        ax4.set_title('策略净值对比')
        ax4.set_ylabel('净值')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 回撤曲线
        ax5 = axes[2, 0]
        ax5.fill_between(self.data['日期'], 0, self.data['回撤率'], color='red', alpha=0.3)
        ax5.plot(self.data['日期'], self.data['回撤率'], color='red', linewidth=1)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax5.set_title('策略回撤')
        ax5.set_ylabel('回撤率 (%)')
        ax5.grid(True, alpha=0.3)

        # 6. 日收益率分布
        ax6 = axes[2, 1]
        daily_returns = self.data['日收益率'].dropna()

        ax6.hist(daily_returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax6.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax6.axvline(x=daily_returns.mean(), color='green', linestyle='--', linewidth=1,
                    label=f'均值: {daily_returns.mean():.3f}%')

        ax6.set_title('日收益率分布')
        ax6.set_xlabel('日收益率 (%)')
        ax6.set_ylabel('频次')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 设置日期格式
        for ax in axes.flat:
            if hasattr(ax, 'xaxis'):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def write_report(self, report_path: str):
        """
        编写excel回测报告

        Args:
            report_path: 报告存放路径
        """
        if self.data is None or not self.metrics:
            raise ValueError("请先运行回测并计算指标")

        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # Sheet1: 参数说明
            params_data = {
                '参数名称': [
                    '初始资金',
                    '佣金费率',
                    '印花税费率',
                    '最低佣金',
                    '止损线',
                    '无风险利率',
                    'RSI周期',
                    '超卖阈值',
                    '超买阈值'
                ],
                '参数值': [
                    f"{self.initial_capital:,.0f}",
                    f"{self.commission_rate * 100:.3f}%",
                    f"{self.stamp_tax_rate * 100:.2f}%",
                    f"{self.min_commission}",
                    f"{self.stop_loss * 100:.1f}%" if self.stop_loss else '无',
                    f"{self.risk_free_rate * 100:.1f}%",
                    f"{self.rsi_period}",
                    f"{self.oversold_threshold}",
                    f"{self.overbought_threshold}"
                ]
            }
            pd.DataFrame(params_data).to_excel(writer, sheet_name='参数说明', index=False)

            # Sheet2: 绩效指标 - 指标名称不加%，数值上加%
            metrics_data = {
                '指标名称': ['总收益率', '年化收益率', '最大回撤', '夏普比率', '交易次数', '胜率', '盈亏比'],
                '本策略指标值': [
                    f"{self.metrics['strategy']['总收益率']:.2f}%",
                    f"{self.metrics['strategy']['年化收益率']:.2f}%",
                    f"{abs(self.metrics['strategy']['最大回撤']):.2f}%",  # 取绝对值
                    f"{self.metrics['strategy']['夏普比率']:.2f}",
                    f"{self.metrics['strategy']['交易次数']}",
                    f"{self.metrics['strategy']['胜率']:.2f}%",
                    f"{self.metrics['strategy']['盈亏比']:.2f}"
                ],
                '买入持有指标值': [
                    f"{self.metrics['hold']['总收益率']:.2f}%",
                    f"{self.metrics['hold']['年化收益率']:.2f}%",
                    f"{abs(self.metrics['hold']['最大回撤']):.2f}%",  # 取绝对值
                    f"{self.metrics['hold']['夏普比率']:.2f}",
                    f"{self.metrics['hold']['交易次数']}",
                    f"{self.metrics['hold']['胜率']:.2f}%",
                    f"{self.metrics['hold']['盈亏比']:.2f}"
                ]
            }
            pd.DataFrame(metrics_data).to_excel(writer, sheet_name='绩效指标', index=False)

            # Sheet3: 日度数据 - 回撤率、日收益率、累计收益率的值都带上%
            daily_columns = ['日期', '收盘', 'RSI', '交易信号', '策略净值', '持仓市值', '可用资金', '总资产', '回撤率',
                             '日收益率', '累计收益率']
            daily_data = self.data[daily_columns].copy()
            daily_data['日期'] = daily_data['日期'].dt.strftime('%Y-%m-%d')

            # 格式化百分比列
            daily_data['回撤率'] = daily_data['回撤率'].abs().apply(lambda x: f"{x:.2f}%")
            daily_data['日收益率'] = daily_data['日收益率'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
            daily_data['累计收益率'] = daily_data['累计收益率'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")

            # 格式化其他数值列
            daily_data['收盘'] = daily_data['收盘'].apply(lambda x: f"{x:.2f}")
            daily_data['RSI'] = daily_data['RSI'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            daily_data['策略净值'] = daily_data['策略净值'].apply(lambda x: f"{x:.4f}")
            daily_data['持仓市值'] = daily_data['持仓市值'].apply(lambda x: f"{x:.2f}")
            daily_data['可用资金'] = daily_data['可用资金'].apply(lambda x: f"{x:.2f}")
            daily_data['总资产'] = daily_data['总资产'].apply(lambda x: f"{x:.2f}")

            daily_data.to_excel(writer, sheet_name='日度数据', index=False)

            # Sheet4: 交易记录
            if self.trade_records:
                trade_data = []
                for trade in self.trade_records:
                    trade_data.append({
                        '买入日期': trade['买入日期'].strftime('%Y-%m-%d') if hasattr(trade['买入日期'],
                                                                                      'strftime') else trade[
                            '买入日期'],
                        '卖出日期': trade['卖出日期'].strftime('%Y-%m-%d') if hasattr(trade['卖出日期'],
                                                                                      'strftime') else trade[
                            '卖出日期'],
                        '买入价': f"{trade['买入价']:.2f}",
                        '卖出价': f"{trade['卖出价']:.2f}",
                        '股数': f"{trade['股数']}",
                        '收益率': f"{trade['收益率']:.2f}%",
                        '盈亏金额': f"{trade['盈亏金额']:.2f}",
                        '交易类型': trade['交易类型']
                    })
                pd.DataFrame(trade_data).to_excel(writer, sheet_name='交易记录', index=False)
            else:
                pd.DataFrame().to_excel(writer, sheet_name='交易记录', index=False)

    def run_complete_analysis(self, file_path: str, output_dir: str) -> Dict:
        """
        运行完整分析流程

        Args:
            file_path: 数据文件路径
            output_dir: 存放输出的文件夹

        Returns:
            绩效指标字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"开始 {self.stock_code} RSI策略分析...".center(80))
        print("=" * 80)

        # 加载数据
        self.load_data(file_path)
        print(f"股票代码: {self.stock_code}")
        print(f"数据行数: {len(self.data)}")

        # 数据预处理
        self.preprocess_data()
        print("数据预处理完成")

        # 运行回测
        self.run_backtest()
        print("回测完成")

        # 计算指标
        self.calculate_metrics()
        print("绩效指标计算完成")

        # 打印结果
        self._print_results()

        # 绘制图表
        plot_path = output_path / f"{self.stock_code}_RSI策略图表.png"
        self.plot_result(str(plot_path))
        print(f"图表已保存: {plot_path}")

        # 编写报告
        report_path = output_path / f"{self.stock_code}_RSI策略报告.xlsx"
        self.write_report(str(report_path))
        print(f"报告已保存: {report_path}")

        print("=" * 80)
        print("分析完成！".center(80))
        print("=" * 80)

        return self.metrics

    def _print_results(self):
        """打印回测结果"""
        s = self.metrics['strategy']
        h = self.metrics['hold']

        print("\n" + "=" * 100)
        print(f"{self.stock_name} ({self.stock_code}) - RSI策略回测结果".center(100))
        print("=" * 100)

        print(f"\n{'指标':<20} {'本策略':>15} {'买入持有':>15} {'对比':>15}")
        print("-" * 70)
        print(
            f"{'总收益率(%)':<20} {s['总收益率']:>15.2f} {h['总收益率']:>15.2f} {(s['总收益率'] - h['总收益率']):>+15.2f}")
        print(
            f"{'年化收益率(%)':<20} {s['年化收益率']:>15.2f} {h['年化收益率']:>15.2f} {(s['年化收益率'] - h['年化收益率']):>+15.2f}")
        print(
            f"{'最大回撤(%)':<20} {abs(s['最大回撤']):>15.2f} {abs(h['最大回撤']):>15.2f} {(abs(s['最大回撤']) - abs(h['最大回撤'])):>+15.2f}")
        print(
            f"{'夏普比率':<20} {s['夏普比率']:>15.2f} {h['夏普比率']:>15.2f} {(s['夏普比率'] - h['夏普比率']):>+15.2f}")
        print(f"{'交易次数':<20} {s['交易次数']:>15} {h['交易次数']:>15} {'-':>15}")
        print(f"{'胜率(%)':<20} {s['胜率']:>15.2f} {h['胜率']:>15} {'-':>15}")
        print(f"{'盈亏比':<20} {s['盈亏比']:>15.2f} {h['盈亏比']:>15} {'-':>15}")
        print("=" * 100)

    @staticmethod
    def compare_stocks(input_source: Union[str, List[str]],
                       initial_capital: float = 1000000,
                       commission_rate: float = 0.0001,
                       stamp_tax_rate: float = 0.001,
                       min_commission: float = 5,
                       stop_loss: float = None,
                       risk_free_rate: float = 0.03,
                       rsi_period: int = 14,
                       oversold_threshold: int = 30,
                       overbought_threshold: int = 70,
                       output_dir: str = "./output") -> pd.DataFrame:
        """
        多股票对比

        Args:
            input_source: 文件夹路径或数据文件路径列表
            initial_capital: 初始资金
            commission_rate: 佣金费率
            stamp_tax_rate: 印花税费率
            min_commission: 最低佣金
            stop_loss: 止损线
            risk_free_rate: 无风险利率
            rsi_period: RSI周期
            oversold_threshold: 超卖阈值
            overbought_threshold: 超买阈值
            output_dir: 输出文件夹

        Returns:
            对比结果数据框
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取所有股票文件
        stock_files = []
        stock_codes = []

        if isinstance(input_source, str):
            folder_path = Path(input_source)
            if not folder_path.exists():
                raise ValueError(f"文件夹不存在: {folder_path}")

            stock_files = list(folder_path.glob("*.csv"))
            for f in stock_files:
                stem = f.stem
                if '_' in stem:
                    parts = stem.split('_')
                    if len(parts) >= 3:
                        stock_codes.append(parts[2])
                    else:
                        stock_codes.append(stem)
                else:
                    stock_codes.append(stem)
        else:
            stock_files = [Path(f) for f in input_source]
            for f in stock_files:
                if not f.exists():
                    raise ValueError(f"文件不存在: {f}")
                stem = f.stem
                if '_' in stem:
                    parts = stem.split('_')
                    if len(parts) >= 3:
                        stock_codes.append(parts[2])
                    else:
                        stock_codes.append(stem)
                else:
                    stock_codes.append(stem)

        print(f"找到 {len(stock_files)} 个股票文件")

        # 执行回测
        results = []

        for i, (file_path, code) in enumerate(zip(stock_files, stock_codes)):
            print(f"\n▶ 正在处理 [{i + 1}/{len(stock_files)}] {code}...")

            try:
                # 创建策略实例
                strategy = RSIStrategy(
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    stamp_tax_rate=stamp_tax_rate,
                    min_commission=min_commission,
                    stop_loss=stop_loss,
                    risk_free_rate=risk_free_rate,
                    rsi_period=rsi_period,
                    oversold_threshold=oversold_threshold,
                    overbought_threshold=overbought_threshold
                )

                # 加载数据
                strategy.load_data(str(file_path))
                strategy.stock_code = code

                # 预处理
                strategy.preprocess_data()

                # 回测
                strategy.run_backtest()

                # 计算指标
                strategy.calculate_metrics()

                # 收集结果
                s = strategy.metrics['strategy']
                h = strategy.metrics['hold']

                # 计算回撤改善：正数表示策略回撤小于买入持有（表现更好）
                # 策略最大回撤（绝对值）越小越好，所以用买入持有回撤减去策略回撤
                strategy_dd_abs = abs(s['最大回撤'])
                hold_dd_abs = abs(h['最大回撤'])
                dd_improve = hold_dd_abs - strategy_dd_abs  # 正数表示改善，负数表示恶化

                results.append({
                    '股票代码': code,
                    '策略总收益率': s['总收益率'],
                    '策略年化收益率': s['年化收益率'],
                    '策略最大回撤率': strategy_dd_abs,  # 取绝对值
                    '策略夏普比率': s['夏普比率'],
                    '策略胜率': s['胜率'],
                    '策略盈亏比': s['盈亏比'],
                    '策略交易次数': s['交易次数'],
                    '买入持有总收益率': h['总收益率'],
                    '买入持有最大回撤率': hold_dd_abs,  # 取绝对值
                    '买入持有夏普比率': h['夏普比率'],
                    '超额总收益率': s['总收益率'] - h['总收益率'],
                    '回撤改善': dd_improve  # 正数表示改善
                })

                print(f"  ✓ 完成 - 收益率: {s['总收益率']:.2f}%")

            except Exception as e:
                print(f"  ✗ 失败: {str(e)}")
                continue

        if not results:
            raise ValueError("没有成功处理任何股票")

        # 创建对比数据框
        comparison = pd.DataFrame(results)

        # 保存明细表 - 格式化数值带%
        comparison_formatted = comparison.copy()
        for col in comparison_formatted.columns:
            if col == '股票代码':
                continue
            elif col in ['策略夏普比率', '买入持有夏普比率', '策略盈亏比']:
                comparison_formatted[col] = comparison_formatted[col].apply(lambda x: f"{x:.2f}")
            elif col == '策略交易次数':
                comparison_formatted[col] = comparison_formatted[col].apply(lambda x: f"{int(x)}")
            elif col in ['策略总收益率', '策略年化收益率', '策略最大回撤率', '策略胜率',
                         '买入持有总收益率', '买入持有最大回撤率', '超额总收益率', '回撤改善']:
                comparison_formatted[col] = comparison_formatted[col].apply(lambda x: f"{x:.2f}%")

        excel_path = output_path / "RSI_多股票对比明细表.xlsx"
        comparison_formatted.to_excel(excel_path, index=False)
        print(f"\n明细表已保存: {excel_path}")

        # 绘制对比图
        fig = RSIStrategy._plot_comparison_charts(comparison)
        chart_path = output_path / "RSI_多股票对比图.png"
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"对比图已保存: {chart_path}")

        # 控制台打印总体效果
        RSIStrategy._print_comparison_summary(comparison)

        return comparison

    @staticmethod
    def _plot_comparison_charts(comparison: pd.DataFrame) -> plt.Figure:
        """绘制多股票对比图表"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('多股票RSI策略对比分析', fontsize=16, fontweight='bold')

        # a. 最大回撤分布对比直方图
        ax1 = axes[0, 0]
        strategy_dd = comparison['策略最大回撤率'].dropna()
        hold_dd = comparison['买入持有最大回撤率'].dropna()

        bins = np.linspace(min(strategy_dd.min(), hold_dd.min()),
                           max(strategy_dd.max(), hold_dd.max()), 20)

        ax1.hist(strategy_dd, bins=bins, alpha=0.7, color='red', edgecolor='black', label='RSI策略')
        ax1.hist(hold_dd, bins=bins, alpha=0.5, color='blue', edgecolor='black', label='买入持有')

        ax1.axvline(strategy_dd.mean(), color='darkred', linestyle='--', linewidth=2,
                    label=f'策略均值: {strategy_dd.mean():.1f}%')
        ax1.axvline(hold_dd.mean(), color='darkblue', linestyle='--', linewidth=2,
                    label=f'买入持有均值: {hold_dd.mean():.1f}%')

        ax1.set_xlabel('最大回撤 (%)')
        ax1.set_ylabel('股票个数')
        ax1.set_title('a. 最大回撤分布对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # b. 胜率 vs 盈亏比四象限散点图
        ax2 = axes[0, 1]
        win_rate = comparison['策略胜率'].dropna()
        pl_ratio = comparison['策略盈亏比'].dropna()
        sharpe = comparison['策略夏普比率'].dropna()

        if len(win_rate) > 0:
            win_median = win_rate.median()
            pl_median = pl_ratio.median()

            ax2.axhline(y=pl_median, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(x=win_median, color='gray', linestyle='--', alpha=0.5)

            scatter = ax2.scatter(win_rate, pl_ratio, c=sharpe, cmap='RdYlGn',
                                  s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

            plt.colorbar(scatter, ax=ax2, label='夏普比率')

        ax2.set_xlabel('胜率 (%)')
        ax2.set_ylabel('盈亏比')
        ax2.set_title('b. 策略胜率 vs 盈亏比分布')
        ax2.grid(True, alpha=0.3)

        # c. 策略收益率 vs 买入持有收益率散点图
        ax3 = axes[1, 0]
        strategy_return = comparison['策略总收益率'].dropna()
        hold_return = comparison['买入持有总收益率'].dropna()
        excess_return = comparison['超额总收益率'].dropna()

        if len(strategy_return) > 0:
            max_val = max(strategy_return.max(), hold_return.max())
            min_val = min(strategy_return.min(), hold_return.min())

            ax3.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.5, linewidth=2,
                     label='收益率相等')

            scatter = ax3.scatter(hold_return, strategy_return, c=excess_return, cmap='RdYlGn',
                                  s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

            plt.colorbar(scatter, ax=ax3, label='超额收益(%)')

        ax3.set_xlabel('买入持有收益率 (%)')
        ax3.set_ylabel('策略收益率 (%)')
        ax3.set_title('c. 策略 vs 买入持有收益率对比')
        ax3.grid(True, alpha=0.3)

        # d. 超额收益 vs 回撤改善四象限散点图
        ax4 = axes[1, 1]
        excess_return = comparison['超额总收益率'].dropna()
        dd_improve = comparison['回撤改善'].dropna()
        sharpe = comparison['策略夏普比率'].dropna()

        if len(excess_return) > 0:
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            scatter = ax4.scatter(excess_return, dd_improve, c=sharpe, cmap='RdYlGn',
                                  s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

            plt.colorbar(scatter, ax=ax4, label='夏普比率')

        ax4.set_xlabel('超额收益 (%)')
        ax4.set_ylabel('回撤改善 (%)')
        ax4.set_title('d. 超额收益 vs 回撤改善分布')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def _print_comparison_summary(comparison: pd.DataFrame):
        """打印多股票对比总结"""
        print("\n" + "=" * 100)
        print("策略总体效果".center(100))
        print("=" * 100)

        # 跑赢基准比例
        beat_count = (comparison['超额总收益率'] > 0).sum()
        beat_pct = beat_count / len(comparison) * 100
        print(f"📊 跑赢基准比例: {beat_pct:.1f}% ({beat_count}/{len(comparison)})")

        # 回撤改善比例（回撤改善 > 0 表示策略回撤更小）
        improve_count = (comparison['回撤改善'] > 0).sum()
        improve_pct = improve_count / len(comparison) * 100
        print(f"📉 回撤改善比例: {improve_pct:.1f}% ({improve_count}/{len(comparison)})")

        # 平均回撤对比 - 取正数
        avg_strategy_dd = comparison['策略最大回撤率'].mean()
        avg_hold_dd = comparison['买入持有最大回撤率'].mean()
        print(f"📈 策略平均回撤 vs 买入持有平均回撤：{avg_strategy_dd:.1f}% vs {avg_hold_dd:.1f}%")

        # 平均夏普对比
        avg_strategy_sharpe = comparison['策略夏普比率'].mean()
        avg_hold_sharpe = comparison['买入持有夏普比率'].mean()
        print(f"⚡ 策略平均夏普 vs 买入持有平均夏普：{avg_strategy_sharpe:.2f} vs {avg_hold_sharpe:.2f}")

        # 平均指标
        avg_return = comparison['策略总收益率'].mean()
        avg_trades = comparison['策略交易次数'].mean()
        avg_win_rate = comparison['策略胜率'].mean()
        avg_pl_ratio = comparison['策略盈亏比'].mean()

        print(
            f"🎯 平均总收益率: {avg_return:.1f}%, 平均交易次数: {avg_trades:.0f}, 平均胜率: {avg_win_rate:.1f}%, 平均盈亏比: {avg_pl_ratio:.2f}")
        print("=" * 100)

    @staticmethod
    def compare_strategies(file_path: str,
                           param_combinations: List[Tuple[int, int, int]],
                           initial_capital: float = 1000000,
                           commission_rate: float = 0.0001,
                           stamp_tax_rate: float = 0.001,
                           min_commission: float = 5,
                           stop_loss: float = None,
                           risk_free_rate: float = 0.03,
                           output_dir: str = "./output") -> pd.DataFrame:
        """
        比较同一只股票，使用不同参数组合的RSI策略效果

        Args:
            file_path: 数据文件路径
            param_combinations: 参数组合列表，每个元素为 (rsi_period, oversold_threshold, overbought_threshold)
            initial_capital: 初始资金
            commission_rate: 佣金费率
            stamp_tax_rate: 印花税费率
            min_commission: 最低佣金
            stop_loss: 止损线
            risk_free_rate: 无风险利率
            output_dir: 输出文件夹

        Returns:
            对比结果数据框
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取股票代码
        file_stem = Path(file_path).stem
        if '_' in file_stem:
            parts = file_stem.split('_')
            if len(parts) >= 3:
                stock_code = parts[2]
            else:
                stock_code = file_stem
        else:
            stock_code = file_stem

        print("=" * 100)
        print(f"RSI参数对比分析 - 股票: {stock_code}".center(100))
        print("=" * 100)

        print(f"\n待测试的RSI参数组合 ({len(param_combinations)}组):")
        for i, params in enumerate(param_combinations):
            print(f"  组合{i + 1}: RSI周期={params[0]}, 超卖阈值={params[1]}, 超买阈值={params[2]}")

        # 执行回测
        results = []
        nav_data = {}

        for i, params in enumerate(param_combinations):
            rsi_period, oversold, overbought = params

            print(f"\n▶ 测试组合{i + 1}: RSI周期={rsi_period}, 超卖={oversold}, 超买={overbought}")

            try:
                # 创建策略实例
                strategy = RSIStrategy(
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    stamp_tax_rate=stamp_tax_rate,
                    min_commission=min_commission,
                    stop_loss=stop_loss,
                    risk_free_rate=risk_free_rate,
                    rsi_period=rsi_period,
                    oversold_threshold=oversold,
                    overbought_threshold=overbought
                )

                # 加载数据
                strategy.load_data(file_path)

                # 预处理
                strategy.preprocess_data()

                # 回测
                strategy.run_backtest()

                # 计算指标
                strategy.calculate_metrics()

                # 存储净值数据
                nav_key = f"策略{i + 1}({rsi_period},{oversold},{overbought})"
                nav_data[nav_key] = strategy.data[['日期', '策略净值']].copy()

                # 收集结果
                s = strategy.metrics['strategy']
                h = strategy.metrics['hold']

                # 计算回撤改善：正数表示策略回撤小于买入持有（表现更好）
                strategy_dd_abs = abs(s['最大回撤'])
                hold_dd_abs = abs(h['最大回撤'])
                dd_improve = hold_dd_abs - strategy_dd_abs  # 正数表示改善，负数表示恶化

                results.append({
                    '参数组合': nav_key,
                    'rsi_period': rsi_period,
                    'oversold': oversold,
                    'overbought': overbought,
                    '总收益率': s['总收益率'],
                    '年化收益率': s['年化收益率'],
                    '最大回撤率': strategy_dd_abs,  # 取绝对值
                    '夏普比率': s['夏普比率'],
                    '胜率': s['胜率'],
                    '盈亏比': s['盈亏比'],
                    '交易次数': s['交易次数'],
                    '超额收益': s['总收益率'] - h['总收益率'],
                    '回撤改善': dd_improve,  # 正数表示改善
                    'strategy': strategy
                })

                print(f"  ✓ 完成 - 收益率: {s['总收益率']:.2f}%, 夏普: {s['夏普比率']:.2f}")

            except Exception as e:
                print(f"  ✗ 失败: {str(e)}")
                continue

        if not results:
            raise ValueError("没有成功测试任何参数组合")

        # 获取买入持有数据
        base_strategy = RSIStrategy(initial_capital=initial_capital)
        base_strategy.load_data(file_path)
        base_strategy.preprocess_data()
        base_strategy.run_backtest()
        base_strategy.calculate_metrics()
        hold_return = base_strategy.metrics['hold']['总收益率']
        hold_dd = abs(base_strategy.metrics['hold']['最大回撤'])  # 取绝对值
        hold_sharpe = base_strategy.metrics['hold']['夏普比率']
        hold_nav = base_strategy.data[['日期', '收盘']].copy()
        hold_nav['策略净值'] = hold_nav['收盘'] / hold_nav.iloc[0]['收盘']

        # 创建对比数据框
        comparison = pd.DataFrame(results)

        # 添加买入持有作为最后一行
        hold_row = {
            '参数组合': '买入持有',
            '总收益率': hold_return,
            '年化收益率': base_strategy.metrics['hold']['年化收益率'],
            '最大回撤率': hold_dd,  # 已取绝对值
            '夏普比率': hold_sharpe,
            '胜率': 100 if hold_return > 0 else 0,
            '盈亏比': None,
            '交易次数': 1,
            '超额收益': 0,
            '回撤改善': 0
        }
        comparison = pd.concat([comparison, pd.DataFrame([hold_row])], ignore_index=True)

        # 保存详细数据 - 格式化数值带%
        detail_columns = ['参数组合', '总收益率', '年化收益率', '最大回撤率', '夏普比率',
                          '胜率', '盈亏比', '交易次数', '超额收益', '回撤改善']
        detail_df = comparison[detail_columns].copy()

        # 格式化数值
        for col in detail_df.columns:
            if col == '参数组合':
                continue
            elif col in ['夏普比率', '盈亏比']:
                detail_df[col] = detail_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            elif col == '交易次数':
                detail_df[col] = detail_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "-")
            elif col in ['总收益率', '年化收益率', '最大回撤率', '胜率', '超额收益', '回撤改善']:
                detail_df[col] = detail_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")

        excel_path = output_path / f"{stock_code}_RSI参数对比详细数据.xlsx"
        detail_df.to_excel(excel_path, index=False)
        print(f"\n详细数据已保存: {excel_path}")

        # 绘制净值曲线
        fig, ax = plt.subplots(figsize=(14, 8))

        # 绘制买入持有曲线
        ax.semilogy(hold_nav['日期'], hold_nav['策略净值'],
                    color='black', linewidth=2, linestyle='--', label='买入持有', alpha=0.7)

        # 为不同参数组合绘制曲线
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for i, (r, color) in enumerate(zip(results, colors)):
            nav_df = nav_data[r['参数组合']]
            ax.semilogy(nav_df['日期'], nav_df['策略净值'],
                        color=color, linewidth=1.5, label=r['参数组合'], alpha=0.8)

        ax.set_xlabel('日期')
        ax.set_ylabel('净值 (对数坐标)')
        ax.set_title(f'不同RSI参数策略净值对比 - {stock_code}')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # 设置日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.tick_params(axis='x', rotation=45)

        ax.axhline(y=1, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        plt.tight_layout()

        chart_path = output_path / f"{stock_code}_RSI参数对比净值曲线.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"净值曲线已保存: {chart_path}")

        return comparison

class BollStrategy:
    """布林带策略类"""

    def __init__(self,
                 initial_capital: float = 1000000.0,
                 commission_rate: float = 0.0001,
                 stamp_duty_rate: float = 0.001,
                 min_commission: float = 5.0,
                 stop_loss_rate: float = 0.05,
                 risk_free_rate: float = 0.02,
                 boll_period: int = 20,
                 boll_width: float = 2.0,
                 min_bandwidth: float = 0.02,
                 min_signal_interval: int = 5):
        """
        初始化布林带策略

        Parameters:
        -----------
        initial_capital : float
            初始资金，默认100万
        commission_rate : float
            佣金费率，默认万1
        stamp_duty_rate : float
            印花税费率（卖出收），默认千1
        min_commission : float
            最低佣金，默认5元
        stop_loss_rate : float
            止损线，默认5%
        risk_free_rate : float
            无风险利率，默认2%
        boll_period : int
            布林带周期，默认20
        boll_width : float
            布林带宽度系数，默认2
        min_bandwidth : float
            最小带宽比例，默认2%
        min_signal_interval : int
            最小信号间隔天数，默认5
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.min_commission = min_commission
        self.stop_loss_rate = stop_loss_rate
        self.risk_free_rate = risk_free_rate
        self.boll_period = boll_period
        self.boll_width = boll_width
        self.min_bandwidth = min_bandwidth
        self.min_signal_interval = min_signal_interval

        self.stock_code = None
        self.data = None
        self.result_df = None
        self.trades = []
        self.metrics = {}
        self.buy_hold = None

    def load_data(self, file_path: str) -> None:
        """
        加载数据

        Parameters:
        -----------
        file_path : str
            数据文件路径
        """
        # 从文件名提取股票代码
        filename = os.path.basename(file_path)
        match = re.search(r'df_pre_(\d+)', filename)
        if match:
            self.stock_code = match.group(1)
        else:
            self.stock_code = "未知"

        # 尝试多种编码方式读取文件
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin1']
        df = None

        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                print(f"成功使用 {enc} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                raise FileNotFoundError(f"文件 {file_path} 未找到")

        if df is None:
            raise ValueError("无法读取文件，请检查文件编码")

        self.data = df
        print(f"数据加载完成，共 {len(self.data)} 行")

    def preprocess_data(self) -> None:
        """数据处理：计算策略指标和交易信号"""
        if self.data is None:
            raise ValueError("请先加载数据")

        df = self.data.copy()

        # 确保日期列格式正确
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)

        # 计算布林带指标
        df['中轨'] = df['收盘'].shift(1).rolling(window=self.boll_period,
                                                 min_periods=self.boll_period).mean()
        df['std'] = df['收盘'].shift(1).rolling(window=self.boll_period,
                                                min_periods=self.boll_period).std()
        df['上轨'] = df['中轨'] + self.boll_width * df['std']
        df['下轨'] = df['中轨'] - self.boll_width * df['std']
        df['带宽'] = (df['上轨'] - df['下轨']) / df['中轨']

        # 初始化交易信号
        df['交易信号'] = None

        # 生成买入信号
        buy_condition = (
                (df['收盘'].shift(1) < df['下轨'].shift(1)) &  # 昨日收盘在下轨之下
                (df['收盘'] >= df['下轨']) &  # 今日收盘穿越下轨
                (df['收盘'] < df['中轨']) &  # 今日收盘在中轨之下
                (df['带宽'] >= self.min_bandwidth)  # 带宽不小于阈值
        )

        # 生成卖出信号
        sell_condition = (
                (df['收盘'].shift(1) > df['上轨'].shift(1)) &  # 昨日收盘在上轨之上
                (df['收盘'] <= df['上轨']) &  # 今日收盘穿越上轨
                (df['收盘'] > df['中轨'])  # 今日收盘在中轨之上
        )

        # 标记异常情况
        if '异常情况' in df.columns:
            df['异常情况'] = df['异常情况'].fillna('')
            abnormal = (df['异常情况'].str.contains('价格异常', na=False)) | \
                       (df['异常情况'].str.contains('停牌', na=False))
        else:
            abnormal = pd.Series([False] * len(df))

        # 标记原始信号
        df.loc[buy_condition & ~abnormal, '交易信号'] = '买入'
        df.loc[sell_condition & ~abnormal, '交易信号'] = '卖出'

        # 处理连续信号
        df['交易信号'] = self._clean_signals(df)

        # 计算ln(E)
        self._calculate_ln_e(df)

        self.data = df

    def _clean_signals(self, df: pd.DataFrame) -> pd.Series:
        """清理交易信号，确保首笔为买入且信号间隔足够"""
        cleaned_signals = []
        prev_signal = None
        prev_signal_date = None

        for idx, row in df.iterrows():
            current_signal = row['交易信号']
            current_date = row['日期']

            if current_signal is not None and not pd.isna(current_signal):
                if prev_signal is None:
                    # 第一个信号必须是买入
                    if current_signal == '买入':
                        cleaned_signals.append(current_signal)
                        prev_signal = '买入'
                        prev_signal_date = current_date
                    else:
                        cleaned_signals.append(None)
                else:
                    # 检查信号间隔
                    days_diff = (current_date - prev_signal_date).days
                    if current_signal != prev_signal and days_diff >= self.min_signal_interval:
                        cleaned_signals.append(current_signal)
                        prev_signal = current_signal
                        prev_signal_date = current_date
                    else:
                        cleaned_signals.append(None)
            else:
                cleaned_signals.append(None)

        return pd.Series(cleaned_signals)

    def _calculate_ln_e(self, df: pd.DataFrame) -> None:
        """计算ln(E)指标"""
        df['ln(E)'] = np.nan

        buy_indices = df[df['交易信号'] == '买入'].index

        for idx in buy_indices:
            start_loc = df.index.get_loc(idx) + 1
            end_loc = min(start_loc + 10, len(df))

            if start_loc < end_loc:
                buy_price = df.loc[idx, '收盘']
                future_prices = df.iloc[start_loc:end_loc]['收盘']

                # 计算A
                highest_price = future_prices.max()
                A = (highest_price - buy_price) / buy_price
                if A < 0:
                    A = 0

                # 计算B
                lowest_price = future_prices.min()
                B = (buy_price - lowest_price) / buy_price
                if B < 0:
                    B = 0

                # 计算ln(E)
                if B == 0 or A == 0:
                    ln_e = 10 if A > 0 else -10
                else:
                    ln_e = np.log(A / B)

                # 限制范围
                ln_e = max(min(ln_e, 10), -10)
                df.loc[idx, 'ln(E)'] = ln_e

    def run_backtest(self) -> pd.DataFrame:
        """运行策略回测"""
        if self.data is None:
            raise ValueError("请先进行数据处理")

        df = self.data.copy()

        # 初始化回测变量
        cash = self.initial_capital
        position = 0
        trades = []
        daily_net_values = []
        daily_positions = []

        in_position = False
        buy_price = 0.0
        buy_date = None
        stop_loss_triggered = False
        trade_pair_count = 0

        # 遍历每一天进行回测
        for i, row in df.iterrows():
            date = row['日期']
            close_price = float(row['收盘'])
            signal = row['交易信号']

            # 检查止损
            if in_position and not stop_loss_triggered:
                loss_ratio = (close_price - buy_price) / buy_price
                if loss_ratio <= -self.stop_loss_rate:
                    # 触发止损卖出
                    sell_price = close_price

                    # 计算卖出费用
                    sell_value = sell_price * position
                    commission = max(sell_value * self.commission_rate, self.min_commission)
                    stamp_duty = sell_value * self.stamp_duty_rate
                    cash_received = sell_value - commission - stamp_duty

                    # 更新现金
                    cash += cash_received

                    # 计算盈亏
                    buy_value = buy_price * position
                    buy_commission = max(buy_value * self.commission_rate, self.min_commission)
                    trade_profit = cash_received - buy_value - buy_commission

                    # 记录交易
                    trades.append({
                        '买入日期': buy_date,
                        '买入价格': buy_price,
                        '卖出日期': date,
                        '卖出价格': sell_price,
                        '股数': position,
                        '盈亏': trade_profit,
                        '盈亏比例': trade_profit / (buy_value + cash - cash_received),
                        '类型': '止损'
                    })
                    trade_pair_count += 1

                    in_position = False
                    position = 0
                    stop_loss_triggered = True

            # 处理买入信号
            if signal == '买入' and not in_position:
                max_shares = int(cash / (close_price * 100)) * 100

                if max_shares >= 100:
                    for shares in range(max_shares, 0, -100):
                        buy_value = close_price * shares
                        commission = max(buy_value * self.commission_rate, self.min_commission)
                        total_cost = buy_value + commission

                        if total_cost <= cash:
                            position = shares
                            cash -= total_cost
                            buy_price = close_price
                            buy_date = date
                            in_position = True
                            stop_loss_triggered = False
                            break

            # 处理卖出信号
            if signal == '卖出' and in_position:
                sell_price = close_price

                sell_value = sell_price * position
                commission = max(sell_value * self.commission_rate, self.min_commission)
                stamp_duty = sell_value * self.stamp_duty_rate
                cash_received = sell_value - commission - stamp_duty

                cash += cash_received

                buy_value = buy_price * position
                buy_commission = max(buy_value * self.commission_rate, self.min_commission)
                trade_profit = cash_received - buy_value - buy_commission

                trades.append({
                    '买入日期': buy_date,
                    '买入价格': buy_price,
                    '卖出日期': date,
                    '卖出价格': sell_price,
                    '股数': position,
                    '盈亏': trade_profit,
                    '盈亏比例': trade_profit / (buy_value + cash - cash_received),
                    '类型': '正常'
                })
                trade_pair_count += 1

                in_position = False
                position = 0

            # 计算当日总资产
            if in_position:
                total_value = cash + position * close_price
                daily_positions.append(1)
            else:
                total_value = cash
                daily_positions.append(0)

            daily_net_values.append(total_value)

        # 创建结果DataFrame
        self.result_df = pd.DataFrame({
            '日期': df['日期'],
            '策略净值': daily_net_values,
            '持仓状态': daily_positions
        })

        self.trades = trades
        self.data = df

        print(f"回测完成！")
        print(f"初始净值: {self.result_df['策略净值'].iloc[0]:.2f}")
        print(f"最终净值: {self.result_df['策略净值'].iloc[-1]:.2f}")
        print(f"总交易次数（完整买卖对）: {trade_pair_count}")

        return self.result_df

    def calculate_metrics(self) -> Dict:
        """计算绩效指标"""
        if self.result_df is None:
            raise ValueError("请先运行回测")

        # 计算买入持有策略
        self.buy_hold = BuyHoldStrategy(
            self.initial_capital,
            self.commission_rate,
            self.stamp_duty_rate,
            self.min_commission
        )
        bh_net_values = self.buy_hold.run_backtest(self.data)
        bh_metrics = self.buy_hold.calculate_metrics(self.risk_free_rate)

        # 策略净值序列
        net_series = pd.Series(
            self.result_df['策略净值'].values,
            index=pd.to_datetime(self.result_df['日期'])
        )

        # 总收益率
        strategy_total_return = (net_series.iloc[-1] / net_series.iloc[0]) - 1

        # 年化收益率
        days = (net_series.index[-1] - net_series.index[0]).days
        if days > 0:
            strategy_annual_return = (1 + strategy_total_return) ** (252 / days) - 1
        else:
            strategy_annual_return = 0

        # 最大回撤
        strategy_rolling_max = net_series.expanding().max()
        strategy_drawdown = (net_series - strategy_rolling_max) / strategy_rolling_max
        strategy_max_drawdown = strategy_drawdown.min()

        # 夏普比率
        strategy_daily_returns = net_series.pct_change().dropna()
        if len(strategy_daily_returns) > 0 and strategy_daily_returns.std() > 0:
            strategy_excess = strategy_daily_returns - self.risk_free_rate / 252
            strategy_sharpe = strategy_excess.mean() / strategy_daily_returns.std() * np.sqrt(252)
        else:
            strategy_sharpe = 0

        # 交易相关指标
        if len(self.trades) > 0:
            trade_count = len(self.trades)
            profitable_trades = [t for t in self.trades if t['盈亏'] > 0]
            win_rate = len(profitable_trades) / trade_count

            profits = [t['盈亏'] for t in self.trades if t['盈亏'] > 0]
            losses = [t['盈亏'] for t in self.trades if t['盈亏'] < 0]

            avg_profit = np.mean(profits) if profits else 0
            avg_loss = abs(np.mean(losses)) if losses else 1

            profit_loss_ratio = avg_profit / avg_loss if avg_loss != 0 else 0
        else:
            trade_count = 0
            win_rate = 0
            profit_loss_ratio = 0

        # 平均ln(E)
        ln_e_values = self.data['ln(E)'].dropna()
        avg_ln_e = ln_e_values.mean() if len(ln_e_values) > 0 else 0

        # 计算超额收益和回撤改善
        excess_return = strategy_total_return - bh_metrics['总收益率']
        drawdown_improvement =  abs(bh_metrics['最大回撤']) - abs(strategy_max_drawdown)

        self.metrics = {
            '策略': {
                '总收益率': strategy_total_return,
                '年化收益率': strategy_annual_return,
                '最大回撤': strategy_max_drawdown,
                '夏普比率': strategy_sharpe,
                '交易次数': trade_count,
                '胜率': win_rate,
                '盈亏比': profit_loss_ratio,
                '平均ln(E)': avg_ln_e
            },
            '买入持有': bh_metrics,
            '超额收益': excess_return,
            '回撤改善': drawdown_improvement
        }

        # 打印绩效对比表格
        self._print_metrics_table()

        return self.metrics

    def _print_metrics_table(self):
        """打印绩效指标对比表格"""
        print("\n" + "=" * 80)
        print("绩效指标对比")
        print("=" * 80)
        print(f"{'指标':<15} {'本策略':<25} {'买入持有':<25}")
        print("-" * 80)

        strategy = self.metrics['策略']
        bh = self.metrics['买入持有']

        for key in ['总收益率', '年化收益率', '最大回撤', '夏普比率', '交易次数', '胜率', '盈亏比']:
            strategy_val = strategy[key]
            bh_val = bh[key]

            if key in ['总收益率', '年化收益率', '最大回撤', '胜率']:
                strategy_str = f"{strategy_val:.2%}"
                bh_str = f"{bh_val:.2%}" if key != '交易次数' else f"{bh_val:.0f}"
            elif key == '夏普比率':
                strategy_str = f"{strategy_val:.2f}"
                bh_str = f"{bh_val:.2f}"
            elif key == '盈亏比':
                strategy_str = f"{strategy_val:.2f}"
                bh_str = f"{bh_val:.2f}"
            else:
                strategy_str = f"{strategy_val}"
                bh_str = f"{bh_val}"

            print(f"{key:<15} {strategy_str:<25} {bh_str:<25}")

        print("=" * 80)
        print(f"超额收益: {self.metrics['超额收益']:.2%}")
        print(f"回撤改善: {self.metrics['回撤改善']:.2%}")

    def plot_result(self, save_path: str = None) -> None:
        """
        绘制图表

        Parameters:
        -----------
        save_path : str
            图表保存路径，如果为None则显示不保存
        """
        if self.result_df is None:
            raise ValueError("请先运行回测")

        fig, axes = plt.subplots(7, 1, figsize=(16, 32))
        fig.suptitle(f'股票{self.stock_code} 布林带策略回测分析报告',
                     fontsize=16, fontweight='bold')

        df = self.data
        result_df = self.result_df

        # 计算买入持有净值
        bh_net_values = self.buy_hold.net_values

        # 子图1：价格 + 布林带 + 买卖点
        ax1 = axes[0]
        ax1.plot(df['日期'], df['收盘'], label='收盘价', color='black', linewidth=1, alpha=0.7)
        ax1.plot(df['日期'], df['中轨'], label='中轨', color='blue', linewidth=1, linestyle='--')
        ax1.plot(df['日期'], df['上轨'], label='上轨', color='red', linewidth=1, linestyle='--')
        ax1.plot(df['日期'], df['下轨'], label='下轨', color='green', linewidth=1, linestyle='--')

        buy_signals = df[df['交易信号'] == '买入']
        sell_signals = df[df['交易信号'] == '卖出']
        ax1.scatter(buy_signals['日期'], buy_signals['收盘'], color='red', marker='^', s=30, label='买入', zorder=5)
        ax1.scatter(sell_signals['日期'], sell_signals['收盘'], color='green', marker='v', s=30, label='卖出', zorder=5)

        ax1.set_title('价格与布林带（带买卖点）')
        ax1.set_ylabel('价格')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 子图2：持仓状态
        ax2 = axes[1]
        ax2.fill_between(result_df['日期'], 0, result_df['持仓状态'], alpha=0.5, color='blue', step='mid')
        ax2.set_title('持仓状态（1=持仓，0=空仓）')
        ax2.set_ylabel('持仓状态')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)

        # 子图3：策略净值 vs 买入持有净值
        ax3 = axes[2]
        ax3.plot(result_df['日期'], result_df['策略净值'], label='策略净值', color='blue', linewidth=2)
        ax3.plot(df['日期'], bh_net_values, label='买入持有净值', color='red', linewidth=2, alpha=0.7)
        ax3.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='初始资金')
        ax3.set_title('净值曲线对比')
        ax3.set_ylabel('净值')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        # 子图4：回撤曲线
        ax4 = axes[3]
        net_series = pd.Series(result_df['策略净值'].values, index=pd.to_datetime(result_df['日期']))
        rolling_max = net_series.expanding().max()
        drawdown = (net_series - rolling_max) / rolling_max * 100
        ax4.fill_between(result_df['日期'], 0, drawdown.values, alpha=0.3, color='red',
                         label=f'最大回撤: {self.metrics["策略"]["最大回撤"]:.2%}')
        ax4.set_title('策略回撤曲线')
        ax4.set_ylabel('回撤 (%)')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # 子图5：每日收益率直方图
        ax5 = axes[4]
        daily_returns = net_series.pct_change().dropna() * 100
        if len(daily_returns) > 0:
            ax5.hist(daily_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax5.axvline(x=daily_returns.mean(), color='red', linestyle='-', linewidth=2,
                        label=f'均值: {daily_returns.mean():.2f}%')
            ax5.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax5.set_title(f'每日收益率分布 (均值: {daily_returns.mean():.2f}%, 标准差: {daily_returns.std():.2f}%)')
        else:
            ax5.set_title('每日收益率分布 (无数据)')
        ax5.set_xlabel('收益率 (%)')
        ax5.set_ylabel('天数')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 子图6：收盘价到中轨距离散点图
        ax6 = axes[5]
        distance_to_middle = (df['收盘'] - df['中轨']) / df['中轨'] * 100
        ax6.scatter(df['日期'], distance_to_middle, alpha=0.5, s=5, color='blue')
        ax6.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax6.axhline(y=2, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='+2%')
        ax6.axhline(y=-2, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='-2%')
        ax6.set_title('收盘价到中轨距离分布')
        ax6.set_ylabel('距离中轨 (%)')
        ax6.legend(loc='best', fontsize=8)
        ax6.grid(True, alpha=0.3)

        # 子图7：距离直方图
        ax7 = axes[6]
        distance_clean = distance_to_middle.dropna()
        if len(distance_clean) > 0:
            mean_dist = distance_clean.mean()
            std_dist = distance_clean.std()

            ax7.hist(distance_clean, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax7.axvline(x=mean_dist, color='red', linestyle='-', linewidth=2,
                        label=f'均值: {mean_dist:.2f}%')
            ax7.axvline(x=mean_dist + 2 * std_dist, color='orange', linestyle='--', linewidth=2,
                        label=f'+2σ: {mean_dist + 2 * std_dist:.2f}%')
            ax7.axvline(x=mean_dist - 2 * std_dist, color='orange', linestyle='--', linewidth=2,
                        label=f'-2σ: {mean_dist - 2 * std_dist:.2f}%')
            ax7.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
            ax7.set_title(f'收盘价到中轨距离直方图 (均值: {mean_dist:.2f}%, 标准差: {std_dist:.2f}%)')
        ax7.set_xlabel('距离中轨 (%)')
        ax7.set_ylabel('频数')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到：{save_path}")
        else:
            plt.show()

    def write_report(self, report_path: str) -> None:
        """
        编写回测报告

        Parameters:
        -----------
        report_path : str
            报告保存路径（Excel文件）
        """
        if self.data is None or self.result_df is None:
            raise ValueError("请先完成回测")

        # 准备参数说明
        params_data = {
            '参数名称': [
                '股票代码',
                '初始资金（元）',
                '佣金费率',
                '印花税费率（卖出收）',
                '最低佣金（元）',
                '止损线',
                '无风险利率（年化）',
                '布林带周期',
                '布林带宽度系数',
                '最小带宽比例',
                '最小信号间隔（天）',
                '回测开始日期',
                '回测结束日期',
                '回测天数'
            ],
            '参数值': [
                self.stock_code,
                f'{self.initial_capital:,.0f}',
                f'{self.commission_rate:.4%}',
                f'{self.stamp_duty_rate:.4%}',
                f'{self.min_commission:.0f}',
                f'{self.stop_loss_rate:.2%}',
                f'{self.risk_free_rate:.2%}',
                f'{self.boll_period}',
                f'{self.boll_width}',
                f'{self.min_bandwidth:.2%}',
                f'{self.min_signal_interval}',
                self.data['日期'].min().strftime('%Y-%m-%d'),
                self.data['日期'].max().strftime('%Y-%m-%d'),
                f'{(self.data["日期"].max() - self.data["日期"].min()).days}'
            ]
        }
        params_df = pd.DataFrame(params_data)

        # 准备绩效指标
        strategy = self.metrics['策略']
        bh = self.metrics['买入持有']

        metrics_data = {
            '指标名称': ['总收益率', '年化收益率', '最大回撤', '夏普比率', '交易次数', '胜率', '盈亏比', '平均ln(E)'],
            '本策略指标值': [
                f'{strategy["总收益率"]:.2%}',
                f'{strategy["年化收益率"]:.2%}',
                f'{abs(strategy["最大回撤"]):.2%}',  # 转换为正数
                f'{strategy["夏普比率"]:.2f}',
                strategy["交易次数"],
                f'{strategy["胜率"]:.2%}',
                f'{strategy["盈亏比"]:.2f}',
                f'{strategy["平均ln(E)"]:.4f}'
            ],
            '买入持有指标值': [
                f'{bh["总收益率"]:.2%}',
                f'{bh["年化收益率"]:.2%}',
                f'{abs(bh["最大回撤"]):.2%}',  # 转换为正数
                f'{bh["夏普比率"]:.2f}',
                0,
                '0.00%',
                '0.00',
                '0.0000'
            ]
        }
        metrics_df = pd.DataFrame(metrics_data)

        # 准备日度数据
        daily_data = self.data.copy()
        daily_data['策略净值'] = self.result_df['策略净值'].values

        # 先初始化所有列为 float 类型
        daily_data['持仓市值'] = 0.0
        daily_data['可用资金'] = float(self.initial_capital)
        daily_data['总资产'] = daily_data['策略净值'].values

        # 重新计算持仓市值和可用资金
        cash = float(self.initial_capital)
        position = 0
        in_position = False

        for i, row in daily_data.iterrows():
            close_price = float(row['收盘'])
            signal = row['交易信号']

            # 模拟交易
            if signal == '买入' and not in_position:
                max_shares = int(cash / (close_price * 100)) * 100
                if max_shares >= 100:
                    for shares in range(max_shares, 0, -100):
                        buy_value = close_price * shares
                        commission = max(buy_value * self.commission_rate, self.min_commission)
                        total_cost = buy_value + commission

                        if total_cost <= cash:
                            position = shares
                            cash -= total_cost
                            in_position = True
                            break

            elif signal == '卖出' and in_position:
                sell_value = close_price * position
                commission = max(sell_value * self.commission_rate, self.min_commission)
                stamp_duty = sell_value * self.stamp_duty_rate
                cash_received = sell_value - commission - stamp_duty

                cash += cash_received
                position = 0
                in_position = False

            # 记录当日持仓市值和可用资金
            daily_data.loc[i, '持仓市值'] = float(position * close_price)
            daily_data.loc[i, '可用资金'] = float(cash)

        # 计算回撤率（转换为正数）
        net_series = pd.Series(daily_data['策略净值'].values, index=range(len(daily_data)))
        rolling_max = net_series.expanding().max()
        # 原回撤为负数，取绝对值转换为正数
        daily_data['回撤率'] = abs((net_series - rolling_max) / rolling_max)

        # 计算日收益率
        daily_data['日收益率'] = daily_data['策略净值'].pct_change()

        # 计算累计收益率
        daily_data['累计收益率'] = daily_data['策略净值'] / daily_data['策略净值'].iloc[0] - 1

        # 重命名列
        daily_data = daily_data.rename(columns={
            '中轨': '布林带中轨',
            '上轨': '布林带上轨',
            '下轨': '布林带下轨',
            '带宽': '布林带带宽',
            'ln(E)': 'E比率'
        })

        # 选择需要的列
        daily_columns = [
            '日期', '收盘', '异常情况', '布林带中轨', '布林带上轨',
            '布林带下轨', '布林带带宽', '交易信号', 'E比率',
            '策略净值', '持仓市值', '可用资金', '总资产',
            '回撤率', '日收益率', '累计收益率'
        ]

        # 确保所有列都存在
        for col in daily_columns:
            if col not in daily_data.columns:
                daily_data[col] = np.nan

        daily_df = daily_data[daily_columns].copy()

        # 格式化数值
        for col in ['收盘', '布林带中轨', '布林带上轨', '布林带下轨', '策略净值',
                    '持仓市值', '可用资金', '总资产']:
            if col in daily_df.columns:
                daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')
                daily_df[col] = daily_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else '')

        for col in ['布林带带宽', 'E比率', '回撤率', '日收益率', '累计收益率']:
            if col in daily_df.columns:
                daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')
                # 回撤率已经是正数，直接格式化
                daily_df[col] = daily_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')

        # 准备交易记录
        if len(self.trades) > 0:
            trades_data = []
            for trade in self.trades:
                trades_data.append({
                    '买入日期': trade['买入日期'],
                    '卖出日期': trade['卖出日期'],
                    '买入价': trade['买入价格'],
                    '卖出价': trade['卖出价格'],
                    '股数': trade['股数'],
                    '收益率': trade['盈亏比例'],
                    '盈亏金额': trade['盈亏'],
                    '交易类型': trade['类型']
                })
            trades_df = pd.DataFrame(trades_data)

            # 格式化
            trades_df['买入价'] = pd.to_numeric(trades_df['买入价'], errors='coerce').apply(lambda x: f'{x:.2f}')
            trades_df['卖出价'] = pd.to_numeric(trades_df['卖出价'], errors='coerce').apply(lambda x: f'{x:.2f}')
            trades_df['股数'] = trades_df['股数'].apply(lambda x: f'{x:.0f}')
            trades_df['收益率'] = pd.to_numeric(trades_df['收益率'], errors='coerce').apply(lambda x: f'{x:.2%}')
            trades_df['盈亏金额'] = pd.to_numeric(trades_df['盈亏金额'], errors='coerce').apply(lambda x: f'{x:.2f}')
        else:
            trades_df = pd.DataFrame(columns=['买入日期', '卖出日期', '买入价', '卖出价',
                                              '股数', '收益率', '盈亏金额', '交易类型'])

        # 保存Excel
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            params_df.to_excel(writer, sheet_name='参数说明', index=False)
            metrics_df.to_excel(writer, sheet_name='绩效指标', index=False)
            daily_df.to_excel(writer, sheet_name='日度数据', index=False)
            trades_df.to_excel(writer, sheet_name='交易记录', index=False)

            # 获取工作表对象进行格式设置
            workbook = writer.book

            # 设置参数说明表格式
            params_sheet = writer.sheets['参数说明']
            for column in params_sheet.columns:
                max_length = 0
                column = list(column)
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                params_sheet.column_dimensions[column[0].column_letter].width = adjusted_width

            # 设置绩效指标表格式
            metrics_sheet = writer.sheets['绩效指标']
            for column in metrics_sheet.columns:
                max_length = 0
                column = list(column)
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                metrics_sheet.column_dimensions[column[0].column_letter].width = adjusted_width

            # 设置日度数据表格式
            daily_sheet = writer.sheets['日度数据']
            for column in daily_sheet.columns:
                daily_sheet.column_dimensions[column[0].column_letter].width = 15

            # 设置交易记录表格式
            trades_sheet = writer.sheets['交易记录']
            for column in trades_sheet.columns:
                max_length = 0
                column = list(column)
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 30)
                trades_sheet.column_dimensions[column[0].column_letter].width = adjusted_width

        print(f"报告已生成：{report_path}")

    def run_complete_analysis(self, data_path: str, output_folder: str) -> None:
        """
        完整分析流程

        Parameters:
        -----------
        data_path : str
            数据文件路径
        output_folder : str
            输出文件夹
        """
        # 创建输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"已创建文件夹：{output_folder}")

        # 加载数据
        self.load_data(data_path)

        # 数据处理
        self.preprocess_data()
        print(f"买入信号数量：{(self.data['交易信号'] == '买入').sum()}")
        print(f"卖出信号数量：{(self.data['交易信号'] == '卖出').sum()}")

        # 运行回测
        self.run_backtest()

        # 计算绩效指标
        self.calculate_metrics()

        # 绘制图表
        plot_path = os.path.join(output_folder, f'{self.stock_code}_回测分析报告.png')
        self.plot_result(plot_path)

        # 生成报告
        report_path = os.path.join(output_folder, f'{self.stock_code}_布林带_回测报告.xlsx')
        self.write_report(report_path)

        print(f"\n分析完成！所有结果已保存到：{output_folder}")

    @staticmethod
    def compare_stocks(file_or_folder: Union[str, List[str]],
                       params: Dict,
                       output_folder: str) -> pd.DataFrame:
        """
        多股票对比

        Parameters:
        -----------
        file_or_folder : str or List[str]
            文件夹路径或文件路径列表
        params : Dict
            初始化参数
        output_folder : str
            输出文件夹

        Returns:
        --------
        pd.DataFrame : 对比明细表
        """
        # 获取所有文件
        if isinstance(file_or_folder, str):
            if os.path.isdir(file_or_folder):
                files = [os.path.join(file_or_folder, f) for f in os.listdir(file_or_folder)
                         if f.endswith('.csv') and 'df_pre' in f]
            else:
                files = [file_or_folder]
        else:
            files = file_or_folder

        print(f"找到 {len(files)} 个数据文件")

        results = []

        for file_path in files:
            try:
                print(f"\n处理文件：{os.path.basename(file_path)}")

                # 创建策略实例
                strategy = BollStrategy(**params)

                # 完整分析
                strategy.load_data(file_path)
                strategy.preprocess_data()
                strategy.run_backtest()
                strategy.calculate_metrics()

                # 提取结果
                stock_code = strategy.stock_code
                s = strategy.metrics['策略']
                bh = strategy.metrics['买入持有']

                results.append({
                    '股票代码': stock_code,
                    '策略总收益率': s['总收益率'],
                    '策略年化收益率': s['年化收益率'],
                    '策略最大回撤率': abs(s['最大回撤']),
                    '策略夏普比率': s['夏普比率'],
                    '策略胜率': s['胜率'],
                    '策略盈亏比': s['盈亏比'],
                    '策略交易次数': s['交易次数'],
                    '买入持有总收益率': bh['总收益率'],
                    '买入持有最大回撤率': abs(bh['最大回撤']),
                    '买入持有夏普比率': bh['夏普比率'],
                    '超额总收益率': s['总收益率'] - bh['总收益率'],
                    '回撤改善':abs(bh['最大回撤']) - abs(s['最大回撤']),
                    '平均ln(E)': s['平均ln(E)']
                })

            except Exception as e:
                print(f"处理文件 {file_path} 时出错：{e}")
                continue

        if not results:
            print("没有成功处理任何文件")
            return pd.DataFrame()

        # 创建对比DataFrame
        df_results = pd.DataFrame(results)

        # 创建输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 保存对比明细表
        excel_path = os.path.join(output_folder, '布林带_多股票对比明细表.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='对比明细', index=False)

        # 绘制对比图表
        BollStrategy._plot_comparison_charts(df_results, output_folder)

        # 打印策略总体效果
        BollStrategy._print_comparison_summary(df_results)

        return df_results

    @staticmethod
    def _plot_comparison_charts(df: pd.DataFrame, output_folder: str) -> None:
        """绘制多股票对比图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('布林带策略多股票对比分析', fontsize=16, fontweight='bold')

        # a. 最大回撤分布对比直方图
        ax1 = axes[0, 0]
        ax1.hist(df['策略最大回撤率'], bins=15, alpha=0.5, label='本策略', color='blue', edgecolor='black')
        ax1.hist(df['买入持有最大回撤率'], bins=15, alpha=0.5, label='买入持有', color='red', edgecolor='black')
        ax1.set_xlabel('最大回撤率')
        ax1.set_ylabel('股票个数')
        ax1.set_title('最大回撤分布对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # b. 本策略胜率 vs 盈亏比四象限散点图
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['策略胜率'], df['策略盈亏比'],
                              c=df['策略夏普比率'], cmap='RdYlGn',
                              s=50, alpha=0.7, edgecolor='black')
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('胜率')
        ax2.set_ylabel('盈亏比')
        ax2.set_title('胜率 vs 盈亏比 (颜色=夏普比率)')
        plt.colorbar(scatter, ax=ax2)
        ax2.grid(True, alpha=0.3)

        # c. 本策略收益率 vs 买入持有收益率散点图
        ax3 = axes[0, 2]
        scatter = ax3.scatter(df['买入持有总收益率'], df['策略总收益率'],
                              c=df['超额总收益率'], cmap='coolwarm',
                              s=50, alpha=0.7, edgecolor='black')

        # 添加45度线
        min_val = min(df['买入持有总收益率'].min(), df['策略总收益率'].min())
        max_val = max(df['买入持有总收益率'].max(), df['策略总收益率'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='45°线')

        ax3.set_xlabel('买入持有收益率')
        ax3.set_ylabel('本策略收益率')
        ax3.set_title('策略收益率对比 (颜色=超额收益)')
        plt.colorbar(scatter, ax=ax3)
        ax3.grid(True, alpha=0.3)

        # d. 超额收益 vs 回撤改善四象限散点图
        ax4 = axes[1, 0]
        scatter = ax4.scatter(df['超额总收益率'], df['回撤改善'],
                              c=df['策略夏普比率'], cmap='RdYlGn',
                              s=50, alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('超额收益率')
        ax4.set_ylabel('回撤改善')
        ax4.set_title('超额收益 vs 回撤改善 (颜色=夏普比率)')
        plt.colorbar(scatter, ax=ax4)
        ax4.grid(True, alpha=0.3)

        # e. 平均ln(E)直方图
        ax5 = axes[1, 1]
        ax5.hist(df['平均ln(E)'], bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax5.axvline(x=df['平均ln(E)'].mean(), color='red', linestyle='-',
                    linewidth=2, label=f"均值: {df['平均ln(E)'].mean():.4f}")
        ax5.axvline(x=df['平均ln(E)'].median(), color='blue', linestyle='--',
                    linewidth=2, label=f"中位数: {df['平均ln(E)'].median():.4f}")
        ax5.set_xlabel('平均ln(E)')
        ax5.set_ylabel('股票个数')
        ax5.set_title('平均ln(E)分布')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 留空一个子图
        axes[1, 2].axis('off')

        plt.tight_layout()

        # 保存图片
        plot_path = os.path.join(output_folder, '布林带_多股票对比图.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到：{plot_path}")
        plt.close()

    @staticmethod
    def _print_comparison_summary(df: pd.DataFrame) -> None:
        """打印多股票对比总体效果"""
        print("\n" + "=" * 80)
        print("策略总体效果")
        print("=" * 80)

        # f. 跑赢基准比例
        beat_benchmark = (df['超额总收益率'] > 0).sum() / len(df) * 100
        print(f"跑赢基准比例: {beat_benchmark:.2f}%")

        # g. 回撤改善比例
        improve_drawdown = (df['回撤改善'] > 0).sum() / len(df) * 100
        print(f"回撤改善比例: {improve_drawdown:.2f}%")

        # h. 策略平均回撤 vs 买入持有平均回撤
        avg_strategy_dd = df['策略最大回撤率'].mean()
        avg_bh_dd = df['买入持有最大回撤率'].mean()
        print(f"策略平均回撤: {avg_strategy_dd:.2%} vs 买入持有平均回撤: {avg_bh_dd:.2%}")

        # i. 策略平均夏普 vs 买入持有平均夏普
        avg_strategy_sharpe = df['策略夏普比率'].mean()
        avg_bh_sharpe = df['买入持有夏普比率'].mean()
        print(f"策略平均夏普: {avg_strategy_sharpe:.4f} vs 买入持有平均夏普: {avg_bh_sharpe:.4f}")

        # j. 平均ln(E)的统计
        print(f"平均ln(E): 中位数={df['平均ln(E)'].median():.4f}, "
              f"平均数={df['平均ln(E)'].mean():.4f}, "
              f"最大值={df['平均ln(E)'].max():.4f}, "
              f"最小值={df['平均ln(E)'].min():.4f}")

        # k. 平均总收益率、平均买入次数、平均胜率、平均盈亏比
        print(f"平均总收益率: {df['策略总收益率'].mean():.2%}")
        print(f"平均买入次数: {df['策略交易次数'].mean():.1f}")
        print(f"平均胜率: {df['策略胜率'].mean():.2%}")
        print(f"平均盈亏比: {df['策略盈亏比'].mean():.2f}")

        print("=" * 80)

    @staticmethod
    def compare_strategies(data_path: str,
                           param_combinations: List[Dict],
                           output_folder: str) -> pd.DataFrame:
        """
        比较同一只股票的不同参数组合

        Parameters:
        -----------
        data_path : str
            数据文件路径
        param_combinations : List[Dict]
            参数组合列表
        output_folder : str
            输出文件夹

        Returns:
        --------
        pd.DataFrame : 参数对比详细数据
        """
        # 创建输出文件夹
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 从文件名提取股票代码
        filename = os.path.basename(data_path)
        match = re.search(r'df_pre_(\d+)', filename)
        stock_code = match.group(1) if match else "未知"

        results = []
        net_value_curves = {}

        # 运行基准策略（默认参数）
        base_params = {
            'initial_capital': 1000000,
            'commission_rate': 0.0001,
            'stamp_duty_rate': 0.001,
            'min_commission': 5,
            'stop_loss_rate': 0.05,
            'risk_free_rate': 0.02,
            'boll_period': 20,
            'boll_width': 2,
            'min_bandwidth': 0.02,
            'min_signal_interval': 5
        }

        print("运行买入持有基准...")
        base_strategy = BollStrategy(**base_params)
        base_strategy.load_data(data_path)
        base_strategy.preprocess_data()
        base_strategy.run_backtest()
        base_strategy.calculate_metrics()
        bh_metrics = base_strategy.metrics['买入持有']

        # 测试每种参数组合
        for i, params in enumerate(param_combinations):
            try:
                # 合并默认参数
                test_params = base_params.copy()
                test_params.update(params)

                # 创建参数描述
                param_desc = []
                for key, value in params.items():
                    if key == 'boll_period':
                        param_desc.append(f"周期{value}")
                    elif key == 'boll_width':
                        param_desc.append(f"宽度{value}")
                    elif key == 'min_bandwidth':
                        param_desc.append(f"带宽{value:.2%}")
                    elif key == 'stop_loss_rate':
                        param_desc.append(f"止损{value:.0%}")
                    elif key == 'min_signal_interval':
                        param_desc.append(f"间隔{value}")

                param_name = "_".join(param_desc) if param_desc else f"组合{i + 1}"

                print(f"测试参数组合：{param_name}")

                # 运行策略
                strategy = BollStrategy(**test_params)
                strategy.load_data(data_path)
                strategy.preprocess_data()
                strategy.run_backtest()
                strategy.calculate_metrics()

                # 保存净值曲线
                net_value_curves[param_name] = strategy.result_df['策略净值'].values

                # 提取结果
                s = strategy.metrics['策略']
                excess_return = s['总收益率'] - bh_metrics['总收益率']
                drawdown_improvement =  abs(bh_metrics['最大回撤']) - abs(s['最大回撤'])

                results.append({
                    '参数组合': param_name,
                    '总收益率': s['总收益率'],
                    '年化收益率': s['年化收益率'],
                    '最大回撤率': abs(s['最大回撤']),
                    '夏普比率': s['夏普比率'],
                    '胜率': s['胜率'],
                    '盈亏比': s['盈亏比'],
                    '交易次数': s['交易次数'],
                    '超额收益': excess_return,
                    '回撤改善': drawdown_improvement
                })

            except Exception as e:
                print(f"参数组合 {params} 测试失败：{e}")
                continue

        if not results:
            print("没有成功测试任何参数组合")
            return pd.DataFrame()

        # 创建结果DataFrame
        df_results = pd.DataFrame(results)

        # 添加买入持有作为最后一行
        bh_row = {
            '参数组合': '买入持有',
            '总收益率': bh_metrics['总收益率'],
            '年化收益率': bh_metrics['年化收益率'],
            '最大回撤率': abs(bh_metrics['最大回撤']),
            '夏普比率': bh_metrics['夏普比率'],
            '胜率': 0,
            '盈亏比': 0,
            '交易次数': 0,
            '超额收益': 0,
            '回撤改善': 0
        }
        df_results = pd.concat([df_results, pd.DataFrame([bh_row])], ignore_index=True)

        # 保存详细数据
        excel_path = os.path.join(output_folder, f'{stock_code}_布林带参数对比详细数据.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='参数对比', index=False)

        # 绘制净值曲线对比图
        BollStrategy._plot_parameter_comparison(data_path, stock_code, net_value_curves,
                                                bh_metrics, output_folder)

        print(f"参数对比数据已保存到：{excel_path}")

        return df_results

    @staticmethod
    def _plot_parameter_comparison(data_path: str, stock_code: str,
                                   net_value_curves: Dict,
                                   bh_metrics: Dict,
                                   output_folder: str) -> None:
        """绘制参数对比净值曲线"""
        # 加载原始数据获取日期
        strategy = BollStrategy()
        strategy.load_data(data_path)
        dates = strategy.data['日期']

        # 计算买入持有净值
        bh_strategy = BuyHoldStrategy(1000000, 0.0001, 0.001, 5)
        bh_net_values = bh_strategy.run_backtest(strategy.data)

        fig, ax = plt.subplots(figsize=(14, 8))

        # 绘制每个参数组合的净值曲线
        colors = plt.cm.tab10(np.linspace(0, 1, len(net_value_curves)))
        for (param_name, net_values), color in zip(net_value_curves.items(), colors):
            ax.plot(dates, net_values, label=param_name, color=color, linewidth=1.5)

        # 绘制买入持有曲线
        ax.plot(dates, bh_net_values, label='买入持有', color='black',
                linewidth=2.5, linestyle='--', alpha=0.7)

        ax.set_title(f'股票{stock_code} 不同参数布林带策略净值对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('净值')
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        plot_path = os.path.join(output_folder, f'{stock_code}_布林带参数对比净值曲线.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"参数对比图已保存到：{plot_path}")
        plt.close()

class StrategyCompare:
    """
    策略对比分析类

    用于比较多个量化策略的回测结果，生成对比报告和可视化图表。
    """

    def __init__(self, input_path: Union[str, List[str]], output_dir: str):
        """
        初始化策略对比类

        Parameters:
        -----------
        input_path : str or List[str]
            策略分析结果文件路径列表，或者包含所有文件的文件夹路径
        output_dir : str
            输出结果保存文件夹路径
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.strategy_data = {}
        self.summary_stats = {}
        self.load_errors = []

        os.makedirs(output_dir, exist_ok=True)
        self._load_data()

    def _extract_strategy_name(self, filename: str) -> str:
        """从文件名中获取“_”之前的部分作为策略名称"""
        name_without_ext = Path(filename).stem
        if '_' in name_without_ext:
            return name_without_ext.split('_')[0]
        return name_without_ext

    def _safe_mean(self, series: pd.Series) -> float:
        """安全地计算平均值，处理inf和异常值"""
        if series.dtype not in ['float64', 'int64']:
            try:
                series = pd.to_numeric(series, errors='coerce')
            except:
                return np.nan

        clean_series = series.replace([np.inf, -np.inf], np.nan)
        valid_data = clean_series[np.isfinite(clean_series)]

        if len(valid_data) == 0:
            return np.nan
        return valid_data.mean()

    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """将DataFrame中的数值列从字符串转换为数值类型"""
        numeric_fields = [
            '策略总收益率', '策略年化收益率', '策略最大回撤率', '策略夏普比率',
            '策略胜率', '策略盈亏比', '策略交易次数', '买入持有总收益率',
            '买入持有最大回撤率', '买入持有夏普比率', '超额总收益率', '回撤改善'
        ]

        df_converted = df.copy()

        for field in numeric_fields:
            if field in df_converted.columns:
                if df_converted[field].dtype == 'object' or df_converted[field].dtype == 'string':
                    try:
                        if df_converted[field].astype(str).str.contains('%').any():
                            df_converted[field] = df_converted[field].astype(str).str.replace('%', '')
                            df_converted[field] = pd.to_numeric(df_converted[field], errors='coerce') / 100
                        else:
                            df_converted[field] = pd.to_numeric(df_converted[field], errors='coerce')
                    except:
                        df_converted[field] = np.nan
                else:
                    df_converted[field] = pd.to_numeric(df_converted[field], errors='coerce')

                if df_converted[field].dtype in ['float64', 'int64']:
                    df_converted[field] = df_converted[field].replace([np.inf, -np.inf], np.nan)

        return df_converted

    def _load_data(self):
        """加载策略数据"""
        file_paths = []

        if isinstance(self.input_path, str):
            if os.path.isdir(self.input_path):
                for file in os.listdir(self.input_path):
                    if file.endswith(('.xlsx', '.xls')):
                        file_paths.append(os.path.join(self.input_path, file))
            else:
                raise ValueError(f"输入的路径不是文件夹: {self.input_path}")
        elif isinstance(self.input_path, list):
            file_paths = [f for f in self.input_path if f.endswith(('.xlsx', '.xls'))]
        else:
            raise TypeError("input_path必须是文件夹路径或文件路径列表")

        if not file_paths:
            raise ValueError("没有找到Excel文件")

        print(f"找到 {len(file_paths)} 个策略文件")

        loaded_count = 0
        for file_path in file_paths:
            try:
                strategy_name = self._extract_strategy_name(os.path.basename(file_path))
                print(f"\n正在加载策略: {strategy_name}")

                df = pd.read_excel(file_path)

                required_fields = ['策略总收益率', '买入持有总收益率']
                missing_critical = [f for f in required_fields if f not in df.columns]
                if missing_critical:
                    self.load_errors.append(f"{strategy_name}: 缺少关键字段 {missing_critical}")
                    continue

                df = self._convert_to_numeric(df)

                if df['策略总收益率'].isnull().all():
                    self.load_errors.append(f"{strategy_name}: 策略总收益率全部为NaN")
                    continue

                self.strategy_data[strategy_name] = df
                loaded_count += 1
                print(f"  ✓ 成功加载策略: {strategy_name}")

            except Exception as e:
                self.load_errors.append(f"{os.path.basename(file_path)}: 加载失败 - {str(e)}")

        if loaded_count == 0:
            raise ValueError("没有成功加载任何策略数据")

        print(f"\n总共成功加载 {loaded_count} 个策略")
        if self.load_errors:
            print(f"\n加载过程中出现 {len(self.load_errors)} 个错误:")
            for error in self.load_errors:
                print(f"  - {error}")

    def _calculate_summary_stats(self):
        """计算每个策略的汇总统计指标"""
        buy_hold_returns = []

        for strategy_name, df in self.strategy_data.items():
            print(f"\n计算策略 {strategy_name} 的统计指标:")

            try:
                valid_mask = df['策略总收益率'].notna() & np.isfinite(df['策略总收益率'])
                valid_count = valid_mask.sum()
                print(f"  有效样本数: {valid_count}/{len(df)}")

                if valid_count == 0:
                    continue

                if '买入持有总收益率' in df.columns:
                    both_valid = valid_mask & df['买入持有总收益率'].notna() & np.isfinite(df['买入持有总收益率'])
                    win_benchmark = (df.loc[both_valid, '策略总收益率'] > df.loc[
                        both_valid, '买入持有总收益率']).sum() / both_valid.sum() if both_valid.sum() > 0 else np.nan
                else:
                    win_benchmark = np.nan

                if '策略最大回撤率' in df.columns and '买入持有最大回撤率' in df.columns:
                    drawdown_mask = (df['策略最大回撤率'].notna() & np.isfinite(df['策略最大回撤率']) &
                                     df['买入持有最大回撤率'].notna() & np.isfinite(df['买入持有最大回撤率']))
                    drawdown_improve = (df.loc[drawdown_mask, '策略最大回撤率'] < df.loc[
                        drawdown_mask, '买入持有最大回撤率']).sum() / drawdown_mask.sum() if drawdown_mask.sum() > 0 else np.nan
                else:
                    drawdown_improve = np.nan

                avg_return = self._safe_mean(df['策略总收益率'])
                avg_win_rate = self._safe_mean(df['策略胜率']) * 100 if '策略胜率' in df.columns else np.nan
                avg_win_loss = self._safe_mean(df['策略盈亏比']) if '策略盈亏比' in df.columns else np.nan
                avg_trades = self._safe_mean(df['策略交易次数']) if '策略交易次数' in df.columns else np.nan

                print(f"  跑赢基准比率: {win_benchmark:.4f}" if not np.isnan(win_benchmark) else "  跑赢基准比率: NaN")
                print(f"  平均收益率: {avg_return:.6f}" if not np.isnan(avg_return) else "  平均收益率: NaN")
                print(f"  平均胜率: {avg_win_rate:.4f}%" if not np.isnan(avg_win_rate) else "  平均胜率: NaN")
                print(f"  平均盈亏比: {avg_win_loss:.4f}" if not np.isnan(avg_win_loss) else "  平均盈亏比: NaN")

                self.summary_stats[strategy_name] = {
                    '策略名称': strategy_name,
                    '跑赢基准比率': win_benchmark,
                    '回撤改善比率': drawdown_improve,
                    '平均收益率': avg_return,
                    '平均胜率': avg_win_rate,
                    '平均盈亏比': avg_win_loss,
                    '平均交易次数': avg_trades
                }

                if '买入持有总收益率' in df.columns:
                    clean_bh = df['买入持有总收益率'].replace([np.inf, -np.inf], np.nan).dropna()
                    buy_hold_returns.extend(clean_bh.tolist())

            except Exception as e:
                print(f"  计算策略 {strategy_name} 时出错: {e}")
                continue

        avg_buy_hold_return = np.mean(buy_hold_returns) if buy_hold_returns else 0

        self.summary_stats['买入持有'] = {
            '策略名称': '买入持有',
            '跑赢基准比率': np.nan,
            '回撤改善比率': np.nan,
            '平均收益率': avg_buy_hold_return,
            '平均胜率': np.nan,
            '平均盈亏比': np.nan,
            '平均交易次数': 1.0
        }

        print(f"\n买入持有平均收益率: {avg_buy_hold_return:.6f}")

    def generate_excel_report(self) -> str:
        """生成多策略对比分析报告Excel文件"""
        print("\n生成Excel报告...")
        self._calculate_summary_stats()

        df_summary = pd.DataFrame(list(self.summary_stats.values()))
        df_to_save = df_summary.copy()

        df_display = df_summary.copy()
        df_display['跑赢基准比率'] = df_display['跑赢基准比率'].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(x) else "-")
        df_display['回撤改善比率'] = df_display['回撤改善比率'].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(x) else "-")
        df_display['平均收益率'] = df_display['平均收益率'].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(x) else "-")
        df_display['平均胜率'] = df_display['平均胜率'].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) and np.isfinite(x) else "-")
        df_display['平均盈亏比'] = df_display['平均盈亏比'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "-")
        df_display['平均交易次数'] = df_display['平均交易次数'].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) and np.isfinite(x) else "-")

        output_path = os.path.join(self.output_dir, '多策略对比分析报告.xlsx')

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_display.to_excel(writer, sheet_name='策略对比报告', index=False)
            df_to_save.to_excel(writer, sheet_name='原始数据', index=False)

        print(f"✓ Excel报告已生成: {output_path}")
        return output_path

    def plot_scatter(self, strategy1: str, strategy2: str) -> str:
        """生成两个策略的收益率对比散点图"""
        if strategy1 not in self.strategy_data or strategy2 not in self.strategy_data:
            print(f"策略名称不存在，可用策略: {list(self.strategy_data.keys())}")
            return None

        df1 = self.strategy_data[strategy1]
        df2 = self.strategy_data[strategy2]

        if '股票代码' not in df1.columns or '股票代码' not in df2.columns:
            return None

        merged = pd.merge(
            df1[['股票代码', '策略总收益率']],
            df2[['股票代码', '策略总收益率']],
            on='股票代码',
            suffixes=(f'_{strategy1}', f'_{strategy2}')
        )

        if len(merged) == 0:
            return None

        plt.figure(figsize=(10, 8))

        x = merged[f'策略总收益率_{strategy1}']
        y = merged[f'策略总收益率_{strategy2}']

        valid_mask = ~(np.isnan(x) | np.isnan(y) | ~np.isfinite(x) | ~np.isfinite(y))
        x = x[valid_mask]
        y = y[valid_mask]

        if len(x) == 0:
            return None

        plt.scatter(x, y, alpha=0.6, edgecolors='white', linewidth=0.5)

        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')

        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot([min_val, max_val], [p(min_val), p(max_val)], 'b-', alpha=0.7,
                     label=f'趋势线 (斜率={z[0]:.3f})')
        except:
            pass

        plt.xlabel(f'{strategy1} 收益率')
        plt.ylabel(f'{strategy2} 收益率')
        plt.title(f'{strategy1} vs {strategy2} 收益率对比')
        plt.grid(True, alpha=0.3)
        plt.legend()

        corr = np.corrcoef(x, y)[0, 1]
        plt.text(0.05, 0.95, f'相关系数: {corr:.4f}\n样本数: {len(x)}',
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, f'scatter_{strategy1}_vs_{strategy2}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_all_scatter(self):
        """为所有策略两两生成散点图"""
        strategy_names = list(self.strategy_data.keys())

        if len(strategy_names) < 2:
            return

        print("\n生成两两对比散点图...")
        for i in range(len(strategy_names)):
            for j in range(i + 1, len(strategy_names)):
                try:
                    self.plot_scatter(strategy_names[i], strategy_names[j])
                except Exception as e:
                    print(f"生成 {strategy_names[i]} vs {strategy_names[j]} 散点图失败: {e}")

    def plot_histograms(self) -> Dict[str, str]:
        """生成三个直方图：总收益率对比、最大回撤率对比、胜率对比"""
        print("\n生成分布直方图...")

        all_returns = []
        all_drawdowns = []
        all_win_rates = []
        strategy_labels = []

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.strategy_data)))

        for i, (strategy_name, df) in enumerate(self.strategy_data.items()):
            returns = df['策略总收益率'].replace([np.inf, -np.inf],
                                                 np.nan).dropna() if '策略总收益率' in df.columns else pd.Series([])
            drawdowns = df['策略最大回撤率'].replace([np.inf, -np.inf],
                                                     np.nan).dropna() if '策略最大回撤率' in df.columns else pd.Series(
                [])
            win_rates = df['策略胜率'].replace([np.inf, -np.inf],
                                               np.nan).dropna() * 100 if '策略胜率' in df.columns else pd.Series([])

            if len(returns) > 0:
                all_returns.append(returns)
                all_drawdowns.append(drawdowns)
                all_win_rates.append(win_rates)
                strategy_labels.append(strategy_name)
                print(f"  {strategy_name}: 有效收益率样本数={len(returns)}")

        if not all_returns:
            return {}

        plt.figure(figsize=(14, 8))
        for i, (returns, label) in enumerate(zip(all_returns, strategy_labels)):
            plt.hist(returns, bins=30, alpha=0.5, label=label,
                     color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

        plt.xlabel('总收益率', fontsize=12)
        plt.ylabel('样本数', fontsize=12)
        plt.title('各策略总收益率分布对比', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        returns_hist_path = os.path.join(self.output_dir, 'histogram_returns.png')
        plt.savefig(returns_hist_path, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(14, 8))
        for i, (drawdowns, label) in enumerate(zip(all_drawdowns, strategy_labels)):
            if len(drawdowns) > 0:
                plt.hist(drawdowns, bins=30, alpha=0.5, label=label,
                         color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

        plt.xlabel('最大回撤率', fontsize=12)
        plt.ylabel('样本数', fontsize=12)
        plt.title('各策略最大回撤率分布对比', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        drawdowns_hist_path = os.path.join(self.output_dir, 'histogram_drawdowns.png')
        plt.savefig(drawdowns_hist_path, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(14, 8))
        for i, (win_rates, label) in enumerate(zip(all_win_rates, strategy_labels)):
            if len(win_rates) > 0:
                plt.hist(win_rates, bins=30, alpha=0.5, label=label,
                         color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

        plt.xlabel('胜率 (%)', fontsize=12)
        plt.ylabel('样本数', fontsize=12)
        plt.title('各策略胜率分布对比', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        win_rates_hist_path = os.path.join(self.output_dir, 'histogram_win_rates.png')
        plt.savefig(win_rates_hist_path, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'returns_hist': returns_hist_path,
            'drawdowns_hist': drawdowns_hist_path,
            'win_rates_hist': win_rates_hist_path
        }

    def run_full_analysis(self) -> Dict:
        """运行完整的策略对比分析"""
        print("=" * 50)
        print("开始策略对比分析")
        print("=" * 50)

        print(f"\n当前成功加载的策略: {list(self.strategy_data.keys())}")

        results = {}
        results['excel_report'] = self.generate_excel_report()
        self.plot_all_scatter()
        results.update(self.plot_histograms())

        print("=" * 50)
        print("策略对比分析完成!")
        print(f"所有结果已保存至: {self.output_dir}")
        print("=" * 50)

        return results
