import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union, Any
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
from tabulate import tabulate


class DoubleMovingAverageStrategy:
    """
    双均线策略回测类
    策略逻辑：短期均线上穿长期均线买入，下穿长期均线卖出
    考虑涨跌停限制：涨停时不能买入，跌停时不能卖出，交易延迟到下一个交易日

    参数可自定义：
    - short_ma: 短期均线周期 (默认: 10)
    - long_ma: 长期均线周期 (默认: 120)
    """

    def __init__(self, short_ma=10, long_ma=120, initial_capital=100000,
                 commission_rate=0.0001, min_commission=5, stamp_tax_rate=0.001,
                 risk_free_rate=0.02):
        """
        初始化策略参数

        参数:
        short_ma: 短期均线周期 (默认: 10日均线)
        long_ma: 长期均线周期 (默认: 120日均线)
        initial_capital: 初始资金 (默认: 100000)
        commission_rate: 手续费率 (默认: 0.0001, 万1)
        min_commission: 最低手续费 (默认: 5)
        stamp_tax_rate: 印花税率 (默认: 0.001, 千1, 仅卖出)
        risk_free_rate: 无风险利率 (默认: 0.02, 用于夏普比率计算)
        """
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.risk_free_rate = risk_free_rate

        # 数据相关
        self.data = None
        self.results = None

        # 策略指标
        self.metrics = {}
        self.buyhold_metrics = {}

        # 验证参数有效性
        if short_ma >= long_ma:
            print("警告：短期均线周期应小于长期均线周期，否则可能导致信号异常")

    def load_data(self, filepath, encoding='gbk', stock_code=None):
        """
        加载数据

        参数:
        filepath: 数据文件路径
        encoding: 文件编码 (默认: 'gbk')
        stock_code: 股票代码，如果不提供则从文件名提取
        """
        self.data = pd.read_csv(filepath, encoding=encoding)

        # 设置股票代码
        if stock_code:
            self.stock_code = stock_code
        else:
            # 尝试从文件名提取股票代码
            filename = os.path.basename(filepath)
            # 假设文件名格式如 "df_pre_000001.csv"
            import re
            match = re.search(r'(\d{6})', filename)
            self.stock_code = match.group(1) if match else filename

        print(f"股票 {self.stock_code} 数据加载完成，共{len(self.data)}条记录")
        return self

    def preprocess_data(self):
        """数据预处理，计算所需字段"""
        if self.data is None:
            raise ValueError("请先加载数据")

        # 确保日期列存在并处理
        if '日期' not in self.data.columns:
            raise ValueError("数据中缺少'日期'列")

        # 初始化交易相关字段
        self.data['position'] = 0  # 持仓股数
        self.data['cash'] = 0.0  # 资金余额
        self.data['hold_value'] = 0.0  # 持仓市值
        self.data['commission'] = 0.0  # 手续费
        self.data['stamp_tax'] = 0.0  # 印花税

        # 计算均线（shift(1)确保使用前一天的收盘价计算）
        ma_short_name = f'ma_{self.short_ma}'
        ma_long_name = f'ma_{self.long_ma}'

        self.data[ma_short_name] = self.data['收盘'].rolling(window=self.short_ma).mean().shift(1)
        self.data[ma_long_name] = self.data['收盘'].rolling(window=self.long_ma).mean().shift(1)

        # 存储均线列名供后续使用
        self.ma_short_col = ma_short_name
        self.ma_long_col = ma_long_name

        # 识别金叉和死叉
        self.data['golden_cross'] = (self.data[ma_short_name] > self.data[ma_long_name]) & \
                                    (self.data[ma_short_name].shift(1) <= self.data[ma_long_name].shift(1))
        self.data['death_cross'] = (self.data[ma_short_name] < self.data[ma_long_name]) & \
                                   (self.data[ma_short_name].shift(1) >= self.data[ma_long_name].shift(1))

        print(f"数据预处理完成，使用均线: MA{self.short_ma} 和 MA{self.long_ma}")
        return self

    def _check_limit(self, row):
        """检查涨跌停"""
        if row['开盘'] == 0:  # 避免除零
            return False, False

        change_pct = (row['收盘'] - row['开盘']) / row['开盘']
        limit_up = change_pct >= 0.099
        limit_down = change_pct <= -0.099
        return limit_up, limit_down

    def generate_signals(self):
        """生成交易信号"""
        if self.data is None:
            raise ValueError("请先加载和预处理数据")

        position = 0
        signals = []
        pending = None

        for i in range(len(self.data)):
            row = self.data.iloc[i]
            signal = ''

            limit_up, limit_down = self._check_limit(row)

            # 执行待处理的延迟信号
            if pending == 'buy' and not limit_up:
                signal = 'buy'
                position = 1
                pending = None
            elif pending == 'sell' and not limit_down:
                signal = 'sell'
                position = 0
                pending = None

            # 触发新信号
            elif position == 0 and row['golden_cross'] and not pending:
                if limit_up:
                    pending = 'buy'
                else:
                    signal = 'buy'
                    position = 1

            elif position == 1 and row['death_cross'] and not pending:
                if limit_down:
                    pending = 'sell'
                else:
                    signal = 'sell'
                    position = 0

            signals.append(signal)

        self.data['signal'] = signals
        print(f"信号生成完成，买入信号: {(self.data['signal'] == 'buy').sum()}次，"
              f"卖出信号: {(self.data['signal'] == 'sell').sum()}次")
        return self

    def run_backtest(self):
        """运行回测"""
        if self.data is None or 'signal' not in self.data.columns:
            raise ValueError("请先生成交易信号")

        cash = self.initial_capital
        position = 0

        for i in range(len(self.data)):
            row = self.data.iloc[i]

            if row['signal'] == 'buy':
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
                        cash -= total_cost
                        self.data.at[i, 'commission'] = commission
                        self.data.at[i, 'stamp_tax'] = 0
                        break
                    else:
                        max_shares -= 100

            elif row['signal'] == 'sell':
                if position > 0:
                    price = row['收盘']
                    trade_amount = position * price
                    commission = max(trade_amount * self.commission_rate, self.min_commission)
                    stamp_tax = trade_amount * self.stamp_tax_rate
                    cash += trade_amount - commission - stamp_tax
                    self.data.at[i, 'commission'] = commission
                    self.data.at[i, 'stamp_tax'] = stamp_tax
                    position = 0

            # 记录每日状态
            self.data.at[i, 'position'] = position
            self.data.at[i, 'cash'] = cash
            self.data.at[i, 'hold_value'] = position * row['收盘']

        # 计算每日总资产
        self.data['total_asset'] = self.data['cash'] + self.data['hold_value']
        self.data['daily_return'] = self.data['total_asset'].pct_change()

        print("回测执行完成")
        return self

    def calculate_metrics(self):
        """计算策略绩效指标"""
        if self.data is None:
            raise ValueError("请先运行回测")

        try:
            # 最终资产
            final_price = self.data['收盘'].iloc[-1]
            final_asset = self.data['cash'].iloc[-1] + self.data['position'].iloc[-1] * final_price

            # 基础指标
            trading_days = len(self.data)
            self.metrics['total_return'] = final_asset / self.initial_capital - 1

            # 安全计算年化收益率
            if self.metrics['total_return'] > -1:  # 确保总收益率大于-100%
                try:
                    self.metrics['annual_return'] = (1 + self.metrics['total_return']) ** (252 / trading_days) - 1
                except (ValueError, ZeroDivisionError):
                    self.metrics['annual_return'] = 0
            else:
                self.metrics['annual_return'] = -1  # 全部亏损的情况

            # 最大回撤
            self.data['cum_max'] = self.data['total_asset'].cummax()

            # 避免除零
            mask = self.data['cum_max'] > 0
            self.data['drawdown'] = 0.0
            self.data.loc[mask, 'drawdown'] = 1 - self.data.loc[mask, 'total_asset'] / self.data.loc[mask, 'cum_max']

            self.metrics['max_drawdown'] = self.data['drawdown'].max()

            # 夏普比率（处理除零）
            daily_returns = self.data['daily_return'].dropna()

            if len(daily_returns) > 0:
                daily_std = daily_returns.std()

                if daily_std > 0 and not np.isnan(daily_std) and not np.isinf(daily_std):
                    excess_return = daily_returns.mean() * 252 - self.risk_free_rate
                    volatility = daily_std * np.sqrt(252)

                    if volatility > 0 and not np.isnan(volatility) and not np.isinf(volatility):
                        self.metrics['sharpe_ratio'] = excess_return / volatility
                    else:
                        self.metrics['sharpe_ratio'] = 0
                else:
                    self.metrics['sharpe_ratio'] = 0
            else:
                self.metrics['sharpe_ratio'] = 0

            # 交易统计
            self.metrics['trade_count'] = self._calculate_trade_count()
            self.metrics['win_rate'], self.metrics['profit_loss_ratio'], \
                self.metrics['avg_profit'], self.metrics['avg_loss'] = self._calculate_trade_stats()

            # 确保所有指标都是有效的数值
            for key in self.metrics:
                value = self.metrics[key]
                if pd.isna(value) or np.isinf(value):
                    self.metrics[key] = 0

        except Exception as e:
            print(f"计算策略指标时出错: {e}")
            # 设置默认值
            self.metrics = {
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'trade_count': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'avg_profit': 0,
                'avg_loss': 0
            }

        return self

    def _calculate_trade_count(self):
        """计算交易次数"""
        trades = self.data[self.data['signal'] != '']
        return len(trades) // 2

    def _calculate_trade_stats(self):
        """计算交易胜率等相关统计"""
        try:
            trades = self.data[self.data['signal'] != ''][
                ['日期', 'signal', '收盘', 'position', 'commission', 'stamp_tax']].copy()

            if trades.empty:
                return 0, 0, 0, 1

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

                win_rate = win_count / len(profits) if len(profits) > 0 else 0

                total_profit = sum(p for p in profits if p > 0)
                total_loss = sum(-p for p in profits if p < 0)

                profit_loss_ratio = total_profit / total_loss if total_loss > 0 else float('inf')

                avg_profit = total_profit / win_count if win_count > 0 else 0
                avg_loss = total_loss / loss_count if loss_count > 0 else 1

                # 处理可能的NaN或无穷大
                if pd.isna(win_rate) or np.isinf(win_rate):
                    win_rate = 0
                if pd.isna(profit_loss_ratio) or np.isinf(profit_loss_ratio):
                    profit_loss_ratio = 0
                if pd.isna(avg_profit) or np.isinf(avg_profit):
                    avg_profit = 0
                if pd.isna(avg_loss) or np.isinf(avg_loss):
                    avg_loss = 1

                return win_rate, profit_loss_ratio, avg_profit, avg_loss

            return 0, 0, 0, 1

        except Exception as e:
            print(f"计算交易统计时出错: {e}")
            return 0, 0, 0, 1

    def calculate_buyhold_metrics(self):
        """计算买入持有策略指标（用于对比）"""
        if self.data is None:
            raise ValueError("请先加载数据")

        try:
            initial_price = self.data['收盘'].iloc[0]

            # 避免除零
            if initial_price == 0:
                print("警告: 初始价格为0，无法计算买入持有指标")
                self.buyhold_metrics = {
                    'total_return': 0,
                    'annual_return': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                }
                return self

            buy_hold_shares = self.initial_capital / initial_price

            # 每日资产
            self.data['buy_hold_asset'] = buy_hold_shares * self.data['收盘']
            self.data['buy_hold_return'] = self.data['buy_hold_asset'].pct_change()

            # 最终资产
            final_buyhold_asset = self.data['buy_hold_asset'].iloc[-1]
            trading_days = len(self.data)

            # 基础指标
            self.buyhold_metrics['total_return'] = final_buyhold_asset / self.initial_capital - 1

            # 安全计算年化收益率
            if self.buyhold_metrics['total_return'] > -1:  # 确保总收益率大于-100%
                try:
                    self.buyhold_metrics['annual_return'] = (1 + self.buyhold_metrics['total_return']) ** (
                                252 / trading_days) - 1
                except (ValueError, ZeroDivisionError) as e:
                    print(f"警告: 年化收益率计算失败 - {e}")
                    self.buyhold_metrics['annual_return'] = 0
            else:
                self.buyhold_metrics['annual_return'] = -1  # 全部亏损的情况

            # 最大回撤
            self.data['buyhold_cum_max'] = self.data['buy_hold_asset'].cummax()

            # 避免除零
            mask = self.data['buyhold_cum_max'] > 0
            self.data['buyhold_drawdown'] = 0.0
            self.data.loc[mask, 'buyhold_drawdown'] = 1 - self.data.loc[mask, 'buy_hold_asset'] / self.data.loc[
                mask, 'buyhold_cum_max']

            self.buyhold_metrics['max_drawdown'] = self.data['buyhold_drawdown'].max()

            # 夏普比率
            buyhold_returns = self.data['buy_hold_return'].dropna()

            if len(buyhold_returns) > 0:
                buyhold_mean_return = buyhold_returns.mean()
                buyhold_std_return = buyhold_returns.std()

                # 避免除零和无效值
                if buyhold_std_return > 0 and not np.isnan(buyhold_std_return) and not np.isinf(buyhold_std_return):
                    buyhold_excess_return = buyhold_mean_return * 252 - self.risk_free_rate
                    buyhold_volatility = buyhold_std_return * np.sqrt(252)

                    if buyhold_volatility > 0 and not np.isnan(buyhold_volatility) and not np.isinf(buyhold_volatility):
                        self.buyhold_metrics['sharpe_ratio'] = buyhold_excess_return / buyhold_volatility
                    else:
                        self.buyhold_metrics['sharpe_ratio'] = 0
                else:
                    self.buyhold_metrics['sharpe_ratio'] = 0
            else:
                self.buyhold_metrics['sharpe_ratio'] = 0

            # 确保所有指标都是有效的数值
            for key in ['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio']:
                if key in self.buyhold_metrics:
                    value = self.buyhold_metrics[key]
                    # 处理NaN或无穷大
                    if pd.isna(value) or np.isinf(value):
                        self.buyhold_metrics[key] = 0

        except Exception as e:
            print(f"计算买入持有指标时出错: {e}")
            # 设置默认值
            self.buyhold_metrics = {
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }

        return self

    def print_results(self, show_stock_code=True):
        """打印回测结果对比"""
        if not self.metrics or not self.buyhold_metrics:
            raise ValueError("请先计算指标")

        title = f"双均线策略回测结果 (MA{self.short_ma} vs MA{self.long_ma})"
        if show_stock_code and hasattr(self, 'stock_code'):
            title = f"股票 {self.stock_code} - {title}"

        print("=" * 70)
        print(title)
        print("=" * 70)
        print(f"{'指标':<20} {'双均线策略':>22} {'买入持有':>22}")
        print("-" * 70)
        print(f"{'总收益率':<20} {self.metrics['total_return']:>22.2%} "
              f"{self.buyhold_metrics['total_return']:>22.2%}")
        print(f"{'年化收益率':<20} {self.metrics['annual_return']:>22.2%} "
              f"{self.buyhold_metrics['annual_return']:>22.2%}")
        print(f"{'最大回撤':<20} {self.metrics['max_drawdown']:>22.2%} "
              f"{self.buyhold_metrics['max_drawdown']:>22.2%}")
        print(f"{'夏普比率':<20} {self.metrics['sharpe_ratio']:>22.2f} "
              f"{self.buyhold_metrics['sharpe_ratio']:>22.2f}")
        print(f"{'交易次数':<20} {self.metrics.get('trade_count', 0):>22} {'1次买入':>22}")
        print(f"{'胜率':<20} {self.metrics.get('win_rate', 0):>22.2%} {'100%':>22}")
        print("=" * 70)

        # 超额收益
        alpha = self.metrics['total_return'] - self.buyhold_metrics['total_return']
        print(f"\n超额收益（相对买入持有）: {alpha:.2%}")

    def plot_results(self, figsize=(14, 16)):
        """
        绘制回测结果图表

        参数:
        figsize: 图表尺寸 (默认: (14, 16))
        """
        if self.data is None:
            raise ValueError("请先运行回测")

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建4个子图
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

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
        ax1.scatter(buy_points['日期'], buy_points['收盘'], marker='^', color='red', s=100, label='买入', zorder=5)
        ax1.scatter(sell_points['日期'], sell_points['收盘'], marker='v', color='green', s=100, label='卖出', zorder=5)

        ax1.set_title(f'价格走势与交易信号 (MA{self.short_ma} vs MA{self.long_ma})', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.set_ylabel('价格')

        # 子图2：持仓状态
        ax2 = axes[1]
        ax2.fill_between(self.data['日期'], 0, self.data['position'] > 0, alpha=0.3, color='blue', label='持仓')
        ax2.set_title('持仓状态', fontsize=12)
        ax2.set_ylabel('持仓(1=有,0=无)')
        ax2.set_ylim(-0.1, 1.1)

        # 子图3：策略净值 vs 买入持有净值（对数坐标）
        ax3 = axes[2]

        # 归一化到1起点
        strategy_nav = self.data['total_asset'] / self.initial_capital
        buyhold_nav = self.data['buy_hold_asset'] / self.initial_capital

        ax3.semilogy(self.data['日期'], strategy_nav, label='策略净值', color='red', linewidth=1.5)
        ax3.semilogy(self.data['日期'], buyhold_nav, label='买入持有净值', color='gray', linewidth=1.5, alpha=0.7)
        ax3.set_title('策略净值 vs 买入持有净值（对数坐标）', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.set_ylabel('净值')

        # 子图4：回撤曲线
        ax4 = axes[3]
        ax4.fill_between(self.data['日期'], 0, self.data['drawdown'], alpha=0.5, color='red', label='回撤')
        ax4.axhline(y=self.metrics['max_drawdown'], color='darkred', linestyle='--',
                    label=f"最大回撤 {self.metrics['max_drawdown']:.2%}")
        ax4.set_title('回撤曲线', fontsize=12)
        ax4.set_ylabel('回撤幅度')
        ax4.set_xlabel('日期')
        ax4.legend(loc='lower left')

        plt.tight_layout()
        plt.show()

        return fig

    def run_complete_analysis(self, filepath, encoding='gbk'):
        """
        运行完整的分析流程

        参数:
        filepath: 数据文件路径
        encoding: 文件编码
        """
        return (self.load_data(filepath, encoding)
                .preprocess_data()
                .generate_signals()
                .run_backtest()
                .calculate_metrics()
                .calculate_buyhold_metrics())

    def get_results(self):
        """获取回测结果数据框"""
        return self.data

    def get_metrics(self):
        """获取策略指标"""
        return self.metrics

    def get_buyhold_metrics(self):
        """获取买入持有指标"""
        return self.buyhold_metrics

    @staticmethod
    def compare_strategies(filepath, ma_pairs, encoding='gbk'):
        """
        静态方法：比较多组均线参数的表现

        参数:
        filepath: 数据文件路径
        ma_pairs: 均线参数对列表，如 [(5,20), (10,60), (20,120)]
        encoding: 文件编码
        """
        results = []

        for short_ma, long_ma in ma_pairs:
            print(f"\n测试均线组合: MA{short_ma} vs MA{long_ma}")
            print("-" * 40)

            strategy = DoubleMovingAverageStrategy(short_ma=short_ma, long_ma=long_ma)
            strategy.run_complete_analysis(filepath, encoding)

            results.append({
                'short_ma': short_ma,
                'long_ma': long_ma,
                'total_return': strategy.metrics['total_return'],
                'annual_return': strategy.metrics['annual_return'],
                'max_drawdown': strategy.metrics['max_drawdown'],
                'sharpe_ratio': strategy.metrics['sharpe_ratio'],
                'trade_count': strategy.metrics['trade_count'],
                'win_rate': strategy.metrics['win_rate']
            })

        # 创建结果DataFrame
        results_df = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("多组均线策略对比")
        print("=" * 80)
        print(results_df.to_string(index=False))

        return results_df

    @staticmethod
    def compare_stocks(file_list, short_ma=10, long_ma=120, encoding='gbk',
                       initial_capital=100000, commission_rate=0.0001,
                       min_commission=5, stamp_tax_rate=0.001, risk_free_rate=0.02):
        """
        比较同一策略在不同股票上的表现

        参数:
        file_list: 股票数据文件路径列表，可以是列表或文件夹路径
        short_ma: 短期均线周期
        long_ma: 长期均线周期
        encoding: 文件编码
        initial_capital: 初始资金
        commission_rate: 手续费率
        min_commission: 最低手续费
        stamp_tax_rate: 印花税率
        risk_free_rate: 无风险利率

        返回:
        DataFrame: 各股票的策略表现对比
        """
        results = []

        # 如果传入的是文件夹路径，获取所有csv文件
        if isinstance(file_list, str) and os.path.isdir(file_list):
            file_list = list(Path(file_list).glob("*.csv"))

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
                    risk_free_rate=risk_free_rate
                )

                # 运行分析
                strategy.run_complete_analysis(filepath, encoding)

                # 安全地获取指标，处理可能的无效值
                def safe_get(metrics_dict, key, default=0):
                    val = metrics_dict.get(key, default)
                    # 处理NaN和无穷大
                    if pd.isna(val) or np.isinf(val):
                        return default
                    return val

                # 收集结果（包含策略和买入持有指标）
                results.append({
                    # 基本信息
                    '股票代码': strategy.stock_code,
                    '文件': os.path.basename(filepath),
                    '数据条数': len(strategy.data) if strategy.data is not None else 0,
                    '开始日期': strategy.data['日期'].iloc[-1] if strategy.data is not None and len(
                        strategy.data) > 0 else 'N/A',
                    '结束日期': strategy.data['日期'].iloc[0] if strategy.data is not None and len(
                        strategy.data) > 0 else 'N/A',

                    # 策略指标
                    '策略_总收益率': safe_get(strategy.metrics, 'total_return'),
                    '策略_年化收益率': safe_get(strategy.metrics, 'annual_return'),
                    '策略_最大回撤': safe_get(strategy.metrics, 'max_drawdown'),
                    '策略_夏普比率': safe_get(strategy.metrics, 'sharpe_ratio'),
                    '策略_交易次数': safe_get(strategy.metrics, 'trade_count', 0),
                    '策略_胜率': safe_get(strategy.metrics, 'win_rate'),
                    '策略_盈亏比': safe_get(strategy.metrics, 'profit_loss_ratio', 1),

                    # 买入持有指标
                    '持有_总收益率': safe_get(strategy.buyhold_metrics, 'total_return'),
                    '持有_年化收益率': safe_get(strategy.buyhold_metrics, 'annual_return'),
                    '持有_最大回撤': safe_get(strategy.buyhold_metrics, 'max_drawdown'),
                    '持有_夏普比率': safe_get(strategy.buyhold_metrics, 'sharpe_ratio'),

                    # 对比指标
                    '超额收益': safe_get(strategy.metrics, 'total_return') - safe_get(strategy.buyhold_metrics,
                                                                                      'total_return'),
                    '胜率优势': safe_get(strategy.metrics, 'win_rate') - 0.5,
                    '回撤改善': safe_get(strategy.buyhold_metrics, 'max_drawdown') - safe_get(strategy.metrics,
                                                                                              'max_drawdown'),
                    '夏普提升': safe_get(strategy.metrics, 'sharpe_ratio') - safe_get(strategy.buyhold_metrics,
                                                                                      'sharpe_ratio')
                })

            except Exception as e:
                print(f"分析 {filepath} 时出错: {e}")
                # 记录失败的股票
                try:
                    stock_code = Path(filepath).stem.replace('df_pre_', '')
                except:
                    stock_code = '未知'

                results.append({
                    '股票代码': stock_code,
                    '文件': os.path.basename(filepath),
                    '数据条数': 0,
                    '开始日期': 'N/A',
                    '结束日期': 'N/A',
                    '策略_总收益率': 0,
                    '策略_年化收益率': 0,
                    '策略_最大回撤': 0,
                    '策略_夏普比率': 0,
                    '策略_交易次数': 0,
                    '策略_胜率': 0,
                    '策略_盈亏比': 0,
                    '持有_总收益率': 0,
                    '持有_年化收益率': 0,
                    '持有_最大回撤': 0,
                    '持有_夏普比率': 0,
                    '超额收益': 0,
                    '胜率优势': -0.5,
                    '回撤改善': 0,
                    '夏普提升': 0
                })
                continue

        # 创建结果DataFrame
        results_df = pd.DataFrame(results)

        # 过滤掉无效数据再排序
        valid_df = results_df[results_df['策略_夏普比率'].notna() &
                              ~np.isinf(results_df['策略_夏普比率'])].copy()

        if not valid_df.empty:
            valid_df = valid_df.sort_values('策略_夏普比率', ascending=False)
            # 将有效数据和无效数据合并
            invalid_df = results_df[~results_df.index.isin(valid_df.index)]
            results_df = pd.concat([valid_df, invalid_df], ignore_index=True)

        # 导出结果
        output_file = "./compare_stocks_results.csv"
        results_df.to_csv(output_file, encoding='gbk', index=False)
        print(f"\n对比分析结果已导出为 {output_file} 文件！")
        print(f"共分析 {len(results_df)} 只股票，成功 {len(results_df[results_df['数据条数'] > 0])} 只")

        return results_df

    @staticmethod
    def print_comparison_summary(results_df):
        """
        打印对比结果的摘要（兼容两种格式：多股票比较和多参数比较）
        """
        if results_df.empty:
            print("没有数据可显示")
            return

        # 判断DataFrame的类型
        if '数据条数' in results_df.columns:
            # 这是多股票比较的结果（带"策略_"前缀）
            success_df = results_df[results_df['数据条数'] > 0].copy()
            title = "多股票策略对比摘要"

            # 多股票比较的列名带有"策略_"前缀
            total_return_col = '策略_总收益率'
            annual_return_col = '策略_年化收益率'
            max_drawdown_col = '策略_最大回撤'
            sharpe_ratio_col = '策略_夏普比率'
            win_rate_col = '策略_胜率'

            # 标识列
            id_cols = ['股票代码']

        else:
            # 这是多参数比较的结果（不带前缀）
            success_df = results_df.copy()
            title = "多均线参数对比摘要"

            # 多参数比较的列名不带前缀
            total_return_col = 'total_return'
            annual_return_col = 'annual_return'
            max_drawdown_col = 'max_drawdown'
            sharpe_ratio_col = 'sharpe_ratio'
            win_rate_col = 'win_rate' if 'win_rate' in results_df.columns else None

            # 标识列
            id_cols = ['short_ma', 'long_ma']

        if success_df.empty:
            print("没有有效的分析结果")
            return

        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        # 整体统计
        print(f"\n分析总数: {len(success_df)}")

        # 策略表现统计
        print("\n--- 策略表现统计 ---")
        print(f"平均总收益率: {success_df[total_return_col].mean():.2%}")
        print(f"平均年化收益率: {success_df[annual_return_col].mean():.2%}")
        print(f"平均最大回撤: {success_df[max_drawdown_col].mean():.2%}")
        print(f"平均夏普比率: {success_df[sharpe_ratio_col].mean():.2f}")

        if win_rate_col and win_rate_col in success_df.columns:
            print(f"平均胜率: {success_df[win_rate_col].mean():.2%}")

        # 最佳表现
        print("\n--- 最佳表现 (按夏普比率) ---")
        best_sharpe = success_df.nlargest(3, sharpe_ratio_col)

        if len(id_cols) == 2:  # 多参数比较
            best_display = best_sharpe[
                ['short_ma', 'long_ma', sharpe_ratio_col, total_return_col, max_drawdown_col]].copy()
            best_display['均线组合'] = 'MA' + best_display['short_ma'].astype(str) + '_MA' + best_display[
                'long_ma'].astype(str)
            best_display = best_display[['均线组合', sharpe_ratio_col, total_return_col, max_drawdown_col]]
        else:  # 多股票比较
            best_display = best_sharpe[['股票代码', sharpe_ratio_col, total_return_col, max_drawdown_col]]

        # 重命名列以便显示
        best_display = best_display.rename(columns={
            sharpe_ratio_col: '夏普比率',
            total_return_col: '总收益率',
            max_drawdown_col: '最大回撤'
        })
        print(best_display.to_string(index=False))

        # 最差表现
        print("\n--- 最差表现 (按夏普比率) ---")
        worst_sharpe = success_df.nsmallest(3, sharpe_ratio_col)

        if len(id_cols) == 2:  # 多参数比较
            worst_display = worst_sharpe[
                ['short_ma', 'long_ma', sharpe_ratio_col, total_return_col, max_drawdown_col]].copy()
            worst_display['均线组合'] = 'MA' + worst_display['short_ma'].astype(str) + '_MA' + worst_display[
                'long_ma'].astype(str)
            worst_display = worst_display[['均线组合', sharpe_ratio_col, total_return_col, max_drawdown_col]]
        else:  # 多股票比较
            worst_display = worst_sharpe[['股票代码', sharpe_ratio_col, total_return_col, max_drawdown_col]]

        # 重命名列以便显示
        worst_display = worst_display.rename(columns={
            sharpe_ratio_col: '夏普比率',
            total_return_col: '总收益率',
            max_drawdown_col: '最大回撤'
        })
        print(worst_display.to_string(index=False))

    @staticmethod
    def compare_strategies_and_stocks(file_list, ma_pairs, encoding='gbk',
                                      initial_capital=100000):
        """
        同时比较不同均线参数在不同股票上的表现

        参数:
        file_list: 股票数据文件路径列表
        ma_pairs: 均线参数对列表，如 [(5,20), (10,60), (20,120)]
        encoding: 文件编码
        initial_capital: 初始资金

        返回:
        DataFrame: 多维比较结果
        """
        all_results = []

        for filepath in file_list:
            filename = os.path.basename(filepath)
            # 提取股票代码
            import re
            match = re.search(r'(\d{6})', filename)
            stock_code = match.group(1) if match else filename

            for short_ma, long_ma in ma_pairs:
                try:
                    print(f"\n测试: 股票 {stock_code}, 均线 MA{short_ma} vs MA{long_ma}")

                    strategy = DoubleMovingAverageStrategy(
                        short_ma=short_ma,
                        long_ma=long_ma,
                        initial_capital=initial_capital
                    )

                    strategy.run_complete_analysis(filepath, encoding)

                    all_results.append({
                        '股票代码': stock_code,
                        '短期均线': short_ma,
                        '长期均线': long_ma,
                        '总收益率': strategy.metrics['total_return'],
                        '年化收益率': strategy.metrics['annual_return'],
                        '最大回撤': strategy.metrics['max_drawdown'],
                        '夏普比率': strategy.metrics['sharpe_ratio'],
                        '交易次数': strategy.metrics['trade_count'],
                        '胜率': strategy.metrics['win_rate']
                    })

                except Exception as e:
                    print(f"测试失败: {e}")
                    continue

        # 创建结果DataFrame
        results_df = pd.DataFrame(all_results)

        return results_df

    @staticmethod
    def plot_stock_comparison(results_df, short_ma=None, long_ma=None, top_n=20, figsize=(16, 12)):
        """
        增强版多股票比较图表 - 快速判断策略有效性

        参数:
        results_df: compare_stocks 方法返回的DataFrame
        short_ma: 短期均线周期
        long_ma: 长期均线周期
        top_n: 显示前N只股票（按夏普比率排序）
        figsize: 图表尺寸
        """
        if results_df.empty:
            print("没有可绘制的数据")
            return

        # 只分析成功的数据
        plot_df = results_df[results_df['数据条数'] > 0].copy()

        if plot_df.empty:
            print("没有成功的数据可绘制")
            return

        # 处理无效值
        for col in plot_df.columns:
            if col not in ['股票代码', '文件', '开始日期', '结束日期']:
                plot_df[col] = plot_df[col].apply(lambda x: x if pd.notna(x) and not np.isinf(x) else 0)

        # 按夏普比率排序，取前top_n只股票
        plot_df = plot_df.sort_values('策略_夏普比率', ascending=False).head(top_n).copy()

        print(f"\n显示前 {len(plot_df)} 只股票 (按策略夏普比率排序)")

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # ===== 图1: 最大回撤分布直方图 =====
        ax1 = axes[0, 0]

        # 统计超出范围的数据
        exceed_count_1 = (plot_df['策略_最大回撤'] > 1.5).sum() + (plot_df['持有_最大回撤'] > 1.5).sum()

        # 过滤数据，只保留在范围内的
        plot_df_1 = plot_df[(plot_df['策略_最大回撤'] <= 1.5) & (plot_df['持有_最大回撤'] <= 1.5)]

        # 使用直方图
        bins = 30
        ax1.hist(plot_df_1['策略_最大回撤'], bins=bins, alpha=0.5, label='策略回撤', color='red', edgecolor='black',
                 density=True)
        ax1.hist(plot_df_1['持有_最大回撤'], bins=bins, alpha=0.5, label='买入持有回撤', color='blue',
                 edgecolor='black', density=True)

        # 添加均值线
        ax1.axvline(x=plot_df_1['策略_最大回撤'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f'策略平均: {plot_df_1["策略_最大回撤"].mean():.2%}')
        ax1.axvline(x=plot_df_1['持有_最大回撤'].mean(), color='blue', linestyle='--', linewidth=2,
                    label=f'持有平均: {plot_df_1["持有_最大回撤"].mean():.2%}')

        ax1.set_xlabel('最大回撤')
        ax1.set_ylabel('个股数量')
        ax1.set_title(f'最大回撤分布对比 (Top {len(plot_df)}只)\n(左移说明策略降低了回撤)')
        ax1.set_xlim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加超出范围的备注
        if exceed_count_1 > 0:
            ax1.text(0.98, 0.98, f'超出范围1.5的数据: {exceed_count_1}个',
                     transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 计算并显示回撤改善的股票比例
        improved_count = (plot_df_1['策略_最大回撤'] < plot_df_1['持有_最大回撤']).sum()
        total_count = len(plot_df_1)
        ax1.text(0.02, 0.98, f'回撤改善比例: {improved_count / total_count:.1%}',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ===== 图2: 胜率vs盈亏比四象限图 =====
        ax2 = axes[0, 1]

        # 处理盈亏比无限大的情况
        plot_df['策略_盈亏比_clean'] = plot_df['策略_盈亏比'].replace([np.inf, -np.inf], 0)

        # 绘制散点（点变小）
        scatter = ax2.scatter(plot_df['策略_胜率'], plot_df['策略_盈亏比_clean'],
                              c=plot_df['策略_夏普比率'], cmap='RdYlGn', s=30, alpha=0.6)

        # 绘制象限分割线
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='盈亏比=1')
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='胜率=50%')

        # 添加象限说明
        ax2.text(0.75, plot_df['策略_盈亏比_clean'].max() * 0.8, '最优区域\n(高胜率+高盈亏比)', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax2.text(0.25, plot_df['策略_盈亏比_clean'].max() * 0.8, '趋势策略区域\n(低胜率+高盈亏比)', ha='center',
                 fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax2.text(0.75, 0.3, '震荡策略区域\n(高胜率+低盈亏比)', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        ax2.text(0.25, 0.3, '无效策略区域\n(双低)', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

        ax2.set_xlabel('胜率')
        ax2.set_ylabel('盈亏比')
        ax2.set_title(f'胜率vs盈亏比四象限图 (Top {len(plot_df)}只)\n(颜色=夏普比率)')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, plot_df['策略_盈亏比_clean'].max() * 1.1)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2)

        # 统计各象限股票数量
        q1 = ((plot_df['策略_胜率'] >= 0.5) & (plot_df['策略_盈亏比_clean'] >= 1)).sum()
        q2 = ((plot_df['策略_胜率'] < 0.5) & (plot_df['策略_盈亏比_clean'] >= 1)).sum()
        q3 = ((plot_df['策略_胜率'] >= 0.5) & (plot_df['策略_盈亏比_clean'] < 1)).sum()
        q4 = ((plot_df['策略_胜率'] < 0.5) & (plot_df['策略_盈亏比_clean'] < 1)).sum()

        ax2.text(0.02, 0.98, f'最优: {q1} | 趋势: {q2} | 震荡: {q3} | 无效: {q4}',
                 transform=ax2.transAxes, verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ===== 图3: 策略vs持有收益率散点图+45度线 =====
        ax3 = axes[1, 0]

        # 设置坐标轴范围
        x_min, x_max = -2.5, 13
        y_min, y_max = -2, 6

        # 统计超出范围的数据
        exceed_x = ((plot_df['持有_总收益率'] < x_min) | (plot_df['持有_总收益率'] > x_max)).sum()
        exceed_y = ((plot_df['策略_总收益率'] < y_min) | (plot_df['策略_总收益率'] > y_max)).sum()

        # 过滤数据，只保留在范围内的
        plot_df_3 = plot_df[
            (plot_df['持有_总收益率'] >= x_min) &
            (plot_df['持有_总收益率'] <= x_max) &
            (plot_df['策略_总收益率'] >= y_min) &
            (plot_df['策略_总收益率'] <= y_max)
            ].copy()

        # 绘制45度线
        ax3.plot([x_min, x_max], [x_min, x_max], 'k--', alpha=0.5, label='45度线(策略=持有)')

        # 绘制散点，颜色表示超额收益大小（点变小）
        if not plot_df_3.empty:
            scatter3 = ax3.scatter(plot_df_3['持有_总收益率'], plot_df_3['策略_总收益率'],
                                   c=plot_df_3['超额收益'], cmap='RdYlGn', s=30, alpha=0.6)
            plt.colorbar(scatter3, ax=ax3, label='超额收益')

        # 添加上下半区说明
        ax3.text(x_max * 0.8, y_max * 0.9, '策略跑赢', ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax3.text(x_max * 0.2, y_min * 0.8, '策略跑输', ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

        ax3.set_xlabel('买入持有收益率')
        ax3.set_ylabel('策略收益率')
        ax3.set_title(f'策略vs持有收益率散点图 (Top {len(plot_df)}只)\n(点位于线上方说明策略跑赢基准)')
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.grid(True, alpha=0.3)

        # 添加超出范围的备注
        exceed_msg = []
        if exceed_x > 0:
            exceed_msg.append(f'X轴超出: {exceed_x}个')
        if exceed_y > 0:
            exceed_msg.append(f'Y轴超出: {exceed_y}个')
        if exceed_msg:
            ax3.text(0.98, 0.98, '\n'.join(exceed_msg),
                     transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 统计跑赢的股票数量（基于过滤后的数据）
        if not plot_df_3.empty:
            beat_count = (plot_df_3['策略_总收益率'] > plot_df_3['持有_总收益率']).sum()
            ax3.text(0.02, 0.98, f'跑赢基准: {beat_count}/{len(plot_df_3)} ({beat_count / len(plot_df_3):.1%})',
                     transform=ax3.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # ===== 图4: 超额收益vs回撤改善四象限图 =====
        ax4 = axes[1, 1]

        # 设置纵轴范围
        y_max_4 = 1.5

        # 统计超出范围的数据
        exceed_y_4 = (plot_df['回撤改善'] > y_max_4).sum()

        # 过滤数据，只保留在范围内的
        plot_df_4 = plot_df[plot_df['回撤改善'] <= y_max_4].copy()

        # 绘制散点（点变小）
        if not plot_df_4.empty:
            scatter4 = ax4.scatter(plot_df_4['超额收益'], plot_df_4['回撤改善'],
                                   c=plot_df_4['策略_夏普比率'], cmap='RdYlGn', s=30, alpha=0.6)
            plt.colorbar(scatter4, ax=ax4, label='策略夏普比率')

        # 绘制象限分割线
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # 添加象限说明
        x_max_4 = max(abs(plot_df_4['超额收益'].max()),
                      abs(plot_df_4['超额收益'].min())) * 0.5 if not plot_df_4.empty else 0.5

        ax4.text(x_max_4 * 0.8, y_max_4 * 0.8, '最佳区域\n(正收益+降回撤)', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax4.text(-x_max_4 * 0.8, y_max_4 * 0.8, '降回撤但亏损', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax4.text(x_max_4 * 0.8, -y_max_4 * 0.2, '有收益但增回撤', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        ax4.text(-x_max_4 * 0.8, -y_max_4 * 0.2, '最差区域\n(亏损+增回撤)', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

        ax4.set_xlabel('超额收益')
        ax4.set_ylabel('回撤改善\n(正值表示策略回撤更小)')
        ax4.set_title(f'超额收益vs回撤改善四象限图 (Top {len(plot_df)}只)\n(颜色=策略夏普比率)')
        ax4.set_ylim(-y_max_4 * 0.3, y_max_4)
        ax4.grid(True, alpha=0.3)

        # 添加超出范围的备注
        if exceed_y_4 > 0:
            ax4.text(0.98, 0.98, f'Y轴超出1.5: {exceed_y_4}个',
                     transform=ax4.transAxes, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 统计各象限股票数量（基于过滤后的数据）
        if not plot_df_4.empty:
            q1_4 = ((plot_df_4['超额收益'] > 0) & (plot_df_4['回撤改善'] > 0)).sum()  # 最佳
            q2_4 = ((plot_df_4['超额收益'] < 0) & (plot_df_4['回撤改善'] > 0)).sum()  # 降回撤但亏损
            q3_4 = ((plot_df_4['超额收益'] > 0) & (plot_df_4['回撤改善'] < 0)).sum()  # 有收益但增回撤
            q4_4 = ((plot_df_4['超额收益'] < 0) & (plot_df_4['回撤改善'] < 0)).sum()  # 最差

            ax4.text(0.02, 0.98, f'最佳: {q1_4} | 降回撤亏损: {q2_4} | 有收益增回撤: {q3_4} | 最差: {q4_4}',
                     transform=ax4.transAxes, verticalalignment='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 设置总标题
        if short_ma is not None and long_ma is not None:
            title = f'策略有效性诊断 (均线 MA{short_ma} vs MA{long_ma} | Top {len(plot_df)}只)'
        else:
            title = f'策略有效性诊断 (Top {len(plot_df)}只)'

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        # 输出关键统计信息
        print("\n" + "=" * 60)
        print("策略有效性快速诊断")
        print("=" * 60)
        print(f"分析股票总数: {len(plot_df)} (按夏普比率排序前{top_n}只)")
        print(f"\n【关键指标】")
        print(
            f"跑赢基准比例: {beat_count / len(plot_df_3):.1%}" if 'plot_df_3' in locals() and not plot_df_3.empty else "跑赢基准比例: N/A")
        print(f"回撤改善比例: {improved_count / len(plot_df_1):.1%}")
        print(
            f"策略平均回撤: {plot_df_1['策略_最大回撤'].mean():.2%} vs 持有平均回撤: {plot_df_1['持有_最大回撤'].mean():.2%}")
        print(
            f"策略平均夏普: {plot_df['策略_夏普比率'].mean():.2f} vs 持有平均夏普: {plot_df['持有_夏普比率'].mean():.2f}")
        print(f"\n【四象限统计】")
        print(f"胜率vs盈亏比 - 最优: {q1} | 趋势: {q2} | 震荡: {q3} | 无效: {q4}")
        if 'plot_df_4' in locals() and not plot_df_4.empty:
            print(f"超额vs回撤  - 最佳: {q1_4} | 降回撤亏损: {q2_4} | 有收益增回撤: {q3_4} | 最差: {q4_4}")

        return fig

    @staticmethod
    def plot_strategies_comparison(results_df, filepath, encoding='gbk', figsize=(12, 16)):
        """
        比较多组均线参数策略的表现

        参数:
        results_df: compare_strategies 方法返回的DataFrame
        filepath: 原始数据文件路径（用于获取时间序列和买入持有数据）
        encoding: 文件编码
        figsize: 图表尺寸
        """
        if results_df.empty:
            print("没有可绘制的数据")
            return

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # 调整子图之间的间距，为总标题留出空间
        plt.subplots_adjust(top=0.95, hspace=0.3)

        # ===== 图1: 各参数策略的净值对比图（对数坐标）=====
        ax1 = axes[0]

        # 获取原始数据用于时间轴和买入持有计算
        data = pd.read_csv(filepath, encoding=encoding)

        # 统一处理日期格式
        if '日期' in data.columns:
            try:
                data['日期'] = pd.to_datetime(data['日期'], format='%Y-%m-%d')
            except:
                try:
                    data['日期'] = pd.to_datetime(data['日期'], format='%Y/%m/%d')
                except:
                    data['日期'] = pd.to_datetime(data['日期'])

        data = data.sort_values('日期')

        # 计算买入持有净值
        initial_price = data['收盘'].iloc[0]
        buy_hold_shares = 100000 / initial_price
        data['buy_hold_asset'] = buy_hold_shares * data['收盘']
        data['buy_hold_nav'] = data['buy_hold_asset'] / 100000

        # 绘制买入持有净值
        ax1.semilogy(data['日期'], data['buy_hold_nav'],
                     label='买入持有', color='black', linewidth=2, linestyle='--', alpha=0.7)

        # 为每个参数组合计算并绘制策略净值
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))

        for idx, row in results_df.iterrows():
            short_ma = int(row['short_ma'])
            long_ma = int(row['long_ma'])

            try:
                print(f"计算 MA{short_ma}-MA{long_ma} 净值...")

                strategy = DoubleMovingAverageStrategy(
                    short_ma=short_ma,
                    long_ma=long_ma,
                    initial_capital=100000
                )
                strategy.run_complete_analysis(filepath, encoding)

                if '日期' in strategy.data.columns:
                    try:
                        strategy.data['日期'] = pd.to_datetime(strategy.data['日期'])
                    except:
                        pass

                strategy_nav = strategy.data['total_asset'] / 100000

                ax1.semilogy(strategy.data['日期'], strategy_nav,
                             label=f'MA{short_ma}-MA{long_ma}',
                             color=colors[idx], linewidth=1.5, alpha=0.8)

            except Exception as e:
                print(f"计算 MA{short_ma}-MA{long_ma} 净值时出错: {e}")
                continue

        ax1.set_xlabel('日期')
        ax1.set_ylabel('净值（对数坐标）')
        ax1.set_title('图1: 各参数策略净值对比', pad=15)
        ax1.legend(loc='upper left', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # ===== 图2: 最大回撤vs收益率四象限图 =====
        ax2 = axes[1]

        # 计算买入持有的回撤和收益率
        buyhold_return = data['buy_hold_nav'].iloc[-1] - 1

        # 计算买入持有的最大回撤
        data['buyhold_cummax'] = data['buy_hold_nav'].cummax()
        data['buyhold_drawdown'] = 1 - data['buy_hold_nav'] / data['buyhold_cummax']
        buyhold_max_drawdown = data['buyhold_drawdown'].max()

        # 绘制各参数策略的点
        scatter = ax2.scatter(results_df['max_drawdown'], results_df['total_return'],
                              c=results_df['sharpe_ratio'], cmap='RdYlGn',
                              s=150, alpha=0.7, zorder=4, edgecolors='black', linewidth=0.5)

        # 绘制买入持有点（五角星）
        ax2.scatter(buyhold_max_drawdown, buyhold_return,
                    s=300, marker='*', color='gold', edgecolors='black', linewidth=1.5,
                    label='买入持有策略', zorder=5)

        # 添加策略标签
        for _, row in results_df.iterrows():
            ax2.annotate(f"MA{int(row['short_ma'])}-{int(row['long_ma'])}",
                         (row['max_drawdown'], row['total_return']),
                         fontsize=9, xytext=(5, 5), textcoords='offset points')

        # 绘制中位数参考线（虚线）
        x_median = results_df['max_drawdown'].median()
        y_median = results_df['total_return'].median()

        ax2.axhline(y=y_median, color='gray', linestyle='--', alpha=0.6, linewidth=1.5,
                    label=f'收益率中位数: {y_median:.2%}')
        ax2.axvline(x=x_median, color='gray', linestyle='--', alpha=0.6, linewidth=1.5,
                    label=f'回撤中位数: {x_median:.2%}')

        # 添加象限说明
        x_range = results_df['max_drawdown'].max() - results_df['max_drawdown'].min()
        y_range = results_df['total_return'].max() - results_df['total_return'].min()

        # 第一象限（右上）：高收益低回撤
        ax2.text(x_median + x_range * 0.2, y_median + y_range * 0.2,
                 '最佳区域\n(高收益低回撤)', ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        # 第二象限（左上）：高收益高回撤
        ax2.text(x_median - x_range * 0.2, y_median + y_range * 0.2,
                 '激进区域\n(高收益高回撤)', ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

        # 第三象限（左下）：低收益高回撤
        ax2.text(x_median - x_range * 0.2, y_median - y_range * 0.2,
                 '最差区域\n(低收益高回撤)', ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

        # 第四象限（右下）：低收益低回撤
        ax2.text(x_median + x_range * 0.2, y_median - y_range * 0.2,
                 '保守区域\n(低收益低回撤)', ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        ax2.set_xlabel('最大回撤')
        ax2.set_ylabel('总收益率')
        ax2.set_title('图2: 最大回撤vs收益率四象限图\n(颜色=夏普比率，★=买入持有策略，虚线=中位数参考线)', pad=15)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='夏普比率')
        ax2.legend(loc='upper left', fontsize=9)

        # ===== 图3: 各策略夏普比率和年化收益率对比 =====
        ax3 = axes[2]

        # 创建参数组合标签
        labels = [f"MA{int(row['short_ma'])}-{int(row['long_ma'])}" for _, row in results_df.iterrows()]
        x = np.arange(len(labels))
        width = 0.35

        # 绘制夏普比率柱状图
        norm_sharpe = results_df['sharpe_ratio'] / results_df['sharpe_ratio'].max() if results_df[
                                                                                           'sharpe_ratio'].max() > 0 else \
        results_df['sharpe_ratio']
        bars1 = ax3.bar(x - width / 2, results_df['sharpe_ratio'], width,
                        label='夏普比率', color=plt.cm.RdYlGn(norm_sharpe), alpha=0.8)

        # 绘制年化收益率柱状图
        norm_return = results_df['annual_return'] / results_df['annual_return'].max() if results_df[
                                                                                             'annual_return'].max() > 0 else \
        results_df['annual_return']
        bars2 = ax3.bar(x + width / 2, results_df['annual_return'], width,
                        label='年化收益率', color=plt.cm.Blues(norm_return), alpha=0.8)

        # 添加买入持有参考线
        buyhold_annual = data['buy_hold_nav'].iloc[-1] ** (252 / len(data)) - 1
        ax3.axhline(y=buyhold_annual, color='blue', linestyle='--', linewidth=2,
                    label=f'买入持有年化: {buyhold_annual:.2%}')

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars1, results_df['sharpe_ratio'])):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        for i, (bar, val) in enumerate(zip(bars2, results_df['annual_return'])):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                     f'{val:.1%}', ha='center', va='bottom', fontsize=8)

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_xlabel('均线参数组合')
        ax3.set_ylabel('数值')
        ax3.set_title('图3: 各策略夏普比率与年化收益率对比', pad=15)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')

        # 设置总标题，位置调高避免与图1标题重叠
        plt.suptitle('不同均线参数策略表现对比', fontsize=16, y=0.99)
        plt.tight_layout()
        plt.show()

        return fig

class RSIStrategy:
    """
    RSI策略主类
    包含数据加载、策略计算、回测、绩效分析、报告生成等功能
    """

    class SignalType(Enum):
            """信号类型枚举"""
            BUY = "买入"
            SELL = "卖出"
            NONE = ""

    @dataclass
    class TradeRecord:
            """交易记录数据类"""
            日期: str
            类型: str
            价格: float
            数量: int
            金额: float
            佣金: float
            印花税: float
            总费用: float
            现金_后: float
            持仓_后: int

    @dataclass
    class PendingSignal:
            """待处理信号数据类"""
            type: 'RSIStrategy.SignalType'
            original_date: str

    @dataclass
    class BacktestResult:
            """回测结果数据类"""
            code: str  # 股票代码
            params: Dict  # 策略参数
            initial_capital: float
            final_value: float
            total_return: float
            annual_return: float
            max_drawdown: float
            sharpe_ratio: float
            win_rate: float
            profit_loss_ratio: float
            total_trades: int
            total_fees: float
            monthly_win: int
            monthly_loss: int
            monthly_avg_return: float
            start_date: str
            end_date: str
            trading_days: int
            daily_df: pd.DataFrame = None
            trade_records: List['RSIStrategy.TradeRecord'] = field(default_factory=list)

    def __init__(self,
                 name: str = "RSI策略",
                 initial_capital: float = 1000000,
                 commission_rate: float = 0.0001,
                 min_commission: float = 5,
                 stamp_tax_rate: float = 0.001,
                 buy_threshold: int = 30,
                 sell_threshold: int = 70,
                 rsi_period: int = 14,
                 min_interval_days: int = 5,
                 output_dir: str = "./output"):
        """
        初始化RSI策略

        Args:
            name: 策略名称
            initial_capital: 初始资金
            commission_rate: 佣金费率
            min_commission: 最低佣金
            stamp_tax_rate: 印花税率
            buy_threshold: 买入阈值
            sell_threshold: 卖出阈值
            rsi_period: RSI计算周期
            min_interval_days: 最小交易间隔（天）
            output_dir: 输出目录
        """
        self.name = name
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.rsi_period = rsi_period
        self.min_interval_days = min_interval_days
        self.output_dir = Path(output_dir)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据相关
        self.df = None
        self.stock_code = None
        self.stock_name = None

        # 回测相关
        self.backtest_results = []
        self.current_result = None

    def load_data(self, file_path: str, stock_code: str = None, stock_name: str = None) -> pd.DataFrame:
            """
            加载数据

            Args:
                file_path: 数据文件路径
                stock_code: 股票代码
                stock_name: 股票名称

            Returns:
                加载的数据框
            """
            self.df = pd.read_csv(file_path)
            self.stock_code = stock_code or Path(file_path).stem.split('_')[2] if '_' in Path(file_path).stem else '未知'
            self.stock_name = stock_name or self.stock_code

            # 确保日期格式正确
            if '日期' in self.df.columns:
                self.df['日期'] = pd.to_datetime(self.df['日期'])

            return self.df

    def load_data_from_df(self, df: pd.DataFrame, stock_code: str = "自定义",
                              stock_name: str = "自定义") -> pd.DataFrame:
            """
            从DataFrame加载数据

            Args:
                df: 数据框，必须包含'日期'和'收盘'列
                stock_code: 股票代码
                stock_name: 股票名称

            Returns:
                数据框
            """
            required_cols = ['日期', '收盘']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"数据框必须包含'{col}'列")

            self.df = df.copy()
            self.stock_code = stock_code
            self.stock_name = stock_name

            # 确保日期格式正确
            if '日期' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['日期']):
                self.df['日期'] = pd.to_datetime(self.df['日期'])

            return self.df

    def preprocess_data(self) -> pd.DataFrame:
            """
            数据预处理：标记异常、停牌等

            Returns:
                预处理后的数据框
            """
            if self.df is None:
                raise ValueError("请先加载数据")

            # 检查是否有异常情况列，如果没有则创建
            if '异常情况' not in self.df.columns:
                self.df['异常情况'] = '正常'

            # 标记停牌和价格异常
            self.df['是停牌'] = self.df['异常情况'] == '停牌'
            self.df['是价格异常'] = self.df['异常情况'] == '价格异常'

            return self.df

    def calculate_rsi(self, period: int = None) -> pd.DataFrame:
            """
            计算RSI指标

            Args:
                period: RSI计算周期，默认使用初始化时的值

            Returns:
                包含RSI的数据框
            """
            if self.df is None:
                raise ValueError("请先加载数据")

            period = period or self.rsi_period

            # 创建仅包含正常交易日的序列
            normal_days = self.df[~self.df['是停牌']].copy()
            normal_days = normal_days.reset_index(drop=True)

            if len(normal_days) == 0:
                raise ValueError("没有正常交易日数据")

            # 计算价格变化
            normal_days['价格变化'] = normal_days['收盘'].diff()
            normal_days['涨幅'] = normal_days['价格变化'].apply(lambda x: x if x > 0 else 0)
            normal_days['跌幅'] = normal_days['价格变化'].apply(lambda x: abs(x) if x < 0 else 0)

            # 计算平均涨幅和跌幅
            normal_days['平均涨幅'] = normal_days['涨幅'].rolling(window=period).mean().shift(1)
            normal_days['平均跌幅'] = normal_days['跌幅'].rolling(window=period).mean().shift(1)

            # 计算RSI
            normal_days['RSI'] = 100 * normal_days['平均涨幅'] / (
                    normal_days['平均涨幅'] + normal_days['平均跌幅'])

            # 将RSI值映射回原DataFrame
            self.df['RSI'] = np.nan
            for _, row in normal_days.iterrows():
                self.df.loc[self.df['日期'] == row['日期'], 'RSI'] = row['RSI']

            # 向前填充RSI值
            self.df['RSI'] = self.df['RSI'].ffill()

            return self.df

    def generate_signals(self) -> pd.DataFrame:
            """
            根据RSI生成交易信号

            Returns:
                包含交易信号的数据框
            """
            if self.df is None or 'RSI' not in self.df.columns:
                raise ValueError("请先计算RSI指标")

            # 判断RSI的位置状态
            self.df['RSI_上穿'] = (self.df['RSI'] >= self.buy_threshold) & (
                        self.df['RSI'].shift(1) < self.buy_threshold)
            self.df['RSI_下穿'] = (self.df['RSI'] <= self.sell_threshold) & (
                        self.df['RSI'].shift(1) > self.sell_threshold)

            # 初始化信号列
            self.df['交易信号'] = ''

            # 状态变量
            last_signal_date = None
            last_signal_type = None
            pending_signal = None

            for i in range(len(self.df)):
                current_date = self.df.iloc[i]['日期']

                # 如果是停牌日，直接跳过
                if self.df.iloc[i]['是停牌']:
                    continue

                # 处理待处理信号
                if pending_signal is not None:
                    if not self.df.iloc[i]['是价格异常']:  # 正常交易日
                        signal_type = pending_signal['type']

                        if signal_type == RSIStrategy.SignalType.BUY:
                            if last_signal_type is None or last_signal_type == RSIStrategy.SignalType.SELL.value:
                                if self._check_interval(current_date, last_signal_date):
                                    self.df.loc[self.df.index[i], '交易信号'] = RSIStrategy.SignalType.BUY.value
                                    last_signal_date = current_date
                                    last_signal_type = RSIStrategy.SignalType.BUY.value

                        elif signal_type == RSIStrategy.SignalType.SELL:
                            if last_signal_type == RSIStrategy.SignalType.BUY.value:
                                if self._check_interval(current_date, last_signal_date):
                                    self.df.loc[self.df.index[i], '交易信号'] = RSIStrategy.SignalType.SELL.value
                                    last_signal_date = current_date
                                    last_signal_type = RSIStrategy.SignalType.SELL.value

                        pending_signal = None
                    # 价格异常日继续等待
                else:
                    # 检查新信号
                    if self.df.iloc[i]['是价格异常']:
                        continue

                    # 买入信号
                    if self.df.iloc[i]['RSI_上穿']:
                        if last_signal_type is None or last_signal_type == RSIStrategy.SignalType.SELL.value:
                            if self._check_interval(current_date, last_signal_date):
                                self.df.loc[self.df.index[i], '交易信号'] = RSIStrategy.SignalType.BUY.value
                                last_signal_date = current_date
                                last_signal_type = RSIStrategy.SignalType.BUY.value

                    # 卖出信号
                    elif self.df.iloc[i]['RSI_下穿']:
                        if last_signal_type == RSIStrategy.SignalType.BUY.value:
                            if self._check_interval(current_date, last_signal_date):
                                self.df.loc[self.df.index[i], '交易信号'] = RSIStrategy.SignalType.SELL.value
                                last_signal_date = current_date
                                last_signal_type = RSIStrategy.SignalType.SELL.value

                # 预判下一个交易日是否有异常
                if i < len(self.df) - 1:
                    next_day = self.df.iloc[i + 1]
                    current_signal = self.df.iloc[i]['交易信号']

                    if current_signal and next_day['是价格异常']:
                        signal_type = RSIStrategy.SignalType.BUY if current_signal == RSIStrategy.SignalType.BUY.value else RSIStrategy.SignalType.SELL
                        self.df.loc[self.df.index[i], '交易信号'] = ''
                        pending_signal = {'type': signal_type, 'original_date': current_date}

                    elif current_signal and next_day['是停牌']:
                        self.df.loc[self.df.index[i], '交易信号'] = ''

            # 确保第一笔是买入
            self._ensure_first_signal_is_buy()

            return self.df

    def _check_interval(self, current_date, last_date):
        """检查交易间隔"""
        if last_date is None:
            return True
        days_diff = (current_date - pd.to_datetime(last_date)).days
        return days_diff >= self.min_interval_days

    def _ensure_first_signal_is_buy(self):
        """确保第一笔信号是买入"""
        first_signal_idx = self.df[self.df['交易信号'] != ''].index
        if len(first_signal_idx) > 0 and self.df.iloc[first_signal_idx[0]]['交易信号'] == RSIStrategy.SignalType.SELL.value:
            self.df.loc[first_signal_idx[0], '交易信号'] = ''

    def run_backtest(self) -> BacktestResult:
        """
        执行回测

        Returns:
            回测结果对象
        """
        if self.df is None or '交易信号' not in self.df.columns:
            raise ValueError("请先生成交易信号")

        # 回测变量初始化
        capital = self.initial_capital
        cash = capital
        position = 0
        trade_records = []

        daily_value = []
        daily_cash = []
        daily_position = []
        daily_price = []
        dates = []

        for i in range(len(self.df)):
            date = self.df.iloc[i]['日期']
            price = self.df.iloc[i]['收盘']
            signal = self.df.iloc[i]['交易信号']
            is_stop = self.df.iloc[i]['是停牌'] if '是停牌' in self.df.columns else False

            # 执行交易
            if not is_stop and signal:
                if signal == RSIStrategy.SignalType.BUY.value:
                    # 买入
                    max_shares = int(cash / price)
                    if max_shares > 0:
                        for shares in range(max_shares, 0, -1):
                            trade_value = shares * price
                            commission = max(trade_value * self.commission_rate, self.min_commission)
                            total_cost = trade_value + commission

                            if total_cost <= cash:
                                cash -= total_cost
                                position += shares
                                trade_records.append(RSIStrategy.TradeRecord(
                                    日期=date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                                    类型='买入',
                                    价格=price,
                                    数量=shares,
                                    金额=trade_value,
                                    佣金=commission,
                                    印花税=0,
                                    总费用=commission,
                                    现金_后=cash,
                                    持仓_后=position
                                ))
                                break

                elif signal == RSIStrategy.SignalType.SELL.value and position > 0:
                    # 卖出
                    trade_value = position * price
                    commission = max(trade_value * self.commission_rate, self.min_commission)
                    stamp_tax = trade_value * self.stamp_tax_rate
                    total_received = trade_value - commission - stamp_tax

                    trade_records.append(RSIStrategy.TradeRecord(
                        日期=date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        类型='卖出',
                        价格=price,
                        数量=position,
                        金额=trade_value,
                        佣金=commission,
                        印花税=stamp_tax,
                        总费用=commission + stamp_tax,
                        现金_后=cash + total_received,
                        持仓_后=0
                    ))

                    cash += total_received
                    position = 0

            # 计算当日资产
            if not is_stop:
                current_value = position * price + cash
            else:
                current_value = daily_value[-1] if daily_value else capital

            dates.append(date)
            daily_value.append(current_value)
            daily_cash.append(cash)
            daily_position.append(position)
            daily_price.append(price)

        # 创建每日资产DataFrame
        df_daily = pd.DataFrame({
            '日期': dates,
            '收盘价': daily_price,
            '资产总值': daily_value,
            '现金': daily_cash,
            '持仓': daily_position,
            '是停牌': self.df['是停牌'] if '是停牌' in self.df.columns else [False] * len(self.df)
        })

        # 计算绩效指标
        result = self._calculate_performance_metrics(df_daily, trade_records)

        # 存储结果
        self.current_result = result
        self.backtest_results.append(result)

        return result

    def _calculate_performance_metrics(self, df_daily: pd.DataFrame,
                                       trade_records: List[TradeRecord]) -> BacktestResult:
        """计算绩效指标"""
        # 计算买入持有资产
        first_price = df_daily.loc[0, '收盘价']
        df_daily['买入持有资产'] = self.initial_capital * df_daily['收盘价'] / first_price
        df_daily.loc[df_daily['是停牌'], '买入持有资产'] = df_daily['买入持有资产'].shift(1).fillna(
            self.initial_capital)

        # 计算日收益率
        df_daily['策略日收益率'] = df_daily['资产总值'].pct_change()
        df_daily['买入持有日收益率'] = df_daily['买入持有资产'].pct_change()

        # 计算最大回撤
        df_daily['策略累计最大值'] = df_daily['资产总值'].cummax()
        df_daily['策略回撤'] = (df_daily['资产总值'] - df_daily['策略累计最大值']) / df_daily['策略累计最大值'] * 100

        # 计算最终资产和总收益率
        final_value = df_daily['资产总值'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100

        # 计算年化收益率
        trading_days = len(df_daily[~df_daily['是停牌']])
        years = trading_days / 245
        annual_return = (final_value / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0

        # 最大回撤
        max_drawdown = df_daily['策略回撤'].min()

        # 夏普比率
        risk_free_rate = 0.03
        strategy_daily_returns = df_daily['策略日收益率'].dropna()
        if len(strategy_daily_returns) > 0 and strategy_daily_returns.std() != 0:
            sharpe_ratio = np.sqrt(245) * (
                        strategy_daily_returns.mean() - risk_free_rate / 245) / strategy_daily_returns.std()
        else:
            sharpe_ratio = 0

        # 胜率和盈亏比
        win_rate = 0
        profit_loss_ratio = 0
        total_fees = 0

        if len(trade_records) > 0:
            trade_df = pd.DataFrame([vars(t) for t in trade_records])
            buy_trades = trade_df[trade_df['类型'] == '买入'].reset_index(drop=True)
            sell_trades = trade_df[trade_df['类型'] == '卖出'].reset_index(drop=True)
            total_fees = trade_df['佣金'].sum() + trade_df['印花税'].sum()

            trade_pnl = []
            win_count = 0
            total_profit = 0
            total_loss = 0

            for i in range(min(len(buy_trades), len(sell_trades))):
                buy = buy_trades.iloc[i]
                sell = sell_trades.iloc[i]
                pnl = sell['金额'] - sell['佣金'] - sell['印花税'] - buy['金额'] - buy['佣金']
                trade_pnl.append(pnl)

                if pnl > 0:
                    win_count += 1
                    total_profit += pnl
                else:
                    total_loss += abs(pnl)

            if len(trade_pnl) > 0:
                win_rate = win_count / len(trade_pnl) * 100
            if total_loss > 0:
                profit_loss_ratio = total_profit / total_loss

        # 月度收益
        df_daily['年月'] = df_daily['日期'].dt.to_period('M')
        monthly = df_daily.groupby('年月')['资产总值'].agg(['first', 'last'])
        monthly['月收益率'] = (monthly['last'] / monthly['first'] - 1) * 100
        monthly_win = len(monthly[monthly['月收益率'] > 0])
        monthly_loss = len(monthly[monthly['月收益率'] < 0])
        monthly_avg = monthly['月收益率'].mean()

        # 创建结果对象
        result = RSIStrategy.BacktestResult(
            code=self.stock_code,
            params={
                'buy_threshold': self.buy_threshold,
                'sell_threshold': self.sell_threshold,
                'rsi_period': self.rsi_period,
                'min_interval_days': self.min_interval_days
            },
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annual_return=annual_return * 100,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            total_trades=len([t for t in trade_records if t.类型 == '买入']),
            total_fees=total_fees,
            monthly_win=monthly_win,
            monthly_loss=monthly_loss,
            monthly_avg_return=monthly_avg,
            start_date=df_daily['日期'].min().strftime('%Y-%m-%d'),
            end_date=df_daily['日期'].max().strftime('%Y-%m-%d'),
            trading_days=trading_days,
            daily_df=df_daily,
            trade_records=trade_records
        )

        return result

    def _calculate_hold_sharpe(self, daily_df: pd.DataFrame) -> float:
        """
        计算买入持有策略的夏普比率

        Args:
            daily_df: 包含每日资产数据的DataFrame

        Returns:
            买入持有策略的夏普比率
        """
        try:
            if daily_df is None:
                return 0.0

            # 计算日收益率
            if '买入持有日收益率' not in daily_df.columns:
                if '买入持有资产' in daily_df.columns:
                    daily_df['买入持有日收益率'] = daily_df['买入持有资产'].pct_change()
                else:
                    return 0.0

            hold_returns = daily_df['买入持有日收益率'].dropna()
            hold_returns = hold_returns[np.isfinite(hold_returns)]

            risk_free_rate = 0.03  # 无风险利率 3%

            if len(hold_returns) > 1 and hold_returns.std() > 1e-8:
                # 年化夏普比率计算
                excess_returns = hold_returns - risk_free_rate / 245
                sharpe = np.sqrt(245) * excess_returns.mean() / hold_returns.std()
                return sharpe
            return 0.0
        except Exception as e:
            warnings.warn(f"计算买入持有夏普比率时出错: {e}")
            return 0.0

    def plot_results(self, save_path: str = None) -> plt.Figure:
        """
        绘制回测结果图表

        Args:
            save_path: 保存路径，默认使用输出目录

        Returns:
            matplotlib图像对象
        """
        if self.current_result is None:
            raise ValueError("请先运行回测")

        df = self.df
        df_daily = self.current_result.daily_df
        trade_records = self.current_result.trade_records

        # 分离买入点和卖出点
        buy_signals = df[df['交易信号'] == RSIStrategy.SignalType.BUY.value]
        sell_signals = df[df['交易信号'] == RSIStrategy.SignalType.SELL.value]

        # 创建图表
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8-darkgrid')

        fig = plt.figure(figsize=(20, 30))
        gs_main = GridSpec(6, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1.2], hspace=0.35)

        # 子图1：价格 + 买卖点
        ax1 = fig.add_subplot(gs_main[0])
        normal_days = df[~df['是停牌']].copy()
        ax1.plot(normal_days['日期'], normal_days['收盘'], label='Close Price', color='blue', linewidth=1.5)
        ax1.scatter(buy_signals['日期'], buy_signals['收盘'], color='red', marker='^', s=100,
                    label='Buy Signal', zorder=5, edgecolors='black', linewidth=1)
        ax1.scatter(sell_signals['日期'], sell_signals['收盘'], color='green', marker='v', s=100,
                    label='Sell Signal', zorder=5, edgecolors='black', linewidth=1)
        ax1.set_title(f'{self.stock_name} - Price Chart with Buy/Sell Signals', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 子图2：RSI指标
        ax2 = fig.add_subplot(gs_main[1])
        ax2.plot(df['日期'], df['RSI'], label=f'RSI({self.rsi_period})', color='purple', linewidth=1.5)
        ax2.axhspan(self.sell_threshold, 100, alpha=0.2, color='red', label='Overbought')
        ax2.axhspan(0, self.buy_threshold, alpha=0.2, color='green', label='Oversold')
        ax2.axhline(y=self.sell_threshold, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.axhline(y=self.buy_threshold, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.scatter(buy_signals['日期'], buy_signals['RSI'], color='red', marker='^', s=100,
                    label='Buy Signal', zorder=5, edgecolors='black', linewidth=1)
        ax2.scatter(sell_signals['日期'], sell_signals['RSI'], color='green', marker='v', s=100,
                    label='Sell Signal', zorder=5, edgecolors='black', linewidth=1)
        ax2.set_title('RSI Indicator with Buy/Sell Signals', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RSI Value', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 子图3：持仓状态
        ax3 = fig.add_subplot(gs_main[2])
        ax3.fill_between(df_daily['日期'], 0, df_daily['持仓'],
                         where=df_daily['持仓'] > 0, color='blue', alpha=0.6, label='In Position')
        ax3.fill_between(df_daily['日期'], 0, df_daily['持仓'],
                         where=df_daily['持仓'] == 0, color='gray', alpha=0.3, label='Out of Position')
        ax3.scatter(buy_signals['日期'], [df_daily['持仓'].max() * 0.8] * len(buy_signals),
                    color='red', marker='^', s=80, label='Buy Signal', zorder=5)
        ax3.scatter(sell_signals['日期'], [0] * len(sell_signals), color='green', marker='v', s=80,
                    label='Sell Signal', zorder=5)
        ax3.set_title('Position Status', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Shares Held', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        # 子图4：策略净值对比
        ax4 = fig.add_subplot(gs_main[3])
        ax4.semilogy(df_daily['日期'], df_daily['资产总值'], label='Strategy NAV', color='red', linewidth=2)
        ax4.semilogy(df_daily['日期'], df_daily['买入持有资产'], label='Buy & Hold NAV', color='blue', linewidth=2,
                     alpha=0.7)
        ax4.axhline(y=self.initial_capital, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                    label='Initial Capital')
        ax4.set_title('Strategy NAV vs Buy & Hold NAV (Log Scale)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('NAV (Log Scale)', fontsize=12)
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3, which='both')

        # 子图5：策略回撤 vs 买入持有回撤对比（简化版）
        ax5 = fig.add_subplot(gs_main[4])

        # 确保有买入持有回撤数据
        if '持有回撤' not in df_daily.columns:
            df_daily['持有累计最大值'] = df_daily['买入持有资产'].cummax()
            df_daily['持有回撤'] = 0.0
            mask = df_daily['持有累计最大值'] > 0
            df_daily.loc[mask, '持有回撤'] = (df_daily.loc[mask, '买入持有资产'] - df_daily.loc[
                mask, '持有累计最大值']) / df_daily.loc[mask, '持有累计最大值'] * 100
            df_daily['持有回撤'] = df_daily['持有回撤'].clip(upper=0)

        # 绘制两条回撤曲线
        ax5.plot(df_daily['日期'], df_daily['策略回撤'], color='red', linewidth=1.5, label='Strategy Drawdown')
        ax5.plot(df_daily['日期'], df_daily['持有回撤'], color='blue', linewidth=1.5, linestyle='--',
                 label='Buy & Hold Drawdown')

        # 填充区域（可选）
        ax5.fill_between(df_daily['日期'], 0, df_daily['策略回撤'],
                         where=df_daily['策略回撤'] < 0, color='red', alpha=0.2)
        ax5.fill_between(df_daily['日期'], 0, df_daily['持有回撤'],
                         where=df_daily['持有回撤'] < 0, color='blue', alpha=0.1)

        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax5.set_title(
            f'Drawdown Comparison - Strategy vs Buy & Hold\n'
            f'Strategy: {self.current_result.max_drawdown:.2f}% | '
            f'Buy & Hold: {df_daily["持有回撤"].min():.2f}%',
            fontsize=14, fontweight='bold'
        )
        ax5.set_ylabel('Drawdown (%)', fontsize=12)
        ax5.set_ylim(min(df_daily['策略回撤'].min(), df_daily['持有回撤'].min()) * 1.1, 5)
        ax5.legend(loc='lower left')
        ax5.grid(True, alpha=0.3)

        # 子图6：收益率分布
        ax6 = fig.add_subplot(gs_main[5])
        daily_returns_pct = df_daily['策略日收益率'].dropna() * 100

        bin_width = 0.1
        min_return = np.floor(daily_returns_pct.min() / bin_width) * bin_width
        max_return = np.ceil(daily_returns_pct.max() / bin_width) * bin_width
        bins = np.arange(min_return, max_return + bin_width, bin_width)

        n, bins, patches = ax6.hist(daily_returns_pct, bins=bins, edgecolor='black', alpha=0.7)

        # 颜色设置
        for i, patch in enumerate(patches):
            x_center = (bins[i] + bins[i + 1]) / 2
            if x_center < -2:
                patch.set_facecolor('darkgreen')
            elif x_center < 0:
                patch.set_facecolor('lightgreen')
            elif x_center < 2:
                patch.set_facecolor('lightcoral')
            else:
                patch.set_facecolor('darkred')

        mean_return = daily_returns_pct.mean()
        ax6.axvline(x=mean_return, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.3f}%')
        ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)

        ax6.set_title('Daily Return Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Daily Return (%)', fontsize=10)
        ax6.set_ylabel('Frequency (Days)', fontsize=10)
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(True, alpha=0.3, axis='y')

        # 设置日期格式
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.tick_params(axis='x', rotation=45)

        plt.subplots_adjust(left=0.06, right=0.94, top=0.97, bottom=0.03, hspace=0.4)

        if save_path is None:
            save_path = self.output_dir / f'{self.stock_code}_RSI策略报告.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def _plot_trade_comparison(self, ax, trade_records, df_daily):
        """绘制交易对比图"""
        trade_df = pd.DataFrame([vars(t) for t in trade_records])
        buy_trades = trade_df[trade_df['类型'] == '买入'].reset_index(drop=True)
        sell_trades = trade_df[trade_df['类型'] == '卖出'].reset_index(drop=True)

        # 计算每笔交易收益率
        rsi_returns = []
        bh_returns = []
        period_labels = []

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy = buy_trades.iloc[i]
            sell = sell_trades.iloc[i]

            buy_cost = buy['金额'] + buy['佣金']
            sell_proceed = sell['金额'] - sell['佣金'] - sell['印花税']
            rsi_return = (sell_proceed - buy_cost) / buy_cost * 100
            rsi_returns.append(rsi_return)

            buy_date = pd.to_datetime(buy['日期'])
            sell_date = pd.to_datetime(sell['日期'])
            period_data = df_daily[(df_daily['日期'] >= buy_date) & (df_daily['日期'] <= sell_date)]

            if len(period_data) > 0:
                bh_return = (period_data.iloc[-1]['收盘价'] / period_data.iloc[0]['收盘价'] - 1) * 100
            else:
                bh_return = 0

            bh_returns.append(bh_return)
            period_labels.append(f"T{i + 1}")

        # 排序
        sorted_indices = sorted(range(len(rsi_returns)), key=lambda k: rsi_returns[k], reverse=True)
        rsi_returns_sorted = [rsi_returns[i] for i in sorted_indices]
        bh_returns_sorted = [bh_returns[i] for i in sorted_indices]
        labels_sorted = [period_labels[i] for i in sorted_indices]

        x_pos = np.arange(len(rsi_returns_sorted))
        width = 0.35

        # 柱状图
        rsi_colors = ['red' if r > 0 else 'green' for r in rsi_returns_sorted]
        ax.bar(x_pos - width / 2, rsi_returns_sorted, width, color=rsi_colors, alpha=0.7,
               edgecolor='black', label='RSI Strategy')

        bh_colors = ['lightcoral' if r > 0 else 'lightgreen' for r in bh_returns_sorted]
        ax.bar(x_pos + width / 2, bh_returns_sorted, width, color=bh_colors, alpha=0.7,
               edgecolor='black', label='Buy & Hold', hatch='//')

        ax.set_xlabel('Trading Periods')
        ax.set_ylabel('Period Return (%)')
        ax.set_title('Strategy vs Buy & Hold - Returns Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_sorted, rotation=45, fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=8)

    def print_results(self):
        """打印回测结果"""
        if self.current_result is None:
            raise ValueError("请先运行回测")

        r = self.current_result

        print("=" * 100)
        print(f"{self.stock_name} ({self.stock_code}) - {self.name} 回测结果".center(100))
        print("=" * 100)

        comparison_data = {
            '绩效指标': [
                '初始资金 (元)',
                '最终资产 (元)',
                '总收益率 (%)',
                '年化收益率 (%)',
                '最大回撤 (%)',
                '夏普比率',
                '交易次数',
                '胜率 (%)',
                '盈亏比',
                '总手续费 (元)',
                '盈利月数',
                '亏损月数',
                '平均月收益率 (%)'
            ],
            self.name: [
                f"{r.initial_capital:,.2f}",
                f"{r.final_value:,.2f}",
                f"{r.total_return:.2f}",
                f"{r.annual_return:.2f}",
                f"{r.max_drawdown:.2f}",
                f"{r.sharpe_ratio:.2f}",
                f"{r.total_trades}",
                f"{r.win_rate:.2f}" if r.total_trades > 0 else "-",
                f"{r.profit_loss_ratio:.2f}" if r.total_trades > 0 else "-",
                f"{r.total_fees:,.2f}" if r.total_trades > 0 else "-",
                f"{r.monthly_win}",
                f"{r.monthly_loss}",
                f"{r.monthly_avg_return:.2f}"
            ]
        }

        comparison_df = pd.DataFrame(comparison_data)
        pd.set_option('display.width', None)
        print(comparison_df.to_string(index=False))
        print("=" * 100)

        print("\n策略参数:")
        print(f"• RSI周期: {r.params['rsi_period']}")
        print(f"• 买入阈值: {r.params['buy_threshold']}")
        print(f"• 卖出阈值: {r.params['sell_threshold']}")
        print(f"• 最小间隔: {r.params['min_interval_days']}天")
        print(f"• 回测周期: {r.start_date} 至 {r.end_date}")
        print(f"• 交易天数: {r.trading_days} 天")

    def save_results(self, prefix: str = None) -> Dict[str, str]:
        """
        保存回测结果

        Args:
            prefix: 文件名前缀

        Returns:
            保存的文件路径字典
        """
        if self.current_result is None:
            raise ValueError("请先运行回测")

        prefix = prefix or self.stock_code
        saved_files = {}

        # 保存带信号的原始数据
        if self.df is not None:
            signal_file = self.output_dir / f'{prefix}_with_signals.csv'
            self.df.to_csv(signal_file, index=False, encoding='utf-8-sig')
            saved_files['signals'] = str(signal_file)

        # 保存每日资产明细
        if self.current_result.daily_df is not None:
            daily_file = self.output_dir / f'{prefix}_每日资产明细.csv'
            self.current_result.daily_df.to_csv(daily_file, index=False, encoding='utf-8-sig')
            saved_files['daily'] = str(daily_file)

        # 保存交易记录
        if self.current_result.trade_records:
            trade_df = pd.DataFrame([vars(t) for t in self.current_result.trade_records])
            trade_file = self.output_dir / f'{prefix}_交易记录.csv'
            trade_df.to_csv(trade_file, index=False, encoding='utf-8-sig')
            saved_files['trades'] = str(trade_file)

        # 保存结果摘要
        summary_file = self.output_dir / f'{prefix}_结果摘要.csv'
        summary_data = {
            '指标': ['股票代码', '初始资金', '最终资产', '总收益率', '年化收益率',
                     '最大回撤', '夏普比率', '交易次数', '胜率', '盈亏比'],
            '值': [
                self.stock_code,
                self.current_result.initial_capital,
                self.current_result.final_value,
                self.current_result.total_return,
                self.current_result.annual_return,
                self.current_result.max_drawdown,
                self.current_result.sharpe_ratio,
                self.current_result.total_trades,
                self.current_result.win_rate,
                self.current_result.profit_loss_ratio
            ]
        }
        pd.DataFrame(summary_data).to_csv(summary_file, index=False, encoding='utf-8-sig')
        saved_files['summary'] = str(summary_file)

        print(f"\n结果已保存到: {self.output_dir}")
        return saved_files

    def run_complete_analysis(self,
                              file_path: str = None,
                              df: pd.DataFrame = None,
                              stock_code: str = None,
                              stock_name: str = None,
                              save_results: bool = True,
                              plot: bool = True) -> BacktestResult:
        """
        运行完整分析流程

        Args:
            file_path: 数据文件路径
            df: 直接传入数据框
            stock_code: 股票代码
            stock_name: 股票名称
            save_results: 是否保存结果
            plot: 是否绘制图表

        Returns:
            回测结果对象
        """
        print("=" * 80)
        print(f"开始 {self.name} 完整分析...".center(80))
        print("=" * 80)

        # 加载数据
        if file_path:
            self.load_data(file_path, stock_code, stock_name)
        elif df is not None:
            self.load_data_from_df(df, stock_code, stock_name)
        else:
            raise ValueError("请提供数据文件路径或数据框")

        print(f"股票代码: {self.stock_code}")
        print(f"数据行数: {len(self.df)}")

        # 数据预处理
        self.preprocess_data()
        print("数据预处理完成")

        # 计算RSI
        self.calculate_rsi()
        print(f"RSI计算完成 (周期={self.rsi_period})")

        # 生成信号
        self.generate_signals()
        buy_count = len(self.df[self.df['交易信号'] == RSIStrategy.SignalType.BUY.value])
        sell_count = len(self.df[self.df['交易信号'] == RSIStrategy.SignalType.SELL.value])
        print(f"信号生成完成 - 买入: {buy_count}, 卖出: {sell_count}")

        # 运行回测
        result = self.run_backtest()
        print("回测完成")

        # 打印结果
        self.print_results()

        # 绘制图表
        if plot:
            self.plot_results()
            print("图表已生成")

        # 保存结果
        if save_results:
            self.save_results()

        print("\n分析完成！")
        print("=" * 80)

        return result

    @staticmethod
    def compare_stocks(
            input_source: Union[str, List[str]],
            stock_codes: List[str] = None,
            initial_capital: float = None,
            rsi_period: int = None,
            buy_threshold: int = None,
            sell_threshold: int = None,
            min_interval_days: int = None,
            commission_rate: float = None,
            stamp_tax_rate: float = None,
            output_dir: str = None,
            save_results: bool = True, #单个股票结果是否展示和保存
            plot: bool = True #单个股票图表是否展示和保存
    ) -> pd.DataFrame:
        """
        多股票比较 - 支持自定义参数
        """

        # ============ 1. 参数处理 ============
        # 创建临时实例获取默认值
        temp_strategy = RSIStrategy(output_dir=output_dir) if output_dir else RSIStrategy()

        params = {
            'initial_capital': initial_capital if initial_capital is not None else temp_strategy.initial_capital,
            'commission_rate': commission_rate if commission_rate is not None else temp_strategy.commission_rate,
            'min_commission': temp_strategy.min_commission,
            'stamp_tax_rate': stamp_tax_rate if stamp_tax_rate is not None else temp_strategy.stamp_tax_rate,
            'buy_threshold': buy_threshold if buy_threshold is not None else temp_strategy.buy_threshold,
            'sell_threshold': sell_threshold if sell_threshold is not None else temp_strategy.sell_threshold,
            'rsi_period': rsi_period if rsi_period is not None else temp_strategy.rsi_period,
            'min_interval_days': min_interval_days if min_interval_days is not None else temp_strategy.min_interval_days,
            'output_dir': Path(output_dir) if output_dir else Path(temp_strategy.output_dir),
            'save_results': save_results,
            'plot': plot
        }

        # 创建输出目录
        params['output_dir'].mkdir(parents=True, exist_ok=True)

        # ============ 2. 获取所有文件路径 ============
        stock_files = []
        custom_codes = []

        if isinstance(input_source, str):
            folder_path = Path(input_source)
            if not folder_path.exists():
                raise ValueError(f"文件夹不存在: {folder_path}")

            stock_files = list(folder_path.glob("*.csv"))
            if not stock_files:
                raise ValueError(f"文件夹中没有CSV文件: {folder_path}")

            print(f"📁 从文件夹读取到 {len(stock_files)} 个CSV文件")
            custom_codes = [f.stem.split('_')[2] if '_' in f.stem else f.stem for f in stock_files]

        elif isinstance(input_source, list):
            stock_files = [Path(f) for f in input_source]
            for f in stock_files:
                if not f.exists():
                    raise ValueError(f"文件不存在: {f}")

            print(f"📄 读取到 {len(stock_files)} 个文件")

            if stock_codes and len(stock_codes) == len(stock_files):
                custom_codes = stock_codes
            else:
                custom_codes = [f.stem.split('_')[2] if '_' in f.stem else f.stem for f in stock_files]
        else:
            raise ValueError("input_source必须是文件夹路径或文件路径列表")

        # 打印使用的参数
        print("\n📌 本次比较使用的参数:")
        print(f"   • 初始资金: {params['initial_capital']:,.0f}元")
        print(f"   • RSI周期: {params['rsi_period']}")
        print(f"   • 买入阈值: {params['buy_threshold']}")
        print(f"   • 卖出阈值: {params['sell_threshold']}")
        print(f"   • 最小间隔: {params['min_interval_days']}天")
        print(f"   • 佣金费率: {params['commission_rate'] * 100}%")
        print(f"   • 印花税率: {params['stamp_tax_rate'] * 100}%")

        # ============ 3. 定义辅助函数 ============
        def calculate_hold_drawdown(daily_df):
            """计算买入持有最大回撤"""
            try:
                if daily_df is None or '买入持有资产' not in daily_df.columns:
                    return 0.0

                # 计算累计最大值
                cummax = daily_df['买入持有资产'].cummax()
                # 避免除零错误
                mask = cummax > 0
                drawdown = pd.Series(0.0, index=daily_df.index)
                drawdown[mask] = (daily_df.loc[mask, '买入持有资产'] - cummax[mask]) / cummax[mask] * 100
                drawdown = drawdown.clip(upper=0)

                return drawdown.min() if not drawdown.empty else 0.0
            except Exception as e:
                warnings.warn(f"计算买入持有回撤时出错: {e}")
                return 0.0

        def calculate_hold_sharpe(daily_df):
            """计算买入持有夏普比率"""
            try:
                if daily_df is None:
                    return 0.0

                # 计算日收益率
                if '买入持有日收益率' not in daily_df.columns:
                    if '买入持有资产' in daily_df.columns:
                        daily_df['买入持有日收益率'] = daily_df['买入持有资产'].pct_change()
                    else:
                        return 0.0

                hold_returns = daily_df['买入持有日收益率'].dropna()
                hold_returns = hold_returns[np.isfinite(hold_returns)]

                risk_free_rate = 0.03

                if len(hold_returns) > 1 and hold_returns.std() > 1e-8:
                    excess_returns = hold_returns - risk_free_rate / 245
                    sharpe = np.sqrt(245) * excess_returns.mean() / hold_returns.std()
                    return sharpe
                return 0.0
            except Exception as e:
                warnings.warn(f"计算买入持有夏普比率时出错: {e}")
                return 0.0

        # ============ 4. 执行回测 ============
        results = []

        print("\n" + "=" * 100)
        print("多股票策略比较分析".center(100))
        print("=" * 100)

        for i, (file_path, code) in enumerate(zip(stock_files, custom_codes)):
            print(f"\n▶ 正在处理 [{i + 1}/{len(stock_files)}] {code}...")

            try:
                # 创建新实例
                strategy = RSIStrategy(
                    name=f"RSI策略_{code}",
                    initial_capital=params['initial_capital'],
                    commission_rate=params['commission_rate'],
                    min_commission=params['min_commission'],
                    stamp_tax_rate=params['stamp_tax_rate'],
                    buy_threshold=params['buy_threshold'],
                    sell_threshold=params['sell_threshold'],
                    rsi_period=params['rsi_period'],
                    min_interval_days=params['min_interval_days'],
                    output_dir=params['output_dir'] / code  # 每个股票单独的输出目录
                )

                # 执行分析
                result = strategy.run_complete_analysis(
                    file_path=str(file_path),
                    stock_code=code,
                    save_results=params['save_results'],
                    plot=params['plot'],
                )

                results.append(result)
                print(f"  ✓ 完成 - 收益率: {result.total_return:.2f}%")

            except Exception as e:
                print(f"  ✗ 失败: {str(e)}")
                continue

        if not results:
            raise ValueError("没有成功处理任何股票")

        # ============ 5. 创建比较表格 ============
        comparison_data = []
        for r in results:
            try:
                hold_return = (r.daily_df['买入持有资产'].iloc[-1] / r.initial_capital - 1) * 100
                hold_dd = calculate_hold_drawdown(r.daily_df)
                hold_sharpe = calculate_hold_sharpe(r.daily_df)
                excess_return = r.total_return - hold_return
                dd_improve = r.max_drawdown - hold_dd

                comparison_data.append({
                    '股票代码': r.code,
                    '策略总收益率(%)': r.total_return,
                    '策略年化收益率(%)': r.annual_return,
                    '策略最大回撤(%)': r.max_drawdown,
                    '策略夏普比率': r.sharpe_ratio,
                    '策略胜率(%)': r.win_rate,
                    '策略盈亏比': r.profit_loss_ratio,
                    '策略交易次数': r.total_trades,
                    '买入持有收益率(%)': hold_return,
                    '买入持有最大回撤(%)': hold_dd,
                    '买入持有夏普比率': hold_sharpe,
                    '超额收益(%)': excess_return,
                    '回撤改善(%)': dd_improve
                })
            except Exception as e:
                warnings.warn(f"处理股票 {r.code} 的数据时出错: {e}")
                continue

        comparison = pd.DataFrame(comparison_data)

        if comparison.empty:
            raise ValueError("没有有效的比较数据")

        # ============ 6. 控制台输出 ============
        print("\n" + "=" * 100)
        print("策略应用整体效果总结".center(100))
        print("=" * 100)

        # 计算各项指标
        beat_benchmark_pct = (comparison['策略总收益率(%)'] > comparison['买入持有收益率(%)']).mean() * 100
        drawdown_improved_pct = (comparison['回撤改善(%)'] > 0).mean() * 100

        avg_strategy_dd = comparison['策略最大回撤(%)'].mean()
        avg_hold_dd = comparison['买入持有最大回撤(%)'].mean()

        avg_strategy_sharpe = comparison['策略夏普比率'].mean()
        avg_hold_sharpe = comparison['买入持有夏普比率'].mean()

        avg_win_rate = comparison['策略胜率(%)'].mean()
        avg_profit_loss = comparison['策略盈亏比'].mean()
        avg_trades = comparison['策略交易次数'].mean()

        # 格式化输出
        print(f"\n📊 跑赢基准比例: {beat_benchmark_pct:.1f}%")
        print(f"📉 回撤改善比例: {drawdown_improved_pct:.1f}%")
        print(f"📈 策略平均回撤 vs 买入持有平均回撤：{avg_strategy_dd:.1f}% vs {avg_hold_dd:.1f}%")
        print(f"⚡ 策略平均夏普 vs 买入持有平均夏普：{avg_strategy_sharpe:.2f} vs {avg_hold_sharpe:.2f}")
        print(
            f"🎯 策略整体表现：平均胜率{avg_win_rate:.1f}%，平均盈亏比{avg_profit_loss:.2f}，平均交易次数{avg_trades:.0f}次")

        print(f"\n📌 统计信息:")
        print(f"   • 成功分析股票数量: {len(comparison)}/{len(stock_files)}")
        print(f"   • 最大超额收益: {comparison['超额收益(%)'].max():.2f}%")
        print(f"   • 最大回撤改善: {comparison['回撤改善(%)'].max():.2f}%")
        print(f"   • 最佳夏普比率: {comparison['策略夏普比率'].max():.2f}")

        # ============ 7. 绘图展示 ============
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建2x2子图布局 - 使用constrained_layout避免tight_layout警告
        fig = plt.figure(figsize=(20, 16), constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # ===== 子图a：最大回撤分布对比直方图（绝对值，0-1范围） =====
        ax1 = fig.add_subplot(gs[0, 0])

        # 提取数据并转换为绝对值（正数）
        strategy_dd = abs(comparison['策略最大回撤(%)'].dropna()) / 100  # 转换为0-1范围
        hold_dd = abs(comparison['买入持有最大回撤(%)'].dropna()) / 100  # 转换为0-1范围

        if len(strategy_dd) > 0 and len(hold_dd) > 0:
            # 统计超出1的股票数
            strategy_outliers = (strategy_dd > 1).sum()
            hold_outliers = (hold_dd > 1).sum()

            # 限制在0-1范围内
            strategy_dd = strategy_dd.clip(upper=1)
            hold_dd = hold_dd.clip(upper=1)

            # 设置bins：0-1范围，每0.02一个柱子
            bins = np.arange(0, 1.02, 0.02)

            # 绘制直方图
            ax1.hist(strategy_dd, bins=bins, alpha=0.7, color='red',
                     edgecolor='black', label='RSI策略', density=True)
            ax1.hist(hold_dd, bins=bins, alpha=0.5, color='blue',
                     edgecolor='black', label='买入持有', density=True)

            # 添加均值线
            ax1.axvline(strategy_dd.mean(), color='darkred', linestyle='--', linewidth=2,
                        label=f'策略均值: {strategy_dd.mean():.2f}')
            ax1.axvline(hold_dd.mean(), color='darkblue', linestyle='--', linewidth=2,
                        label=f'买入持有均值: {hold_dd.mean():.2f}')

            # 添加超出范围的提示
            if strategy_outliers > 0 or hold_outliers > 0:
                outlier_text = f'超出范围(>100%):\n策略: {strategy_outliers}只\n买入持有: {hold_outliers}只'
                ax1.text(0.98, 0.98, outlier_text, transform=ax1.transAxes, fontsize=9,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:
            ax1.text(0.5, 0.5, '无有效回撤数据', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=14)

        ax1.set_xlabel('最大回撤 (绝对值)', fontsize=12)
        ax1.set_ylabel('概率密度', fontsize=12)
        ax1.set_title('a. 最大回撤分布对比 (绝对值)', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.set_xticks(np.arange(0, 1.1, 0.1))
        ax1.set_xticklabels([f'{x:.2f}' for x in np.arange(0, 1.1, 0.1)])
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ===== 子图b：胜率 vs 盈亏比四象限散点图（去掉股票代码标签） =====
        ax2 = fig.add_subplot(gs[0, 1])

        win_rate = comparison['策略胜率(%)'].dropna() / 100  # 转换为0-1范围
        pl_ratio = comparison['策略盈亏比'].dropna()
        sharpe = comparison['策略夏普比率'].dropna()

        if len(win_rate) > 0 and len(pl_ratio) > 0 and len(sharpe) > 0:
            # 计算四分位点
            win_median = win_rate.median()
            pl_median = pl_ratio.median()

            # 绘制四象限分割线
            ax2.axhline(y=pl_median, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax2.axvline(x=win_median, color='gray', linestyle='--', alpha=0.5, linewidth=1)

            # 添加象限标签
            ax2.text(0.02, 0.98, '高盈亏比\n低胜率', transform=ax2.transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax2.text(0.98, 0.98, '高盈亏比\n高胜率', transform=ax2.transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax2.text(0.02, 0.02, '低盈亏比\n低胜率', transform=ax2.transAxes, fontsize=9,
                     verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
            ax2.text(0.98, 0.02, '低盈亏比\n高胜率', transform=ax2.transAxes, fontsize=9,
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 散点图，点变小，不带标签
            scatter = ax2.scatter(win_rate, pl_ratio,
                                  c=sharpe, cmap='RdYlGn',
                                  s=30, alpha=0.6, edgecolors='black', linewidth=0.5, zorder=5)  # s从100改为30

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax2, label='夏普比率')
            cbar.ax.tick_params(labelsize=8)

            # 添加统计信息
            stats_text = f'胜率中位数: {win_median:.1%}\n盈亏比中位数: {pl_median:.2f}'
            ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, fontsize=8,
                     verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax2.text(0.5, 0.5, '无有效数据', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=14)

        ax2.set_xlabel('胜率', fontsize=12)
        ax2.set_ylabel('盈亏比', fontsize=12)
        ax2.set_title('b. 策略胜率 vs 盈亏比分布', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_xticks(np.arange(0, 1.1, 0.2))
        ax2.set_xticklabels([f'{x:.0%}' for x in np.arange(0, 1.1, 0.2)])
        ax2.grid(True, alpha=0.3)

        # ===== 子图c：策略收益率 vs 买入持有收益率散点图（数值/100） =====
        ax3 = fig.add_subplot(gs[1, 0])

        # 转换为0-1范围（除以100）
        strategy_return = comparison['策略总收益率(%)'].dropna() / 100
        hold_return = comparison['买入持有收益率(%)'].dropna() / 100
        excess_return = comparison['超额收益(%)'].dropna() / 100

        if len(strategy_return) > 0 and len(hold_return) > 0 and len(excess_return) > 0:
            # 计算收益率范围
            max_return = max(strategy_return.max(), hold_return.max())
            min_return = min(strategy_return.min(), hold_return.min())
            margin = (max_return - min_return) * 0.1 if max_return != min_return else 0.1

            # 绘制45°对角线
            line_x = [min_return - margin, max_return + margin]
            line_y = [min_return - margin, max_return + margin]
            ax3.plot(line_x, line_y, '--', color='gray', alpha=0.5, linewidth=2, label='收益率相等')

            # 填充区域
            ax3.fill_between(line_x, line_y, max_return + margin,
                             alpha=0.1, color='green', label='策略跑赢')
            ax3.fill_between(line_x, min_return - margin, line_y,
                             alpha=0.1, color='red', label='策略跑输')

            # 散点图，点变小，不带标签
            scatter = ax3.scatter(hold_return, strategy_return,
                                  c=excess_return, cmap='RdYlGn',
                                  s=30, alpha=0.6, edgecolors='black', linewidth=0.5, zorder=5)  # s从100改为30

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax3, label='超额收益')
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

            # 添加统计信息
            stats_text = f'跑赢基准: {(strategy_return > hold_return).sum()}/{len(strategy_return)}\n'
            stats_text += f'平均超额: {excess_return.mean():.2%}'
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=8,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax3.text(0.5, 0.5, '无有效数据', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=14)

        ax3.set_xlabel('买入持有收益率', fontsize=12)
        ax3.set_ylabel('策略收益率', fontsize=12)
        ax3.set_title('c. 策略 vs 买入持有收益率对比', fontsize=14, fontweight='bold')
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax3.legend(loc='upper left', fontsize=10)
        ax3.grid(True, alpha=0.3)

        # ===== 子图d：超额收益 vs 回撤改善四象限散点图（数值/100，纵轴最大1） =====
        ax4 = fig.add_subplot(gs[1, 1])

        # 转换为0-1范围（除以100）
        excess = comparison['超额收益(%)'].dropna() / 100
        dd_improve = comparison['回撤改善(%)'].dropna() / 100
        sharpe = comparison['策略夏普比率'].dropna()

        if len(excess) > 0 and len(dd_improve) > 0 and len(sharpe) > 0:
            # 计算四分位点
            excess_median = excess.median()
            dd_improve_median = dd_improve.median()

            # 绘制四象限分割线
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

            # 散点图，点变小，不带标签
            scatter = ax4.scatter(excess, dd_improve,
                                  c=sharpe, cmap='RdYlGn',
                                  s=30, alpha=0.6, edgecolors='black', linewidth=0.5, zorder=5)  # s从100改为30

            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax4, label='夏普比率')
            cbar.ax.tick_params(labelsize=8)

            # 添加统计信息
            stats_text = f'超额中位数: {excess_median:.2%}\n'
            stats_text += f'改善中位数: {dd_improve_median:.2%}'
            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=8,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax4.text(0.5, 0.5, '无有效数据', ha='center', va='center',
                     transform=ax4.transAxes, fontsize=14)

        ax4.set_xlabel('超额收益', fontsize=12)
        ax4.set_ylabel('回撤改善', fontsize=12)
        ax4.set_title('d. 超额收益 vs 回撤改善分布', fontsize=14, fontweight='bold')
        ax4.set_ylim(-1, 1)  # 纵轴最大值设为1
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax4.grid(True, alpha=0.3)

        # 添加整体标题
        plt.suptitle(f'多股票RSI策略比较分析 (共{len(comparison)}只股票)',
                     fontsize=16, fontweight='bold', y=0.98)

        # 保存图片
        comp_file = Path(params['output_dir']) / '多股票比较_详细分析.png'
        plt.savefig(comp_file, dpi=300, bbox_inches='tight')
        plt.show()

        # 保存比较结果CSV
        comparison_export = comparison.copy()

        # 处理数值列的显示格式
        for col in comparison_export.columns:
            if col == '股票代码':
                continue
            elif col == '策略交易次数':
                # 交易次数保持整数
                comparison_export[col] = comparison_export[col].astype(int)
            elif col == '策略夏普比率' or col == '买入持有夏普比率':
                comparison_export[col] = comparison_export[col]
            else:
                # 其他数值列/100
                comparison_export[col] = comparison_export[col]/100

        # 保存CSV
        comp_csv = Path(params['output_dir']) / '多股票比较_详细数据.csv'
        comparison_export.to_csv(comp_csv, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 100)
        print("分析完成！".center(100))
        print("=" * 100)
        print(f"📁 详细数据已保存到: {comp_csv}")
        print(f"📊 分析图表已保存到: {comp_file}")

        return comparison

    @staticmethod
    def compare_parameters(
            file_dir: str,  # 输入的文件地址
            stock_codes: List[str] = None,
            initial_capital: float = None,
            rsi_parameters: List[List] = None,  # rsi参数组合，包括rsi_period，buy_threshold，sell_threshold三个参数的列表集合
            min_interval_days: int = None,
            commission_rate: float = None,
            stamp_tax_rate: float = None,
            output_dir: str = None,
    ) -> pd.DataFrame:
        """
        对比不同的RSI参数策略作用于同一只股票的效果

        Args:
            file_dir: 股票数据文件路径
            stock_codes: 股票代码列表（如果文件是单个股票，可以传入单个股票代码）
            initial_capital: 初始资金
            rsi_parameters: RSI参数组合列表，例如 [[14,30,70], [9,25,75], [21,20,80]]
                           每个子列表包含 [rsi_period, buy_threshold, sell_threshold]
            min_interval_days: 最小间隔天数
            commission_rate: 佣金费率
            stamp_tax_rate: 印花税率
            output_dir: 输出目录

        Returns:
            参数对比结果数据框
        """

        # ============ 1. 参数处理 ============
        # 创建临时实例获取默认值
        temp_strategy = RSIStrategy(output_dir=output_dir) if output_dir else RSIStrategy()

        # 处理rsi_parameters
        if rsi_parameters is None:
            # 默认参数组合
            rsi_parameters = [
                [9, 25, 75],  # 短期RSI，激进
                [14, 30, 70],  # 标准RSI
                [21, 40, 60],  # 长期RSI，保守
            ]

        # 基础参数
        base_params = {
            'initial_capital': initial_capital if initial_capital is not None else temp_strategy.initial_capital,
            'commission_rate': commission_rate if commission_rate is not None else temp_strategy.commission_rate,
            'min_commission': temp_strategy.min_commission,
            'stamp_tax_rate': stamp_tax_rate if stamp_tax_rate is not None else temp_strategy.stamp_tax_rate,
            'min_interval_days': min_interval_days if min_interval_days is not None else temp_strategy.min_interval_days,
            'output_dir': Path(output_dir) if output_dir else Path(temp_strategy.output_dir) / "parameter_comparison",
        }

        # 创建输出目录
        base_params['output_dir'].mkdir(parents=True, exist_ok=True)

        # ============ 2. 获取文件路径 ============
        file_path = Path(file_dir)
        if not file_path.exists():
            raise ValueError(f"文件不存在: {file_path}")

        # 处理股票代码
        if stock_codes and len(stock_codes) > 0:
            stock_code = stock_codes[0]
        else:
            stock_code = file_path.stem.split('_')[2] if '_' in file_path.stem else file_path.stem

        print("\n" + "=" * 100)
        print(f"RSI参数对比分析 - 股票: {stock_code}".center(100))
        print("=" * 100)

        print(f"\n📌 基础参数:")
        print(f"   • 初始资金: {base_params['initial_capital']:,.0f}元")
        print(f"   • 最小间隔: {base_params['min_interval_days']}天")
        print(f"   • 佣金费率: {base_params['commission_rate'] * 100}%")
        print(f"   • 印花税率: {base_params['stamp_tax_rate'] * 100}%")

        print(f"\n📊 待测试的RSI参数组合 ({len(rsi_parameters)}组):")
        for i, params in enumerate(rsi_parameters):
            print(f"   组合{i + 1}: RSI周期={params[0]}, 买入阈值={params[1]}, 卖出阈值={params[2]}")

        # ============ 3. 执行回测 ============
        results = []
        nav_data = {}  # 存储净值数据用于绘图

        print("\n" + "=" * 100)
        print("开始回测...".center(100))
        print("=" * 100)

        for i, params in enumerate(rsi_parameters):
            rsi_period, buy_threshold, sell_threshold = params

            print(f"\n▶ 测试组合{i + 1}: RSI周期={rsi_period}, 买入={buy_threshold}, 卖出={sell_threshold}")

            try:
                # 创建策略实例
                strategy = RSIStrategy(
                    name=f"RSI_{rsi_period}_{buy_threshold}_{sell_threshold}",
                    initial_capital=base_params['initial_capital'],
                    commission_rate=base_params['commission_rate'],
                    min_commission=base_params['min_commission'],
                    stamp_tax_rate=base_params['stamp_tax_rate'],
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                    rsi_period=rsi_period,
                    min_interval_days=base_params['min_interval_days'],
                    output_dir=base_params['output_dir'] / f"param_{i + 1}"
                )

                # 执行分析
                result = strategy.run_complete_analysis(
                    file_path=str(file_path),
                    stock_code=stock_code,
                    save_results=False,
                    plot=False,
                )

                results.append({
                    '参数名称': f'RSI({rsi_period},{buy_threshold},{sell_threshold})',
                    'rsi_period': rsi_period,
                    'buy_threshold': buy_threshold,
                    'sell_threshold': sell_threshold,
                    'result': result,
                    'nav': result.daily_df[['日期', '资产总值']].copy()
                })

                # 存储净值数据
                nav_data[f'策略{i + 1}'] = result.daily_df[['日期', '资产总值']].copy()

                print(f"  ✓ 完成 - 收益率: {result.total_return:.2f}%, 夏普: {result.sharpe_ratio:.2f}")

            except Exception as e:
                print(f"  ✗ 失败: {str(e)}")
                continue

        if not results:
            raise ValueError("没有成功测试任何参数组合")

        # ============ 4. 定义辅助函数(买入持有最大回撤) ============
        def calculate_hold_drawdown(daily_df):
            """计算买入持有最大回撤"""
            try:
                if daily_df is None or '买入持有资产' not in daily_df.columns:
                    return 0.0

                # 计算累计最大值
                cummax = daily_df['买入持有资产'].cummax()
                # 避免除零错误
                mask = cummax > 0
                drawdown = pd.Series(0.0, index=daily_df.index)
                drawdown[mask] = (daily_df.loc[mask, '买入持有资产'] - cummax[mask]) / cummax[mask] * 100
                drawdown = drawdown.clip(upper=0)

                return drawdown.min() if not drawdown.empty else 0.0
            except Exception as e:
                warnings.warn(f"计算买入持有回撤时出错: {e}")
                return 0.0

        # ============ 4. 获取买入持有数据 ============
        # 重新运行一个基础策略来获取买入持有数据
        base_strategy = RSIStrategy(
            name="买入持有",
            initial_capital=base_params['initial_capital'],
            commission_rate=base_params['commission_rate'],
            min_commission=base_params['min_commission'],
            stamp_tax_rate=base_params['stamp_tax_rate'],
            output_dir=base_params['output_dir'] / "baseline"
        )

        base_result = base_strategy.run_complete_analysis(
            file_path=str(file_path),
            stock_code=stock_code,
            save_results=False,
            plot=False,
        )

        # 计算买入持有指标
        hold_return = (base_result.daily_df['买入持有资产'].iloc[-1] / base_params['initial_capital'] - 1) * 100
        hold_dd = calculate_hold_drawdown(base_result.daily_df)
        hold_sharpe = base_strategy._calculate_hold_sharpe(base_result.daily_df)

        # ============ 5. 创建对比表格 ============
        comparison_data = []
        for r in results:
            excess_return = r['result'].total_return - hold_return
            dd_improve = r['result'].max_drawdown - hold_dd

            comparison_data.append({
                '参数组合': r['参数名称'],
                '总收益率(%)': r['result'].total_return,
                '年化收益率(%)': r['result'].annual_return,
                '最大回撤(%)': r['result'].max_drawdown,
                '夏普比率': r['result'].sharpe_ratio,
                '胜率(%)': r['result'].win_rate,
                '盈亏比': r['result'].profit_loss_ratio,
                '交易次数': r['result'].total_trades,
                '超额收益(%)': excess_return,
                '回撤改善(%)': dd_improve
            })

        # 添加买入持有作为基准
        comparison_data.append({
            '参数组合': '买入持有',
            '总收益率(%)': hold_return,
            '年化收益率(%)': ((base_result.daily_df['买入持有资产'].iloc[-1] / base_params['initial_capital']) ** (
                        1 / (len(base_result.daily_df[~base_result.daily_df['是停牌']]) / 245)) - 1) * 100,
            '最大回撤(%)': hold_dd,
            '夏普比率': hold_sharpe,
            '胜率(%)': 100 if hold_return > 0 else 0,
            '盈亏比': '-',
            '交易次数': 1,
            '超额收益(%)': 0,
            '回撤改善(%)': 0
        })

        comparison = pd.DataFrame(comparison_data)

        # ============ 6. 控制台输出对比表格 ============
        print("\n" + "=" * 100)
        print("RSI参数对比结果".center(100))
        print("=" * 100)

        # 使用tabulate打印表格
        table_data = []
        for _, row in comparison.iterrows():
            table_data.append([
                row['参数组合'],
                f"{row['总收益率(%)']:.2f}%",
                f"{row['最大回撤(%)']:.2f}%",
                f"{row['夏普比率']:.2f}",
                f"{row['超额收益(%)']:+.2f}%" if row['参数组合'] != '买入持有' else '-',
                f"{row['回撤改善(%)']:+.2f}%" if row['参数组合'] != '买入持有' else '-',
                f"{row['交易次数']}" if row['参数组合'] != '买入持有' else '-'
            ])

        headers = ['参数组合', '收益率', '最大回撤', '夏普', '超额收益', '回撤改善', '交易次数']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))

        # 找出最佳参数
        param_results = comparison[comparison['参数组合'] != '买入持有'].copy()
        if not param_results.empty:
            best_sharpe = param_results.loc[param_results['夏普比率'].idxmax()]
            best_return = param_results.loc[param_results['总收益率(%)'].idxmax()]

            print(f"\n🏆 最佳参数（按夏普比率）: {best_sharpe['参数组合']}")
            print(
                f"   夏普比率: {best_sharpe['夏普比率']:.2f}, 收益率: {best_sharpe['总收益率(%)']:.2f}%, 回撤: {best_sharpe['最大回撤(%)']:.2f}%")

            print(f"\n📈 最佳参数（按收益率）: {best_return['参数组合']}")
            print(
                f"   收益率: {best_return['总收益率(%)']:.2f}%, 夏普: {best_return['夏普比率']:.2f}, 回撤: {best_return['最大回撤(%)']:.2f}%")

        # ============ 7. 绘制净值对比曲线 ============
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(14, 8))

        # 绘制买入持有曲线
        ax.plot(base_result.daily_df['日期'], base_result.daily_df['买入持有资产'] / base_params['initial_capital'],
                color='black', linewidth=2, linestyle='--', label='买入持有', alpha=0.7)

        # 为不同参数组合使用不同颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for i, (r, color) in enumerate(zip(results, colors)):
            # 计算净值（归一化到初始资金）
            nav = r['nav']['资产总值'] / base_params['initial_capital']
            ax.plot(r['nav']['日期'], nav, color=color, linewidth=1.5,
                    label=r['参数名称'], alpha=0.8)

        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('净值 (初始资金=1)', fontsize=12)
        ax.set_title(f'不同RSI参数策略净值对比 - {stock_code}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        # 设置日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.tick_params(axis='x', rotation=45)

        # 添加水平线 y=1（初始资金线）
        ax.axhline(y=1, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        plt.tight_layout()

        # 保存图片
        comp_file = base_params['output_dir'] / f'{stock_code}_参数对比_净值曲线.png'
        plt.savefig(comp_file, dpi=300, bbox_inches='tight')
        plt.show()

        # ============ 8. 保存结果 ============
        # 保存比较结果CSV
        comparison_export = comparison.copy()

        # 处理数值列的显示格式
        for col in comparison_export.columns:
            if col == '参数组合':
                continue
            elif col == '夏普比率' or col == '盈亏比' or col == '交易次数':
                comparison_export[col] = comparison_export[col]
            else:
                # 其他数值列/100
                if comparison_export[col].dtype == 'object':
                    # 已经是字符串，不需要转换
                    pass
                else: comparison_export[col] = comparison_export[col] / 100

        # 保存对比表格
        comp_csv = base_params['output_dir'] / f'{stock_code}_参数对比_详细数据.csv'
        comparison_export.to_csv(comp_csv, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 100)
        print("分析完成！".center(100))
        print("=" * 100)
        print(f"📁 详细数据已保存到: {comp_csv}")
        print(f"📊 净值曲线图已保存到: {comp_file}")

        return comparison

