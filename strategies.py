import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


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

