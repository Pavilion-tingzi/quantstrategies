import numpy as np
import pandas as pd
import talib
import chardet
from backtrader.feeds import PandasData
import backtrader as bt
import math


class TechnicalIndicators:
    """技术指标工具类，负责计算ATR、MACD等各类技术指标"""

    @staticmethod
    def calculate_atr(df, period=14, high_col='最高', low_col='最低', close_col='收盘', suspend_col=None):
        """
        计算平均真实波幅（ATR）

        Parameters:
        -----------
        df : pd.DataFrame
            交易数据
        period : int
            ATR周期，默认14
        high_col : str
            最高价列名
        low_col : str
            最低价列名
        close_col : str
            收盘价列名
        suspend_col : str
            停牌标识列名，如果为None则不排除停牌

        Returns:
        --------
        pd.Series : ATR值序列
        """
        df_copy = df.copy()

        # 创建停牌标识
        if suspend_col is None or suspend_col not in df_copy.columns:
            is_suspend = pd.Series(0, index=df_copy.index)
        else:
            is_suspend = df_copy[suspend_col].fillna(0).astype(int)

        # 计算TR
        tr_values = []
        for i in range(len(df_copy)):
            if i == 0:
                tr_values.append(np.nan)
                continue

            # 检查当日是否为停牌
            if is_suspend.iloc[i] == 1:
                tr_values.append(np.nan)
                continue

            high = df_copy.iloc[i][high_col]
            low = df_copy.iloc[i][low_col]

            # 找到前一个非停牌日的收盘价
            prev_close = None
            j = i - 1
            while j >= 0 and prev_close is None:
                if is_suspend.iloc[j] == 0 and not pd.isna(df_copy.iloc[j][close_col]):
                    prev_close = df_copy.iloc[j][close_col]
                j -= 1

            if prev_close is None:
                tr_values.append(np.nan)
                continue

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        df_copy['TR'] = tr_values

        # 计算ATR，排除停牌日
        atr_values = []
        for i in range(len(df_copy)):
            valid_tr = []
            j = i - 1
            search_depth = 0

            while len(valid_tr) < period and j >= 0 and search_depth < 500:
                if is_suspend.iloc[j] == 0 and not pd.isna(df_copy.iloc[j]['TR']):
                    valid_tr.append(df_copy.iloc[j]['TR'])
                j -= 1
                search_depth += 1

            if len(valid_tr) >= period:
                atr_values.append(np.mean(valid_tr))
            else:
                atr_values.append(np.nan)

        return pd.Series(atr_values, index=df_copy.index)

    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9, close_col='收盘', suspend_col=None):
        """
        计算MACD及相关指标

        Returns:
        --------
        tuple : (fast_ema, slow_ema, DIF, DEA) 四个pd.Series
        """
        df_copy = df.copy()

        # 创建停牌标识
        if suspend_col is None or suspend_col not in df_copy.columns:
            is_suspend = pd.Series(0, index=df_copy.index)
        else:
            is_suspend = df_copy[suspend_col].fillna(0).astype(int)

        # 计算EMA（排除停牌日）
        def calculate_ema_excluding_suspend(period):
            ema_values = []
            for i in range(len(df_copy)):
                valid_data = []
                j = i - 1
                search_depth = 0

                while len(valid_data) < period and j >= 0 and search_depth < 500:
                    if is_suspend.iloc[j] == 0 and not pd.isna(df_copy.iloc[j][close_col]):
                        valid_data.append(df_copy.iloc[j][close_col])
                    j -= 1
                    search_depth += 1

                if len(valid_data) < period:
                    ema_values.append(np.nan)
                    continue

                valid_data = valid_data[::-1]
                ema = valid_data[0]
                alpha = 2 / (period + 1)
                for k in range(1, len(valid_data)):
                    ema = valid_data[k] * alpha + ema * (1 - alpha)
                ema_values.append(ema)

            return ema_values

        # 计算快慢EMA
        fast_ema_values = calculate_ema_excluding_suspend(fast)
        slow_ema_values = calculate_ema_excluding_suspend(slow)

        fast_ema = pd.Series(fast_ema_values, index=df_copy.index)
        slow_ema = pd.Series(slow_ema_values, index=df_copy.index)

        # 计算DIF
        dif = fast_ema - slow_ema

        # 计算DEA（前9日DIF的指数移动平均线，不含当日）
        dea_values = []
        for i in range(len(df_copy)):
            if i < signal:
                dea_values.append(np.nan)
                continue

            valid_dif = []
            j = i - 1
            while len(valid_dif) < signal and j >= 0:
                if not pd.isna(dif.iloc[j]):
                    valid_dif.append(dif.iloc[j])
                j -= 1

            if len(valid_dif) < signal:
                dea_values.append(np.nan)
                continue

            valid_dif = valid_dif[::-1]
            dea = valid_dif[0]
            alpha = 2 / (signal + 1)
            for k in range(1, len(valid_dif)):
                dea = valid_dif[k] * alpha + dea * (1 - alpha)
            dea_values.append(dea)

        dea = pd.Series(dea_values, index=df_copy.index)

        return fast_ema, slow_ema, dif, dea

    @staticmethod
    def generate_signals(df, dif_col='DIF', dea_col='DEA', min_interval=5):
        """
        根据DIF和DEA生成交易信号

        Parameters:
        -----------
        df : pd.DataFrame
            交易数据
        dif_col : str
            DIF列名
        dea_col : str
            DEA列名
        min_interval : int
            最小交易间隔

        Returns:
        --------
        pd.Series : 交易信号序列
        """
        signals = pd.Series('无', index=df.index)
        last_signal = None
        last_signal_index = -min_interval

        for i in range(1, len(df)):
            if i < len(df) - 1:
                # 检查金叉
                if (not pd.isna(df.iloc[i - 1][dif_col]) and not pd.isna(df.iloc[i - 1][dea_col]) and
                        not pd.isna(df.iloc[i][dif_col]) and not pd.isna(df.iloc[i][dea_col])):

                    if (df.iloc[i - 1][dif_col] <= df.iloc[i - 1][dea_col] and
                            df.iloc[i][dif_col] > df.iloc[i][dea_col] and
                            df.iloc[i][dif_col] > 0):

                        if (last_signal != '买入' and
                                i - last_signal_index >= min_interval and
                                (last_signal is None or last_signal == '卖出')):
                            signals.iloc[i] = '买入'
                            last_signal = '买入'
                            last_signal_index = i

                    # 检查死叉
                    elif (df.iloc[i - 1][dif_col] >= df.iloc[i - 1][dea_col] and
                          df.iloc[i][dif_col] < df.iloc[i][dea_col]):

                        if last_signal == '买入' and i - last_signal_index >= min_interval:
                            signals.iloc[i] = '卖出'
                            last_signal = '卖出'
                            last_signal_index = i

        return signals

    @staticmethod
    def calculate_ln_e(df, signal_col='交易信号', close_col='收盘', high_col='最高', low_col='最低', window=10):
        """
        计算ln(E)指标，用于衡量信号强度

        Parameters:
        -----------
        df : pd.DataFrame
            交易数据
        signal_col : str
            信号列名
        close_col : str
            收盘价列名
        high_col : str
            最高价列名
        low_col : str
            最低价列名
        window : int
            未来窗口期

        Returns:
        --------
        pd.Series : ln(E)值序列
        """
        ln_e_values = [np.nan] * len(df)

        for i in range(len(df)):
            if df.iloc[i][signal_col] == '买入':
                buy_price = df.iloc[i][close_col]

                end_idx = min(i + window + 1, len(df))
                future_highs = df.iloc[i + 1:end_idx][high_col]
                future_lows = df.iloc[i + 1:end_idx][low_col]

                if len(future_highs) > 0:
                    highest = future_highs.max()
                    lowest = future_lows.min()

                    A = (highest - buy_price) / buy_price if highest > buy_price else 0
                    B = (buy_price - lowest) / buy_price if lowest < buy_price else 0

                    if A > 0 and B > 0:
                        ln_e = np.log(A / B)
                    elif A == 0 and B == 0:
                        ln_e = 0
                    elif B == 0:
                        ln_e = 10
                    elif A == 0:
                        ln_e = -10
                    else:
                        ln_e = 0

                    ln_e = max(-10, min(10, ln_e))
                    ln_e_values[i] = ln_e

        return pd.Series(ln_e_values, index=df.index)

    @staticmethod
    def calculate_moving_average(df, period, close_col='收盘', shift=1, suspend_col=None):
        """
        计算移动平均线（避免未来函数，支持排除停牌）

        Parameters:
        -----------
        df : pd.DataFrame
            交易数据
        period : int
            均线周期
        close_col : str
            收盘价列名，默认'收盘'
        shift : int
            是否向前偏移，默认1（使用前一天数据避免未来函数）
        suspend_col : str
            停牌标识列名，如果为None则不排除停牌

        Returns:
        --------
        pd.Series : 移动平均线序列
        """
        if suspend_col is None or suspend_col not in df.columns:
            # 原有逻辑：不排除停牌
            ma = df[close_col].rolling(window=period).mean()
            if shift > 0:
                ma = ma.shift(shift)
            return ma

        # 排除停牌日的计算
        is_suspend = df[suspend_col].fillna(0).astype(int)
        ma_values = []

        for i in range(len(df)):
            # 如果是停牌日，均线设为NaN
            if is_suspend.iloc[i] == 1:
                ma_values.append(np.nan)
                continue

            # 收集前period个非停牌日的收盘价（不包括当天，避免未来函数）
            valid_prices = []
            j = i - 1  # 从昨天开始往前找
            search_depth = 0
            max_search = 500  # 最大搜索深度

            while len(valid_prices) < period and j >= 0 and search_depth < max_search:
                # 只使用非停牌日的收盘价
                if is_suspend.iloc[j] == 0 and not pd.isna(df.iloc[j][close_col]):
                    valid_prices.append(df.iloc[j][close_col])
                j -= 1
                search_depth += 1

            if len(valid_prices) >= period:
                ma = np.mean(valid_prices)
            else:
                ma = np.nan

            ma_values.append(ma)

        ma_series = pd.Series(ma_values, index=df.index)

        # 注意：因为计算时已经使用了前一天的数据（j = i - 1）
        # 所以不需要再额外shift，否则会多偏移一天
        # 如果调用方需要额外偏移，可以通过参数控制
        if shift > 1:
            ma_series = ma_series.shift(shift - 1)
        elif shift == 0:
            # 如果不需要偏移，说明要用当天数据（不推荐）
            pass
        # shift == 1 时，已经正确使用前一天数据，不需要再偏移

        return ma_series

    @staticmethod
    def detect_golden_death_cross(ma_short, ma_long):
        """
        识别金叉和死叉信号

        Parameters:
        -----------
        ma_short : pd.Series
            短期均线序列
        ma_long : pd.Series
            长期均线序列

        Returns:
        --------
        tuple : (golden_cross, death_cross) 两个布尔型Series
        """
        golden_cross = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
        death_cross = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))
        return golden_cross, death_cross

    @staticmethod
    def filter_signals_with_interval(df, signal_col='raw_signal',
                                     min_interval=5, first_must_buy=True):
        """
        过滤交易信号：首笔必须买入、买卖交替、最小间隔

        Parameters:
        -----------
        df : pd.DataFrame
            交易数据
        signal_col : str
            原始信号列名
        min_interval : int
            最小信号间隔天数
        first_must_buy : bool
            首笔交易是否必须为买入

        Returns:
        --------
        pd.Series : 过滤后的信号序列
        """
        signals = pd.Series('', index=df.index)

        # 获取所有原始信号
        signal_indices = []
        signal_types = []

        for i in range(len(df)):
            if df.iloc[i][signal_col] in ['buy', 'sell']:
                signal_indices.append(i)
                signal_types.append(df.iloc[i][signal_col])

        if not signal_indices:
            return signals

        # 找到第一个买入信号的位置
        first_buy_idx = None
        if first_must_buy:
            for i, signal_type in enumerate(signal_types):
                if signal_type == 'buy':
                    first_buy_idx = i
                    break
        else:
            first_buy_idx = 0

        if first_buy_idx is None:
            return signals

        # 从第一个信号开始筛选
        selected_indices = [signal_indices[first_buy_idx]]
        selected_types = [signal_types[first_buy_idx]]
        last_idx = signal_indices[first_buy_idx]
        last_type = signal_types[first_buy_idx]

        # 遍历后续信号
        for i in range(first_buy_idx + 1, len(signal_indices)):
            current_idx = signal_indices[i]
            current_type = signal_types[i]

            # 检查信号类型是否交替
            if current_type == last_type:
                continue

            # 检查时间间隔
            if current_idx - last_idx < min_interval:
                continue

            # 通过检查，接受该信号
            selected_indices.append(current_idx)
            selected_types.append(current_type)
            last_idx = current_idx
            last_type = current_type

        # 写入信号
        for idx, signal_type in zip(selected_indices, selected_types):
            signals.iloc[idx] = signal_type

        return signals

    @staticmethod
    def calculate_rsi(df, close_col='收盘',suspend_col="is_suspend", period=14):
        """
           计算RSI，自动排除停牌日数据

           Parameters:
           -----------
           df : pd.DataFrame
               原始数据，必须包含日期索引或按时间排序
           close_col : str
               收盘价列名
           suspend_col : str
               停牌标记列名（1表示停牌，0表示交易）
           period : int
               RSI周期

           Returns:
           --------
           pd.DataFrame : 添加了'rsi'列的原始DataFrame
           """
        # 确保数据按时间排序
        if '日期' in df.columns:
            df = df.sort_values('日期')
        elif df.index.name == '日期' or isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        is_suspend = df[suspend_col].fillna(0).astype(int)
        df_close_unsuspend = df.loc[is_suspend == 0, close_col]
        rsi_values = talib.RSI(df_close_unsuspend.values, timeperiod=period)
        rsi_series = pd.Series(rsi_values, index=df_close_unsuspend.index)

        return rsi_series

    @staticmethod
    def generate_signals_rsi(df, rsi_col='RSI', suspend_col="is_suspend",oversold_threshold=30,overbought_threshold=70):
        """生成交易信号"""
        df = df.copy()
        # 判断RSI的位置状态
        df['RSI_上穿'] = (df[rsi_col] >= oversold_threshold) & (
                df[rsi_col].shift(1) < oversold_threshold)
        df['RSI_下穿'] = (df[rsi_col] <= overbought_threshold) & (
                df[rsi_col].shift(1) > overbought_threshold)

        df['交易信号'] =''

        position = 0

        for index, row in df.iterrows():
            if row[suspend_col] == 1:
                df.loc[index,'持仓状态'] = position
                continue
            if row['RSI_上穿'] and position == 0:
                df.loc[index,'交易信号'] = '买入'
                position = 1
            elif row['RSI_下穿'] and position ==1:
                df.loc[index,'交易信号'] = '卖出'
                position = 0
            df.loc[index,'持仓状态'] = position

        return df['交易信号']

    @staticmethod
    def calculate_boll(df, close_col='收盘',suspend_col="is_suspend", period=20):
        """
        计算BOLL，自动排除停牌日数据

        Parameters:
        -----------
        df : pd.DataFrame
            原始数据，必须包含日期索引或按时间排序
        close_col : str
            收盘价列名
        suspend_col : str
            停牌标记列名（1表示停牌，0表示交易）
        period : int
            BOLL周期

        Returns:
        --------
        pd.DataFrame : 添加了'BOLL_Upper'、'BOLL_Middle'、'BOLL_Lower'列的DataFrame
        """
        if '日期' in df.columns:
            df = df.sort_values('日期')
        elif df.index.name == '日期' or isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        is_suspend = df[suspend_col].fillna(0).astype(int)
        df_close_unsuspend = df.loc[is_suspend == 0, close_col]

        BB_Upper, BB_Middle, BB_Lower = talib.BBANDS(df_close_unsuspend.values, timeperiod=period)
        BB_Upper_Series = pd.Series(BB_Upper, index=df_close_unsuspend.index)
        BB_Middle_Series = pd.Series(BB_Middle, index=df_close_unsuspend.index)
        BB_Lower_Series = pd.Series(BB_Lower, index=df_close_unsuspend.index)

        return pd.DataFrame({
        'BB_Upper_Series': BB_Upper_Series,
        'BB_Middle_Series': BB_Middle_Series,
        'BB_Lower_Series': BB_Lower_Series
    })

class RiskManage:
    """风控管理类，负责资金与风险控制"""

    def __init__(self, risk_percent=0.02, add_ratios=[0.4, 0.3, 0.3],
                 add_atr_multiple=1, stop_atr_multiple=2):
        """
        初始化风险管理器

        Parameters:
        -----------
        risk_percent : float
            风险预算比率，默认2%
        add_ratios : list
            加仓比率分配列表，第一个元素是建仓比例，后面是每次加仓的比例
            例如：[0.4, 0.3, 0.3] 表示建仓40%，第一次加仓30%，第二次加仓30%
                  [0.6, 0.4] 表示建仓60%，第一次加仓40%（只有一次加仓）
                  [0.5] 表示建仓50%，不加仓
        add_atr_multiple : float
            加仓间隔(ATR倍数)，默认1
        stop_atr_multiple : float
            止损/止盈设置(ATR倍数)，默认2
        """
        self.risk_percent = risk_percent
        self.add_atr_multiple = add_atr_multiple
        self.stop_atr_multiple = stop_atr_multiple

        # 存储加仓比例列表
        self.add_ratios = add_ratios
        self.add_count_total = len(add_ratios) - 1  # 加仓次数 = 比例数 - 1

        # 验证比例总和为1
        total_ratio = sum(add_ratios)
        if abs(total_ratio - 1.0) > 0.0001:
            raise ValueError(f"加仓比例总和应为1，当前为{total_ratio}")

        # 风险状态
        self.reset_position()

    def reset_position(self):
        """重置状态"""
        self.position_status = 0
        self.add_count = 0
        self.avg_cost = 0.0
        self.stop_loss = 0.0
        self.stop_profit = 0.0
        self.highest_close = 0.0
        self.initial_atr = 0.0
        self.last_buy_price = 0.0
        self.add_shares_list = []  # 各批次股数列表
        self.add_count_total = 0  # 总加仓次数

    def calculate_position_size(self, price, atr_value, cash_available):
        """
        计算建仓股数

        Returns:
        --------
        dict : 包含建仓股数、止损价、止盈价等信息的字典
        """
        if atr_value <= 0:
            return {'shares': 0, 'stop_loss': 0, 'stop_profit': 0,
                    'add_shares_list': []}

        # 基于初始总资金计算总风险仓位
        total_risk_amount = cash_available * self.risk_percent
        total_max_shares = total_risk_amount / (atr_value * self.stop_atr_multiple)
        total_max_shares_int = int(total_max_shares)

        # 按比例计算各次买入的股数
        add_shares_list = []
        for ratio in self.add_ratios:
            shares = int(total_max_shares_int * ratio/100)*100
            if shares < 100:
                shares = 100
            add_shares_list.append(shares)

        # 建仓股数是第一个
        base_shares = add_shares_list[0]

        # 检查建仓资金是否足够
        while base_shares > 0:
            cost = base_shares * price
            if cost <= cash_available * 0.95:
                break
            base_shares -= 100
            # 同时调整后续加仓股数（按比例调整）
            if base_shares > 0:
                scale_factor = base_shares / add_shares_list[0]
                add_shares_list = [int(s * scale_factor) for s in add_shares_list]
                base_shares = add_shares_list[0]

        if base_shares <= 0:
            return {'shares': 0, 'stop_loss': 0, 'stop_profit': 0,
                    'add_shares_list': []}

        stop_price = price - (atr_value * self.stop_atr_multiple)

        return {
            'shares': base_shares,
            'stop_loss': stop_price,
            'stop_profit': stop_price,
            'add_shares_list': add_shares_list,  # 返回所有批次的股数列表
            'atr_value': atr_value
        }

    def calculate_add_position_size(self, price, current_shares, cash_available):
        """
        计算加仓股数
        """
        # 检查加仓次数是否已达上限
        if self.add_count >= self.add_count_total:
            return {'shares': 0, 'new_stop_loss': self.stop_loss}

        # 获取本次加仓的股数（第add_count+1次加仓，对应列表索引add_count+1）
        shares_to_buy = self.add_shares_list[self.add_count + 1]

        if shares_to_buy <= 0:
            return {'shares': 0, 'new_stop_loss': self.stop_loss}

        # 检查资金是否足够
        while shares_to_buy > 0:
            cost = shares_to_buy * price
            if cost <= cash_available:
                break
            shares_to_buy -= 100

        if shares_to_buy <= 0:
            return {'shares': 0, 'new_stop_loss': self.stop_loss}

        # 更新止损线（使用建仓时的ATR）
        new_stop_loss = max(self.stop_loss,
                            (self.avg_cost * current_shares + price * shares_to_buy) /
                            (current_shares + shares_to_buy) -
                            (self.initial_atr * self.stop_atr_multiple))

        return {
            'shares': shares_to_buy,
            'new_stop_loss': new_stop_loss
        }

    def get_next_add_price(self):
        """计算下一次加仓的目标价格"""
        if self.last_buy_price > 0 and self.initial_atr > 0:
            return self.last_buy_price + (self.initial_atr * self.add_atr_multiple)
        return None

    def check_stop(self, prev_close=None, current_atr=None):
        if self.position_status == 0:
            return False, None

        # 更新最高收盘价
        if prev_close > self.highest_close:
            self.highest_close = prev_close

        # 使用前一日ATR计算止盈线（注意：current_atr应该传入前一日ATR）
        new_stop_profit = self.highest_close - (current_atr * self.stop_atr_multiple)

        # 止盈线只能上移，不能下移
        if new_stop_profit > self.stop_profit:
            self.stop_profit = new_stop_profit
        # 使用前一日收盘价判断止损止盈（避免未来函数）
        if prev_close is not None:
            if self.stop_loss > 0 and prev_close <= self.stop_loss:
                return True, '止损'
            if self.stop_profit > 0 and prev_close <= self.stop_profit:
                return True, '止盈'
        else:
            print("第一行数据，暂不计算")

        return False, None

    def update_risk_state(self, action, **kwargs):
        """
        更新风险状态
        """
        if action == 'open':
            self.position_status = 1
            self.add_count = 0  # 已经加仓的次数，初始为0
            self.avg_cost = kwargs.get('price', 0)
            self.stop_loss = kwargs.get('stop_loss', 0)
            self.stop_profit = kwargs.get('stop_profit', 0)
            self.highest_close = kwargs.get('price', 0)
            self.initial_atr = kwargs.get('atr_value', 0)
            self.last_buy_price = kwargs.get('price', 0)
            self.add_shares_list = kwargs.get('add_shares_list', [])  # 存储各批次的股数
            self.add_count_total = len(self.add_shares_list) - 1  # 总加仓次数

        elif action == 'add':
            self.add_count += 1
            old_cost = self.avg_cost
            old_shares = kwargs.get('old_shares', 0)
            new_shares = kwargs.get('new_shares', 0)
            new_price = kwargs.get('price', 0)
            self.avg_cost = (old_cost * old_shares + new_price * new_shares) / (old_shares + new_shares)
            self.last_buy_price = new_price
            if 'new_stop_loss' in kwargs:
                self.stop_loss = kwargs['new_stop_loss']

        elif action == 'close':
            self.reset_position()

    def get_risk_state(self):
        """获取当前风险状态快照"""
        return {
            'position_status': self.position_status,
            'add_count': self.add_count,
            'add_count_total': self.add_count_total,  # 添加总加仓次数
            'avg_cost': self.avg_cost,
            'stop_loss': self.stop_loss,
            'stop_profit': self.stop_profit,
            'highest_close': self.highest_close,
            'initial_atr': self.initial_atr,
            'last_buy_price': self.last_buy_price
        }

    def is_can_add(self, prev_close):
        """
        判断是否可以加仓

        Parameters:
        -----------
        price : float
            当前价格
        prev_close : float
            前一日收盘价

        Returns:
        --------
        bool : 是否可以加仓
        """
        if self.position_status == 0:
            return False
        if self.add_count >= self.add_count_total:
            return False
        if self.last_buy_price <= 0 or self.initial_atr <= 0:
            return False

        next_add_price = self.get_next_add_price()
        if next_add_price is None:
            return False

        # 使用前一日收盘价判断
        return prev_close is not None and prev_close >= next_add_price

    def get_risk_metrics(self):
        """获取当前风险指标"""
        return {
            'risk_percent': self.risk_percent,
            'add_atr_multiple': self.add_atr_multiple,
            'stop_atr_multiple': self.stop_atr_multiple,
            'add_count': self.add_count,
            'add_ratios': self.add_ratios
        }

class BaseTool:
    @staticmethod
    def LoadAsPandasData(df,data_dict=None):
        # 默认配置
        default_dict = {
            'open': '开盘',
            'high': '最高',
            'low': '最低',
            'close': '收盘',
            'volume': '成交量',
            'datetime': '日期'
        }

        if data_dict is None:
            final_dict = default_dict
        else:
            final_dict = {**default_dict, **data_dict}

        # 日期格式转化
        df[final_dict.get('datetime')] = pd.to_datetime(df[final_dict.get('datetime')])

        # 动态构建参数字符串
        class MyPandasData(PandasData):
            # lines 不包含 datetime
            lines = tuple([k for k in final_dict.keys() if k not in ['datetime','close','low','high','open','volume','openinterest']])
            # params 包含所有映射（包括 datetime）
            params = tuple([(k, v) for k, v in final_dict.items()])

        return MyPandasData(dataname=df)

class MyCommInfo(bt.CommInfoBase):
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),  # 按百分比收取
        ('perc', 0.0001),  # 佣金比率 0.01% (万分之一)
        ('min_commission', 5.0),  # 最低佣金 5 元
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        size: 交易数量
        price: 交易价格
        pseudoexec: 是否为虚拟执行（用于计算最小佣金等）
        """
        # 计算按比率收取的佣金
        perc_commission = abs(size * price * self.p.perc)
        # 取最大值：百分比佣金 和 最低佣金 5 元
        commission = max(perc_commission, self.p.min_commission)
        return commission

class MyRMSizer(bt.Sizer):
    params = (
        ('risk_percent', 0.02),  # 风险预算比率
        ('add_ratios', [0.4, 0.3, 0.3]),  # 加仓比率分配
        ('add_atr_multiple', 1.0),  # 加仓间隔(ATR倍数)
        ('stop_atr_multiple', 2.0),  # 止损/止盈(ATR倍数)
    )

    def __init__(self):
        self.risk_mgr = RiskManage(risk_percent=self.p.risk_percent, add_ratios=self.p.add_ratios,
                                   add_atr_multiple=self.p.add_atr_multiple, stop_atr_multiple=self.p.stop_atr_multiple)
        self._pending_signal = None
        self._has_pending_order = False  #是否有挂单

    def _getsizing(self, comminfo, cash, data, isbuy):
        current_pos = self.broker.getposition(data).size
        risk_state = self.risk_mgr.get_risk_state()
        if self._has_pending_order:
            return 0

        # ----- 清仓 -----
        if not isbuy:
            if risk_state['position_status'] == 1:
                self._pending_signal = {'action': 'close'}
                self._has_pending_order = True
            return current_pos

        # ----- 建仓（首次买入）-----
        if risk_state['position_status'] == 0:
            pos_info = self.risk_mgr.calculate_position_size(
                price=data.close[0],
                atr_value=data.atr[0],
                cash_available=cash
            )

            if pos_info['shares'] <= 0:
                return 0

            # 保存信号信息，等待订单成交后更新状态
            self._pending_signal = {
                'action': 'open',
                'price': data.close[0],
                'stop_loss': pos_info['stop_loss'],
                'stop_profit': pos_info['stop_profit'],
                'atr_value': pos_info['atr_value'],
                'add_shares_list': pos_info['add_shares_list']
            }
            self._has_pending_order = True

            return pos_info['shares']

        # ----- 加仓（已有持仓）-----
        elif risk_state['position_status'] == 1:
            if self.risk_mgr.is_can_add(data.close[0]):
                add_info = self.risk_mgr.calculate_add_position_size(
                    price=data.close[0],
                    current_shares=current_pos,
                    cash_available=cash
                )

                if add_info['shares'] <= 0:
                    return 0

                self._pending_signal = {
                    'action': 'add',
                    'price': data.close[0],
                    'old_shares': current_pos,
                    'new_shares': add_info['shares'],
                    'new_stop_loss': add_info['new_stop_loss']
                }
                self._has_pending_order = True

                return add_info['shares']

        return 0

    def notify_order(self, order):
        if order.status in [order.Canceled, order.Rejected, order.Margin]:
            self._has_pending_order = False

        if order.status == order.Completed:
            self.risk_mgr.update_risk_state(**self._pending_signal)
            self._has_pending_order = False

    def is_can_add(self,data):
        return self.risk_mgr.is_can_add(data.close[0])

    def check_stop(self,data):
        return self.risk_mgr.check_stop(prev_close=data.close[0], current_atr=data.atr[0])

    def get_risk_state(self):
        return self.risk_mgr.get_risk_state()

class MyAllInSizer(bt.Sizer):
    params = (('percents', 95),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            size = cash / data.close[0] * (self.p.percents / 100.0)
            # 取整到100
            size = int(size / 100) * 100
            return max(size, 100) if size > 0 else 0
        else:
            return self.broker.getposition(data).size

class MyBroker(bt.brokers.BackBroker):
    def __init__(self, stamp_tax_rate=0.001):
        super().__init__()
        self.stamp_tax_rate = stamp_tax_rate
        self.total_stamp_tax = 0
        self._current_stamp_tax = 0  # 当前交易的印花税

    def submit(self, order, check=True):
        """在订单提交时就设置 stamp_tax 属性"""
        order.stamp_tax = None
        result = super().submit(order, check)
        return result

    def _execute(self, order, ago=None, price=None, cash=None, position=None,
                 dtcoc=None):
        # 执行原订单
        result = super()._execute(order, ago, price, cash, position, dtcoc)
        # 如果是卖出且订单完成，计算并扣除印花税
        if (ago is not None and
                order.status == order.Completed and
                not order.isbuy()):
            # 计算印花税
            stamp_tax = abs(order.executed.size) * order.executed.price * self.stamp_tax_rate

            # 扣除印花税
            self.cash -= stamp_tax
            self.total_stamp_tax += stamp_tax
            self._current_stamp_tax += stamp_tax

            # 记录到订单
            order.stamp_tax = stamp_tax

        elif(ago is not None):
            order.stamp_tax = 0
        else:
            return result

        self.notify(order)

        return result

    def get_total_stamp_tax(self):
        return self.total_stamp_tax

    def notify(self,order):
        if order.stamp_tax is None:
            return
        super().notify(order)

class SignalEffectiveness(bt.Analyzer):
    """买入信号有效性分析器"""

    params = (('lookforward', 10),)

    def __init__(self):
        self.lnEs = []
        self.buy_price = None
        self.everyday_price = []
        self.day_count = None

    def notify_order(self, order):
        if order.status == order.Completed and order.info.get('trade_signal', 'unknown') == 1:
            self.buy_price = order.executed.price
            self.day_count = 0

    def next(self):
        if self.day_count == 10:
            highest = max(self.everyday_price)
            lowest = min(self.everyday_price)
            A = max((highest - self.buy_price) / self.buy_price, 0)
            B = max((self.buy_price - lowest) / self.buy_price, 0)

            if B == 0:
                lnE = 10 if A > 0 else 0
            elif A == 0:
                lnE = -10
            else:
                lnE = math.log(A / B)

            self.lnEs.append(lnE)
            self.buy_price = None
            self.everyday_price = []
            self.day_count = None

            return

        if self.buy_price:
            self.day_count += 1
            self.everyday_price.append(self.datas[0].close[0])

    def get_analysis(self):
        if not self.lnEs:
            return {'avg_lnE': 0}
        return {'avg_lnE': sum(self.lnEs) / len(self.lnEs)}

class MyObserver(bt.Observer):
    alias = ('MyObserver',)
    lines = ('tradesignal',)
    plotinfo = dict(plot=True, subplot=True)

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.signal = 0

    def next(self):
        self.lines.tradesignal[0] = self.signal
        self.signal = 0

    def notify_order(self,order):
        self.signal = order.info.get('trade_signal', 0)
