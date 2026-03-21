import strategies
import pandas as pd

if __name__ == '__main__':
    # ==================== 双均线策略 ====================
    # 【双均线策略】单个股票
    '''
    strategy = strategies.DoubleMovingAverageStrategy(short_ma=10, long_ma=120)
    strategy.run_complete_analysis(
        filepath="df_pre_603993_停牌18_价格异常7_最大异常涨跌幅15.30%.csv",
        output_folder="./ma_results/single_stock"
    )
    '''

    # 【双均线策略】多股票对比
    '''
    stock_files = "./data_dealed"
    comparison_df = strategies.DoubleMovingAverageStrategy.compare_stocks(
        file_list=stock_files,
        output_folder="./ma_results/stock_comparison",
        short_ma=10,
        long_ma=120
    )
    '''

    # 【双均线策略】单股票多参数
    '''
    # 定义要测试的均线参数组合
    ma_combinations = [
        (5, 20),
        (5, 60),
        (10, 60),
        (10, 120),
        (20, 120),
        (20, 250)
    ]

    param_comparison_df = strategies.DoubleMovingAverageStrategy.compare_strategies(
        filepath="df_pre_603993_停牌18_价格异常7_最大异常涨跌幅15.30%.csv",
        ma_pairs=ma_combinations,
        output_folder="./ma_results/param_comparison"
    )
    '''

    # ==================== RSI策略 ====================
    # 【RSI策略】单个股票
    '''
    strategy = strategies.RSIStrategy(
        initial_capital=1000000,
        commission_rate=0.0001,
        stamp_tax_rate=0.001,
        min_commission=5,
        stop_loss=0.1,  # 10%止损
        risk_free_rate=0.03,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70
    )
    result = strategy.run_complete_analysis(
        file_path="df_pre_603993_停牌18_价格异常7_最大异常涨跌幅15.30%.csv",
        output_dir="./rsi_results/single_stock"
    )
    '''

    # 【RSI策略】多股票对比
    '''
    comparison = strategies.RSIStrategy.compare_stocks(
        input_source="./data_dealed",
        initial_capital=1000000,
        commission_rate=0.0001,
        stamp_tax_rate=0.001,
        min_commission=5,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        output_dir=f"./rsi_results/stocks_comparison"
    )
    '''

    # 【RSI策略】单股票多策略对比
    '''
    param_combinations = [
        (9, 25, 75),  # 短期RSI，激进
        (14, 30, 70),  # 标准RSI
        (21, 35, 65),  # 长期RSI，保守
        (7, 20, 80),  # 超短期RSI，非常激进
    ]

    param_comparison = strategies.RSIStrategy.compare_strategies(
        file_path="df_pre_603993_停牌18_价格异常7_最大异常涨跌幅15.30%.csv",
        param_combinations=param_combinations,
        initial_capital=1000000,
        commission_rate=0.0001,
        stamp_tax_rate=0.001,
        min_commission=5,
        output_dir="./rsi_results/param_comparison"
    )
    '''

    # ==================== 布林带策略 ====================
    # 【boll策略】单股票
    '''
    strategy = strategies.BollStrategy(
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
    strategy.run_complete_analysis("./data_test/df_pre_600383_价格异常6_最大异常涨跌幅11.96%.csv", "./boll_results/single_stock/600383")
    '''

    # 【boll策略】多股票对比
    '''
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
    df_comparison = strategies.BollStrategy.compare_stocks("./data_dealed", params, "./boll_results/stocks_comparison")
    '''

    # 【boll策略】单股票多参数对比
    #'''
    param_combinations = [
         {'boll_period': 10, 'stop_loss_rate': 0.05},
         {'boll_period': 20, 'stop_loss_rate': 0.05},
         {'boll_period': 30, 'stop_loss_rate': 0.05},
         {'boll_period': 20, 'stop_loss_rate': 0.1},
         {'boll_period': 20, 'stop_loss_rate': 1.0},
    ]
    df_params = strategies.BollStrategy.compare_strategies("./data_test/df_pre_600383_价格异常6_最大异常涨跌幅11.96%.csv", param_combinations, "./boll_results/param_comparison/600383")
    #'''

    # ==================== 多策略回测结果比较 ====================
    # 示例1：使用文件夹路径
    '''
    comparator = strategies.StrategyCompare(
        input_path="./compare_strategies/strategy_results",
        output_dir="./compare_strategies"
    )
    '''

    # 示例2：使用文件列表
    '''
    comparator = StrategyCompare(
        input_path=["./strategy1.xlsx", "./strategy2.xlsx", "./strategy3.xlsx"],
        output_dir="./analysis_results"
    )
    '''

    # 运行完整分析
    # results = comparator.run_full_analysis()