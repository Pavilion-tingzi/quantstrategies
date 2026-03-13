import strategies
import pandas as pd

if __name__ == '__main__':


    # 多股票双均线策略
    #'''
    compare_stocks_results = strategies.DoubleMovingAverageStrategy.compare_stocks('./data_dealed', short_ma=10, long_ma=120, encoding='utf-8',
                       initial_capital=1000000, commission_rate=0.0001,
                       min_commission=5, stamp_tax_rate=0.001, risk_free_rate=0.02)

    # 打印对比结果总结
    strategies.DoubleMovingAverageStrategy.print_comparison_summary(compare_stocks_results)
    strategies.DoubleMovingAverageStrategy.plot_stock_comparison(compare_stocks_results,top_n=300)
    #'''

    # 单只股票双均线策略图表展示
    '''
    strategy = strategies.DoubleMovingAverageStrategy(short_ma=10, long_ma=120)
    strategy.run_complete_analysis('./data/df_pre_002385.csv', encoding='gbk')

    # 查看结果
    strategy.print_results()
    strategy.plot_results()
    '''

    # 同一只股票不同参数双均线策略
    '''
    ma_pairs = [(10, 60), (10, 120), (90, 120)]
    compare_strategies_results = strategies.DoubleMovingAverageStrategy.compare_strategies('./data/df_pre_002450.csv', ma_pairs, encoding='gbk')

    strategies.DoubleMovingAverageStrategy.print_comparison_summary(compare_strategies_results)
    strategies.DoubleMovingAverageStrategy.plot_strategies_comparison(compare_strategies_results,filepath='./data/df_pre_002450.csv')
    '''

    # 单只股票RSI策略分析
    '''
    # 创建策略实例
    strategy = strategies.RSIStrategy(
        name="RSI择时策略",
        initial_capital=1000000,
        commission_rate=0.0001,
        min_commission=5,
        stamp_tax_rate=0.001,
        buy_threshold=30,
        sell_threshold=70,
        rsi_period=14,
        min_interval_days=5,
        output_dir="./rsi_strategy_output"
    )
    # 进行回撤
    result = strategy.run_complete_analysis(
        file_path='./data_dealed/df_pre_600036_价格异常21_最大异常涨跌幅28.72%.csv',
    )
    '''

    # 多只股票RSI策略分析
    '''
    strategies.RSIStrategy.compare_stocks(
        input_source="./data_test",
        initial_capital=1000000,  # 使用100万初始资金
        rsi_period=14,  # 使用14日RSI
        buy_threshold=30,  # 买入阈值30
        sell_threshold=70,  # 卖出阈值70
        output_dir="./rsi_strategy_output2",
        save_results=True, #单个股票结果是否展示和保存
        plot=True, #单个股票图表是否展示和保存
    )
    '''

    # 单只股票多参数RSI策略分析
    '''
    custom_params = [
        [9, 20, 80],  # 激进型：短期RSI，宽阈值
        [14, 30, 70],  # 标准型
        [21, 40, 60],  # 保守型：长期RSI，窄阈值
        [14, 25, 75],  # 标准周期，宽阈值
        [21, 35, 65],  # 长期周期，中等阈值
    ]

    result = strategies.RSIStrategy.compare_parameters(
        file_dir="./data_test/df_pre_000750.csv",
        rsi_parameters=custom_params,
        initial_capital=1000000,
        output_dir="./rsi_param_comparison"
    )
    '''