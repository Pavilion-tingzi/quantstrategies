import strategies

if __name__ == '__main__':


    # 比较多股票策略
    '''
    compare_stocks_results = strategies.DoubleMovingAverageStrategy.compare_stocks('./data_1', short_ma=10, long_ma=120, encoding='gbk',
                       initial_capital=1000000, commission_rate=0.0001,
                       min_commission=5, stamp_tax_rate=0.001, risk_free_rate=0.02)

    # 打印对比结果总结
    strategies.DoubleMovingAverageStrategy.print_comparison_summary(compare_stocks_results)
    strategies.DoubleMovingAverageStrategy.plot_stock_comparison(compare_stocks_results,top_n=300)
    '''

    # 单只股票策略图表展示
    #'''
    strategy = strategies.DoubleMovingAverageStrategy(short_ma=10, long_ma=120)
    strategy.run_complete_analysis('./data/df_pre_002385.csv', encoding='gbk')

    # 查看结果
    strategy.print_results()
    strategy.plot_results()
    #'''

    # 同一只股票不同参数双均线
    '''
    ma_pairs = [(10, 60), (10, 120), (90, 120)]
    compare_strategies_results = strategies.DoubleMovingAverageStrategy.compare_strategies('./data/df_pre_002450.csv', ma_pairs, encoding='gbk')

    strategies.DoubleMovingAverageStrategy.print_comparison_summary(compare_strategies_results)
    strategies.DoubleMovingAverageStrategy.plot_strategies_comparison(compare_strategies_results,filepath='./data/df_pre_002450.csv')
    '''