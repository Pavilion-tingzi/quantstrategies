import strategies
import tools


if __name__ == '__main__':
    # 创建风险管理实例
    RiskManager = tools.RiskManage(risk_percent=0.02, add_ratios=[0.4, 0.3, 0.3], add_atr_multiple=1, stop_atr_multiple=2)

    # ==================== 双均线策略 ====================
    # 【双均线策略】单个股票
    '''
    strategy = strategies.DoubleMovingAverageStrategy(short_ma=10, long_ma=120)
    strategy.run_complete_analysis(
        filepath="df_pre_002714_停牌9_价格异常4_最大异常涨跌幅12.37%.csv",
        output_folder="./ma_results1/single_stock",
        risk_manager=RiskManager
    )
    '''

    # 【双均线策略】多股票对比
    '''
    stock_files = "./data_test"
    comparison_df = strategies.DoubleMovingAverageStrategy.compare_stocks(
        file_list=stock_files,
        output_folder="./ma_results1/stock_comparison",
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
    strategy.run_complete_analysis(
        'df_pre_002714_停牌9_价格异常4_最大异常涨跌幅12.37%.csv',
        './test',
        risk_manager=RiskManager)
    '''

    # 【RSI策略】多股票对比
    '''
    comparison = strategies.RSIStrategy.compare_stocks(
        input_source="./data_test",
        initial_capital=1000000,
        commission_rate=0.0001,
        stamp_tax_rate=0.001,
        min_commission=5,
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        output_dir=f"./test/stocks_with_riskmng",
        risk_manager = RiskManager
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
        data_path='df_pre_002714_停牌9_价格异常4_最大异常涨跌幅12.37%.csv',
        min_commission=5, commission=0.0001,
        risk_percent=0.02, add_ratios=[0.4, 0.3, 0.3], add_atr_multiple=1.0,
        stop_atr_multiple=2.0, initial_capital=1000000
    )
    strategy.run_complete_analysis('./boll_results/single_stock', use_risk_manager=True, plot=True)
    '''
    # 【boll策略】多股票对比
    #'''
    strategies.BollStrategy.compare_stocks(input_source='./data_test',min_commission=5, commission=0.0001, stamp_tax_rate=0.001,
                       risk_percent=0.02, add_ratios=[0.4, 0.3, 0.3], add_atr_multiple=1.0,
                       stop_atr_multiple=2.0, initial_capital=1000000, risk_free_rate=0.03,
                       boll_period=20, atr_period=14, output_dir = "./boll_results/stock_comparison",
                       use_risk_manager=True)
    #'''
    # ==================== MACD策略 ====================
    # 单只股票完整分析
    '''
    strategy = strategies.MACDStrategy(initial_capital=1000000)
    strategy.run_complete_analysis(
        'df_pre_002714_停牌9_价格异常4_最大异常涨跌幅12.37%.csv',
        './macd_results1/single_stock',
        risk_manager=RiskManager)
    '''

    # 多股票对比
    '''
    strategies.MACDStrategy.compare_stocks(
        files_or_folder='data_test',
        output_folder='./macd_results1/stocks_comparison',
        save_detail=True,
        risk_manager=RiskManager
    )
    '''

    # 单股票多参数对比
    '''
    # 定义要测试的参数组合列表
    param_combinations = [
        # 组合1: 默认参数
        {},
        # 组合2: 修改MACD参数
        {'fast': 8, 'slow': 17, 'signal': 9},
        # 组合3: 修改风险管理参数
        {'risk_percent': 0.03, 'add_ratios': [0.5, 0.3, 0.2]},
        # 组合4: 修改加仓间隔
        {'add_atr_multiple': 1.5, 'stop_atr_multiple': 2.5},
        # 组合5: 修改ATR周期
        {'atr_period': 10},
        # 组合6: 综合修改
        {
            'fast': 10,
            'slow': 20,
            'signal': 8,
            'risk_percent': 0.025,
            'add_ratios': [0.4, 0.35, 0.25],
            'add_atr_multiple': 1.2,
            'stop_atr_multiple': 2.2,
            'atr_period': 12
        }
    ]

    # 运行对比分析
    strategies.MACDStrategy.compare_strategies(
        file_path='df_pre_002714_停牌9_价格异常4_最大异常涨跌幅12.37%.csv',
        param_combinations=param_combinations,
        output_folder='./macd_results/param_comparison'
    )
    '''

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