import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

# 设置中文字体支持
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data():
    """加载两个策略的数据"""
    ma_df = pd.read_csv('ma_compare_stocks.csv', encoding='gbk')
    rsi_df = pd.read_csv('rsi_compare_stocks.csv', encoding='gbk')
    
    # 处理 RSI 数据的回撤率，转换为绝对值
    if '策略最大回撤' in rsi_df.columns:
        rsi_df['策略最大回撤'] = rsi_df['策略最大回撤'].abs()
    
    return ma_df, rsi_df

def compare_strategies(ma_df, rsi_df):
    """比较两个策略的各项指标"""
    # 定义需要比较的指标（注意两个 CSV 文件的列名差异）
    # MA 策略列名：策略_总收益率，RSI 策略列名：策略总收益率
    metrics = {
        '总收益率': ('策略_总收益率', '策略总收益率'),
        '年化收益率': ('策略_年化收益率', '策略年化收益率'),
        '最大回撤': ('策略_最大回撤', '策略最大回撤'),
        '夏普比率': ('策略_夏普比率', '策略夏普比率'),
        '胜率': ('策略_胜率', '策略胜率'),
        '盈亏比': ('策略_盈亏比', '策略盈亏比')
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("双均线策略 vs RSI 策略对比分析")
    print("="*80)
    
    for metric_name, (ma_col, rsi_col) in metrics.items():
        if ma_col in ma_df.columns and rsi_col in rsi_df.columns:
            ma_values = ma_df[ma_col].dropna()
            rsi_values = rsi_df[rsi_col].dropna()
            
            # 计算统计指标
            stats = {
                '平均数': [ma_values.mean(), rsi_values.mean()],
                '最大值': [ma_values.max(), rsi_values.max()],
                '最小值': [ma_values.min(), rsi_values.min()],
                '中位数': [ma_values.median(), rsi_values.median()],
                '标准差': [ma_values.std(), rsi_values.std()]
            }
            
            results[metric_name] = stats
            
            # 打印结果
            print(f"\n{metric_name}:")
            print(f"{'统计指标':<12} {'双均线策略':>15} {'RSI 策略':>15} {'差异':>15}")
            print("-" * 60)
            for stat_name, values in stats.items():
                diff = values[0] - values[1]
                print(f"{stat_name:<12} {values[0]:>15.6f} {values[1]:>15.6f} {diff:>15.6f}")
        else:
            print(f"\n警告：列 '{ma_col}' 或 '{rsi_col}' 在数据文件中不存在")
    
    print("\n" + "="*80)
    return results

def plot_comparison(ma_df, rsi_df):
    """绘制散点图比较两个策略的总收益率"""
    # 合并数据，按股票代码匹配
    merged_df = pd.merge(
        ma_df[['股票代码', '策略_总收益率']], 
        rsi_df[['股票代码', '策略总收益率']], 
        on='股票代码', 
        suffixes=('_MA', '_RSI')
    )
    
    # 重命名列
    merged_df.columns = ['股票代码', 'MA_Total_Return', 'RSI_Total_Return']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制散点图
    scatter = ax.scatter(
        merged_df['MA_Total_Return'], 
        merged_df['RSI_Total_Return'],
        c=merged_df['RSI_Total_Return'] - merged_df['MA_Total_Return'],
        cmap='RdYlGn',
        alpha=0.6,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )
    
    # 添加对角线（表示两个策略收益相等）
    min_val = min(merged_df['MA_Total_Return'].min(), merged_df['RSI_Total_Return'].min())
    max_val = max(merged_df['MA_Total_Return'].max(), merged_df['RSI_Total_Return'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='收益相等线')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('RSI 策略 - 双均线策略 收益差异', fontsize=12)
    
    # 设置标签和标题
    ax.set_xlabel('双均线策略总收益率 (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('RSI 策略总收益率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('双均线策略 vs RSI 策略 - 总收益率对比\n(每个点代表一只股票)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    ax.legend(loc='upper left')
    
    # 在每个点旁边标注股票代码（可选，如果股票数量不多）
    if len(merged_df) <= 50:  # 只在股票数量较少时标注
        for i, row in merged_df.iterrows():
            ax.annotate(row['股票代码'], 
                       (row['MA_Total_Return'], row['RSI_Total_Return']),
                       fontsize=8, 
                       alpha=0.7,
                       xytext=(3, 3), 
                       textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('compare_ma_rsi/strategy_comparison_scatter.png', dpi=300, bbox_inches='tight')
    print("\n散点图已保存为 'compare_ma_rsi/strategy_comparison_scatter.png'")
    plt.show()

def plot_histograms(ma_df, rsi_df):
    """绘制三个直方图对比：总收益率、最大回撤、胜率"""
    # 合并数据，按股票代码匹配
    merged_df = pd.merge(
        ma_df[['股票代码', '策略_总收益率', '策略_最大回撤', '策略_胜率']], 
        rsi_df[['股票代码', '策略总收益率', '策略最大回撤', '策略胜率']], 
        on='股票代码'
    )
    
    # 重命名列，确保列名正确
    merged_df.columns = ['股票代码', 'MA_Total_Return', 'MA_Max_Drawdown', 'MA_Win_Rate', 
                        'RSI_Total_Return', 'RSI_Max_Drawdown', 'RSI_Win_Rate']
    
    # 创建 3 个子图，横向排列
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 定义三个指标的绘图配置
    metrics_config = [
        {
            'title': '总收益率对比',
            'ma_data': merged_df['MA_Total_Return'],
            'rsi_data': merged_df['RSI_Total_Return'],
            'xlabel': '总收益率',
            'bins': 20,
            'ma_color': '#3498db',
            'rsi_color': '#e74c3c',
            'alpha': 0.6
        },
        {
            'title': '最大回撤对比',
            'ma_data': merged_df['MA_Max_Drawdown'],
            'rsi_data': merged_df['RSI_Max_Drawdown'],
            'xlabel': '最大回撤',
            'bins': 20,
            'ma_color': '#2ecc71',
            'rsi_color': '#f39c12',
            'alpha': 0.6
        },
        {
            'title': '胜率对比',
            'ma_data': merged_df['MA_Win_Rate'],
            'rsi_data': merged_df['RSI_Win_Rate'],
            'xlabel': '胜率',
            'bins': 15,
            'ma_color': '#9b59b6',
            'rsi_color': '#1abc9c',
            'alpha': 0.6
        }
    ]
    
    for idx, config in enumerate(metrics_config):
        ax = axes[idx]
        
        # 获取数据
        ma_data = config['ma_data'].dropna()
        rsi_data = config['rsi_data'].dropna()
        
        # 计算统计信息
        ma_mean = ma_data.mean()
        rsi_mean = rsi_data.mean()
        ma_std = ma_data.std()
        rsi_std = rsi_data.std()
        
        # 绘制直方图
        ax.hist(ma_data, bins=config['bins'], alpha=config['alpha'], 
               color=config['ma_color'], label=f'双均线策略\n均值={ma_mean:.4f}\n标准差={ma_std:.4f}',
               density=True, edgecolor='black', linewidth=0.5)
        ax.hist(rsi_data, bins=config['bins'], alpha=config['alpha'], 
               color=config['rsi_color'], label=f'RSI 策略\n均值={rsi_mean:.4f}\n标准差={rsi_std:.4f}',
               density=True, edgecolor='black', linewidth=0.5)
        
        # 添加均值线
        ax.axvline(ma_mean, color=config['ma_color'], linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(rsi_mean, color=config['rsi_color'], linestyle='--', linewidth=2, alpha=0.8)
        
        # 设置标签和标题
        ax.set_xlabel(config['xlabel'], fontsize=11, fontweight='bold')
        ax.set_ylabel('频率密度', fontsize=11, fontweight='bold')
        ax.set_title(config['title'], fontsize=12, fontweight='bold', pad=10)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加图例
        ax.legend(loc='best', fontsize=9)
        
        # 添加统计信息文本框
        stats_text = f'样本数：{len(ma_data)} (MA) vs {len(rsi_data)} (RSI)'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('双均线策略 vs RSI 策略 - 核心指标分布对比', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('compare_ma_rsi/strategy_comparison_histograms.png', dpi=300, bbox_inches='tight')
    print("直方图已保存为 'compare_ma_rsi/strategy_comparison_histograms.png'")
    plt.show()

def main():
    """主函数"""
    print("开始加载数据...")
    ma_df, rsi_df = load_data()
    
    print(f"双均线策略数据：{len(ma_df)} 只股票")
    print(f"RSI 策略数据：{len(rsi_df)} 只股票")
    
    # 比较策略
    print("\n正在比较两个策略...")
    results = compare_strategies(ma_df, rsi_df)
    
    # 绘制散点图
    print("\n正在绘制对比散点图...")
    plot_comparison(ma_df, rsi_df)
    
    # 绘制直方图
    print("\n正在绘制指标直方图...")
    plot_histograms(ma_df, rsi_df)
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()