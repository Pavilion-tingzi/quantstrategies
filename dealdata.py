"""
股票数据批量清洗脚本（修复版-保持原始列名）
处理300只股票的CSV文件，进行数据清洗和异常标注
"""

import os
import pandas as pd
import numpy as np
import chardet
import glob
from datetime import datetime, timedelta
import baostock as bs
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class StockDataCleaner:
    def __init__(self, input_dir='./data', output_dir='./data_dealed'):
        """
        初始化数据清洗器
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.trade_calendar = None
        self.column_mapping = {}  # 存储列名映射关系
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'error_files': 0,
            'encoding_errors': 0,
            'column_errors': 0,
            'files_with_abnormal': {
                '停牌': set(),
                '价格异常': set(),
                '退市': set()
            },
            'abnormal_counts': {
                '停牌': 0,
                '价格异常': 0
            },
            'special_stocks': {
                '600837': '【2025.3.4按1：0.62转601211】',
                '600068': '【2021.9.13按1：1.4242转601868】'
            },
            'file_details': [],
            'encoding_stats': {}
        }

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def detect_encoding_advanced(self, file_path):
        """
        高级编码检测
        """
        chinese_encodings = [
            'gb18030',
            'gbk',
            'gb2312',
            'utf-8',
            'utf-8-sig',
            'big5',
            'cp936',
            'latin1',
        ]

        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(100000)
                result = chardet.detect(raw_data)
                if result['encoding'] and result['confidence'] > 0.7:
                    detected_enc = result['encoding'].lower()
                    if detected_enc in ['gb2312', 'gbk']:
                        detected_enc = 'gb18030'
                    print(f"chardet检测到编码: {detected_enc} (置信度: {result['confidence']:.2f})")
                    return detected_enc
        except Exception as e:
            print(f"chardet检测失败: {e}")

        print("尝试常见编码...")
        for encoding in chinese_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = [f.readline() for _ in range(5)]

                sample_text = ' '.join(lines)
                chinese_keywords = ['日期', '股票代码', '开盘', '收盘', '成交量', '成交额']

                keyword_count = sum(1 for keyword in chinese_keywords if keyword in sample_text)

                if keyword_count >= 2:
                    print(f"成功使用编码: {encoding} (包含{keyword_count}个中文关键词)")
                    return encoding

            except UnicodeDecodeError:
                continue
            except Exception as e:
                continue

        print("未能确定编码，使用默认gb18030")
        return 'gb18030'

    def read_file_with_multiple_encodings(self, file_path):
        """
        尝试多种编码读取文件
        """
        detected_enc = self.detect_encoding_advanced(file_path)

        encoding_priority = [
            detected_enc,
            'gb18030',
            'utf-8-sig',
            'utf-8',
            'gbk',
            'big5',
            'latin1',
            'cp1252'
        ]

        encoding_priority = list(dict.fromkeys(encoding_priority))

        for encoding in encoding_priority:
            try:
                print(f"尝试读取: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding)

                if len(df) > 0:
                    sample_cols = [str(col) for col in df.columns[:3]]
                    sample_text = ' '.join(sample_cols)

                    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in sample_text)

                    keywords = ['日期', '股票代码', '开盘', '收盘', '成交量', '成交额']
                    keyword_count = sum(1 for keyword in keywords if keyword in sample_text)

                    if has_chinese or keyword_count > 0:
                        print(f"成功使用编码: {encoding}")
                        self.stats['encoding_stats'][os.path.basename(file_path)] = encoding
                        return df, encoding

            except Exception as e:
                print(f"编码 {encoding} 失败: {str(e)[:50]}")
                continue

        print("所有编码尝试失败")
        return None, None

    def identify_columns(self, df):
        """
        识别各列的含义（不改变列名）
        """
        column_info = {}

        # 常见的中文列名
        chinese_keywords = {
            '日期': 'date',
            '股票代码': 'code',
            '代码': 'code',
            '开盘': 'open',
            '开盘价': 'open',
            '收盘': 'close',
            '收盘价': 'close',
            '成交量': 'volume',
            '成交额': 'amount',
            '成交金额': 'amount'
        }

        # 尝试通过列名识别
        for i, col in enumerate(df.columns):
            col_str = str(col).strip()
            identified = False

            # 检查是否是已知的中文列名
            for chinese_name, eng_name in chinese_keywords.items():
                if chinese_name in col_str:
                    column_info[col] = eng_name
                    identified = True
                    print(f"列 '{col}' 识别为: {eng_name}")
                    break

            # 如果列名无法识别，尝试通过数据特征识别
            if not identified and len(df) > 0:
                sample_values = df[col].dropna().head(5).astype(str).tolist()
                sample_str = ' '.join(sample_values)

                # 日期特征
                if any(c.isdigit() for c in sample_str) and ('-' in sample_str or '/' in sample_str):
                    # 尝试解析日期
                    try:
                        pd.to_datetime(sample_values[0])
                        column_info[col] = 'date'
                        print(f"列 '{col}' 通过数据特征识别为: date")
                        identified = True
                    except:
                        pass

                # 股票代码特征
                if not identified and any(code in sample_str for code in ['600', '000', '300', '002']):
                    column_info[col] = 'code'
                    print(f"列 '{col}' 通过数据特征识别为: code")
                    identified = True

                # 价格特征
                if not identified and '.' in sample_str and all(
                        c.replace('.', '').isdigit() or c == '.' for c in sample_values[0]):
                    # 判断是开盘还是收盘（通过列名中的关键词）
                    if '开' in col_str:
                        column_info[col] = 'open'
                    elif '收' in col_str or '闭' in col_str:
                        column_info[col] = 'close'
                    else:
                        # 默认假设是收盘价
                        column_info[col] = 'close'
                    print(f"列 '{col}' 通过数据特征识别为: {column_info[col]}")
                    identified = True

                # 成交量特征
                if not identified and ('手' in sample_str or '量' in col_str or 'volume' in col_str.lower()):
                    column_info[col] = 'volume'
                    print(f"列 '{col}' 通过数据特征识别为: volume")
                    identified = True

                # 成交额特征
                if not identified and ('额' in col_str or 'amount' in col_str.lower()):
                    column_info[col] = 'amount'
                    print(f"列 '{col}' 通过数据特征识别为: amount")
                    identified = True

        return column_info

    def get_trade_calendar(self):
        """
        从baostock获取交易日历
        """
        print("正在获取交易日历...")
        try:
            lg = bs.login()
            if lg.error_code != '0':
                print(f"baostock登录失败: {lg.error_msg}")
                return self.generate_default_calendar()

            rs = bs.query_trade_dates(start_date="2015-01-01", end_date="2026-03-05")
            if rs.error_code != '0':
                print(f"获取交易日历失败: {rs.error_msg}")
                return self.generate_default_calendar()

            trade_dates = []
            while (rs.error_code == '0') & rs.next():
                trade_date = rs.get_row_data()
                if trade_date[1] == '1':
                    trade_dates.append(trade_date[0])

            bs.logout()
            print(f"成功获取{len(trade_dates)}个交易日")
            return trade_dates

        except Exception as e:
            print(f"获取交易日历出错: {e}")
            return self.generate_default_calendar()

    def generate_default_calendar(self):
        """
        生成默认的交易日历
        """
        print("使用默认交易日历...")
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2026, 3, 5)
        trade_dates = []
        current_date = start_date

        while current_date <= end_date:
            if current_date.weekday() < 5:
                trade_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        return trade_dates

    def check_price_anomaly(self, change, stock_code):
        """
        检查价格是否异常
        """
        if pd.isna(change):
            return False

        if stock_code.startswith(('000', '600')):
            threshold = 10
        elif stock_code.startswith('300'):
            threshold = 20
        else:
            threshold = 10

        return abs(change) >= threshold

    def process_stock(self, file_path, stock_code):
        """
        处理单个股票文件
        """
        # 读取文件
        df, used_encoding = self.read_file_with_multiple_encodings(file_path)

        if df is None:
            print(f"无法读取文件: {file_path}")
            self.stats['encoding_errors'] += 1
            self.stats['error_files'] += 1
            return None, None

        # 保存原始列名
        original_columns = df.columns.tolist()
        print(f"原始列名: {original_columns}")

        # 识别各列的含义（不改变列名）
        column_info = self.identify_columns(df)
        print(f"列识别结果: {column_info}")

        # 检查必要列是否都存在
        required_fields = ['date', 'code', 'close', 'volume']
        field_to_column = {}
        missing_fields = []

        for field in required_fields:
            found = False
            for col, col_type in column_info.items():
                if col_type == field:
                    field_to_column[field] = col
                    found = True
                    break
            if not found:
                missing_fields.append(field)

        if missing_fields:
            print(f"缺少必要字段: {missing_fields}")
            self.stats['column_errors'] += 1
            self.stats['error_files'] += 1
            return None, None

        print(f"字段映射: {field_to_column}")

        # 添加异常字段（使用新列名）
        df['异常情况'] = ''
        df['异常涨跌幅'] = np.nan

        # 创建临时列用于处理（方便计算）
        temp_date_col = 'date'
        temp_code_col = 'code'
        temp_close_col = 'close'
        temp_volume_col = 'volume'

        df[temp_date_col] = df[field_to_column['date']]
        df[temp_code_col] = df[field_to_column['code']]
        df[temp_close_col] = pd.to_numeric(df[field_to_column['close']], errors='coerce')
        df[temp_volume_col] = pd.to_numeric(df[field_to_column['volume']], errors='coerce')

        # 处理成交量中的单位
        if field_to_column['volume'] in df.columns:
            volume_series = df[field_to_column['volume']].astype(str)
            volume_series = volume_series.str.replace('手', '').str.replace('股', '').str.replace(',', '')
            volume_series = volume_series.str.replace('万', '0000').str.replace('亿', '00000000')
            df[temp_volume_col] = pd.to_numeric(volume_series, errors='coerce')

        # 确保日期格式正确
        try:
            df[temp_date_col] = pd.to_datetime(df[field_to_column['date']]).dt.strftime('%Y-%m-%d')
        except:
            print("日期格式转换失败，尝试其他格式...")
            for date_format in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%d-%m-%Y', '%m/%d/%Y']:
                try:
                    df[temp_date_col] = pd.to_datetime(df[field_to_column['date']], format=date_format).dt.strftime(
                        '%Y-%m-%d')
                    print(f"成功使用格式: {date_format}")
                    break
                except:
                    continue

        # 删除无效日期
        df = df.dropna(subset=[temp_date_col])
        df = df.sort_values(temp_date_col).reset_index(drop=True)

        # 初始化异常统计
        anomaly_counts = {'停牌': 0, '价格异常': 0}
        file_abnormal_types = set()
        max_abnormal_change = 0

        # 第一步：判断退市异常
        print("正在检查退市异常...")
        latest_date = df[temp_date_col].max()
        latest_close = df[df[temp_date_col] == latest_date][temp_close_col].values[0] if latest_date in df[
            temp_date_col].values else 0
        latest_volume = df[df[temp_date_col] == latest_date][temp_volume_col].values[0] if latest_date in df[
            temp_date_col].values else 0

        is_delisted = False
        delisted_suffix = ''

        if stock_code in self.stats['special_stocks']:
            delisted_suffix = self.stats['special_stocks'][stock_code]
        else:
            if (latest_date < '2026-03-05' or
                    (latest_close == 0 and latest_volume == 0)):
                is_delisted = True
                delisted_suffix = '【退市】'
                file_abnormal_types.add('退市')

        # 第二步：处理数据遗漏（补充停牌数据）
        if self.trade_calendar and not is_delisted:
            print("正在检查并补充停牌数据...")
            stock_start = df[temp_date_col].min()
            stock_end = df[temp_date_col].max()

            relevant_dates = [d for d in self.trade_calendar
                              if stock_start <= d <= stock_end]

            existing_dates = set(df[temp_date_col].tolist())
            missing_dates = [d for d in relevant_dates if d not in existing_dates]

            if missing_dates:
                print(f"发现{len(missing_dates)}个缺失交易日，将作为停牌处理")

                # 创建缺失日期的数据行
                missing_rows = []
                for i, missing_date in enumerate(missing_dates):
                    # 找到缺失日期前一个最近的交易日数据
                    prev_dates = [d for d in relevant_dates if d < missing_date and d in existing_dates]
                    if prev_dates:
                        prev_date = max(prev_dates)
                        prev_row = df[df[temp_date_col] == prev_date].iloc[0]
                        prev_close = prev_row[temp_close_col]
                    else:
                        # 如果没有前一个交易日，使用第一个交易日的数据
                        prev_close = df.iloc[0][temp_close_col] if len(df) > 0 else np.nan

                    # 创建新行，保持原始列名结构
                    new_row = {}
                    for col in original_columns:
                        if col == field_to_column['date']:
                            new_row[col] = missing_date
                        elif col == field_to_column['code']:
                            new_row[col] = stock_code
                        elif col == field_to_column['close']:
                            new_row[col] = prev_close
                        elif col == field_to_column['volume']:
                            new_row[col] = 0
                        elif 'open' in column_info and col == field_to_column.get('open'):
                            new_row[col] = prev_close
                        elif 'amount' in column_info and col == field_to_column.get('amount'):
                            new_row[col] = 0
                        else:
                            # 其他列填充空值
                            new_row[col] = np.nan

                    # 添加异常字段
                    new_row['异常情况'] = '停牌'
                    new_row['异常涨跌幅'] = 0
                    missing_rows.append(new_row)

                # 添加缺失的行
                if missing_rows:
                    df_missing = pd.DataFrame(missing_rows)
                    df = pd.concat([df, df_missing], ignore_index=True)
                    anomaly_counts['停牌'] += len(missing_dates)
                    file_abnormal_types.add('停牌')

                # 重新排序
                df = df.sort_values(by=field_to_column['date']).reset_index(drop=True)
                print(f"已补充{len(missing_dates)}条停牌数据")

        # 第三步：判断原有的停牌异常
        print("正在检查原有的停牌异常...")
        for idx, row in df.iterrows():
            volume_val = row[temp_volume_col] if not pd.isna(row[temp_volume_col]) else 0
            if volume_val == 0:
                if idx > 0:
                    prev_close = df.loc[idx - 1, temp_close_col]
                    current_close = row[temp_close_col]
                    if not pd.isna(prev_close) and not pd.isna(current_close):
                        if abs(prev_close - current_close) < 0.001:
                            if '停牌' not in str(row['异常情况']):
                                if pd.isna(row['异常情况']) or row['异常情况'] == '':
                                    df.at[idx, '异常情况'] = '停牌'
                                else:
                                    df.at[idx, '异常情况'] += ',停牌'
                                anomaly_counts['停牌'] += 1
                                file_abnormal_types.add('停牌')

        # 第四步：判断价格异常
        print("正在检查价格异常...")
        df['temp_change'] = (df[temp_close_col] / df[temp_close_col].shift(1) - 1) * 100
        df.loc[0, 'temp_change'] = np.nan

        print(f"价格数据示例（前5行）:")
        print(df[[field_to_column['date'], temp_close_col, 'temp_change']].head())

        for idx, row in df.iterrows():
            if idx > 0 and not pd.isna(row['temp_change']):
                change = row['temp_change']

                if self.check_price_anomaly(change, stock_code):
                    print(
                        f"发现价格异常: 日期={row[field_to_column['date']]}, 涨跌幅={change:.2f}%, 股票代码={stock_code}")

                    if pd.isna(row['异常情况']) or row['异常情况'] == '':
                        df.at[idx, '异常情况'] = '价格异常'
                    else:
                        df.at[idx, '异常情况'] += ',价格异常'

                    df.at[idx, '异常涨跌幅'] = change
                    anomaly_counts['价格异常'] += 1
                    file_abnormal_types.add('价格异常')

                    if abs(change) > abs(max_abnormal_change):
                        max_abnormal_change = change

        # 删除临时列
        df = df.drop(columns=[temp_date_col, temp_code_col, temp_close_col, temp_volume_col, 'temp_change'],
                     errors='ignore')

        # 更新统计
        for abnormal_type in file_abnormal_types:
            if abnormal_type in self.stats['files_with_abnormal']:
                self.stats['files_with_abnormal'][abnormal_type].add(stock_code)

        self.stats['abnormal_counts']['停牌'] += anomaly_counts['停牌']
        self.stats['abnormal_counts']['价格异常'] += anomaly_counts['价格异常']

        print(f"处理完成: 停牌={anomaly_counts['停牌']}, 价格异常={anomaly_counts['价格异常']}")

        return df, (anomaly_counts, max_abnormal_change, delisted_suffix, file_abnormal_types)

    def save_processed_file(self, df, stock_code, anomaly_info, original_filename):
        """
        保存处理后的文件
        """
        anomaly_counts, max_abnormal_change, suffix, file_abnormal_types = anomaly_info

        base_name = original_filename.replace('.csv', '')
        suffix_parts = []

        for key, count in anomaly_counts.items():
            if count > 0:
                suffix_parts.append(f"{key}{count}")

        if max_abnormal_change != 0:
            suffix_parts.append(f"最大异常涨跌幅{abs(max_abnormal_change):.2f}%")

        if suffix_parts:
            new_filename = f"{base_name}_{'_'.join(suffix_parts)}{suffix}.csv"
        else:
            new_filename = f"{base_name}{suffix}.csv"

        output_path = os.path.join(self.output_dir, new_filename)

        # 保存文件，保持原始列名不变，只添加两个新字段
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        return new_filename

    def plot_statistics(self):
        """
        生成统计图表
        """
        anomaly_types = []
        file_counts = []

        for anomaly_type, file_set in self.stats['files_with_abnormal'].items():
            count = len(file_set)
            if count > 0:
                anomaly_types.append(anomaly_type)
                file_counts.append(count)

        if not anomaly_types:
            print("没有发现异常文件，无法生成图表")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        # 柱状图
        bars = ax1.bar(anomaly_types, file_counts, color=colors[:len(anomaly_types)])
        ax1.set_title('各类异常文件数量统计', fontsize=14, fontweight='bold')
        ax1.set_xlabel('异常类型', fontsize=12)
        ax1.set_ylabel('文件数量', fontsize=12)

        for bar, count in zip(bars, file_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        total_with_abnormal = len(set().union(*[self.stats['files_with_abnormal'][t]
                                                for t in anomaly_types]))
        ax1.text(0.5, -0.15,
                 f'总处理文件数: {self.stats["processed_files"]}  |  存在异常的文件数: {total_with_abnormal}',
                 transform=ax1.transAxes, ha='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 饼图
        wedges, texts, autotexts = ax2.pie(file_counts, labels=anomaly_types,
                                           autopct=lambda pct: f'{pct:.1f}%\n({int(pct / 100 * sum(file_counts))}个)',
                                           colors=colors[:len(anomaly_types)],
                                           textprops={'fontsize': 11})
        ax2.set_title('各类异常文件占比', fontsize=14, fontweight='bold')

        ax2.text(0, -1.3,
                 '注：一个文件可能包含多种异常，\n因此各类型文件数之和可能大于总文件数',
                 transform=ax2.transAxes, ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        plt.suptitle(f'股票数据清洗异常文件统计 (共处理{self.stats["processed_files"]}个文件)',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, '异常文件统计图表.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"图表已保存至: {plot_path}")

    def run(self):
        """
        运行数据清洗流程
        """
        print("=" * 60)
        print("股票数据批量清洗程序启动")
        print("=" * 60)

        self.trade_calendar = self.get_trade_calendar()

        file_pattern = os.path.join(self.input_dir, 'df_pre_*.csv')
        stock_files = glob.glob(file_pattern)
        self.stats['total_files'] = len(stock_files)

        print(f"找到 {self.stats['total_files']} 个股票文件")

        failed_files = []
        for i, file_path in enumerate(stock_files, 1):
            try:
                filename = os.path.basename(file_path)
                stock_code = filename.replace('df_pre_', '').replace('.csv', '')

                print(f"\n{'=' * 40}")
                print(f"正在处理 [{i}/{self.stats['total_files']}] {filename}...")
                print(f"{'=' * 40}")

                result = self.process_stock(file_path, stock_code)
                if result[0] is None:
                    failed_files.append(filename)
                    continue

                df, anomaly_info = result
                new_filename = self.save_processed_file(df, stock_code, anomaly_info, filename)

                anomaly_counts, max_change, suffix, file_abnormal_types = anomaly_info
                self.stats['file_details'].append({
                    '股票代码': stock_code,
                    '原始文件名': filename,
                    '新文件名': new_filename,
                    '使用编码': self.stats['encoding_stats'].get(filename, '未知'),
                    '停牌(条数)': anomaly_counts['停牌'],
                    '价格异常(条数)': anomaly_counts['价格异常'],
                    '异常类型': ','.join(file_abnormal_types) if file_abnormal_types else '无',
                    '最大异常涨跌幅': f"{abs(max_change):.2f}%" if max_change != 0 else '无',
                    '退市标记': '是' if suffix else '否'
                })

                self.stats['processed_files'] += 1

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
                self.stats['error_files'] += 1
                failed_files.append(os.path.basename(file_path))

        print("\n" + "=" * 60)
        print("数据处理完成!")
        print("=" * 60)

        self.print_statistics(failed_files)
        self.plot_statistics()
        self.save_statistics()

    def print_statistics(self, failed_files):
        """
        打印统计结果
        """
        print("\n" + "=" * 60)
        print("异常统计结果 (按文件数量统计)")
        print("=" * 60)
        print(f"总文件数: {self.stats['total_files']}")
        print(f"成功处理: {self.stats['processed_files']}")
        print(f"处理失败: {self.stats['error_files']}")
        print(f"编码错误: {self.stats.get('encoding_errors', 0)}")
        print(f"列名错误: {self.stats.get('column_errors', 0)}")

        if self.stats['encoding_stats']:
            print("\n编码使用统计:")
            encoding_count = {}
            for enc in self.stats['encoding_stats'].values():
                encoding_count[enc] = encoding_count.get(enc, 0) + 1
            for enc, count in encoding_count.items():
                print(f"  {enc}: {count}个文件")

        if failed_files:
            print("\n失败文件列表:")
            for f in failed_files[:10]:
                print(f"  - {f}")
            if len(failed_files) > 10:
                print(f"  ... 还有 {len(failed_files) - 10} 个")

        print("\n" + "-" * 40)
        print("出现各类异常的文件数量:")
        print("-" * 40)
        total_abnormal_files = set()
        for anomaly_type, file_set in self.stats['files_with_abnormal'].items():
            count = len(file_set)
            total_abnormal_files.update(file_set)
            if count > 0:
                print(f"  {anomaly_type}: {count} 个文件")

        print(f"\n存在异常的文件总数（去重）: {len(total_abnormal_files)} 个")
        print(f"正常文件数量: {self.stats['processed_files'] - len(total_abnormal_files)} 个")

        print("\n" + "-" * 40)
        print("异常数据条数统计:")
        print("-" * 40)
        for anomaly_type, count in self.stats['abnormal_counts'].items():
            if count > 0:
                print(f"  {anomaly_type}: {count} 条")

    def save_statistics(self):
        """
        保存统计结果到CSV
        """
        if self.stats['file_details']:
            stats_df = pd.DataFrame(self.stats['file_details'])
            stats_path = os.path.join(self.output_dir, '异常统计汇总.csv')
            stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
            print(f"\n详细统计已保存至: {stats_path}")


def main():
    cleaner = StockDataCleaner(input_dir='./data', output_dir='./data_dealed')
    cleaner.run()
    print("\n" + "=" * 60)
    print("程序执行完毕!")
    print("=" * 60)


if __name__ == "__main__":
    main()