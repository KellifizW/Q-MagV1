import time
import os
import subprocess
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import requests
from pytz import timezone
import logging
from typing import Optional, List, Dict, Tuple, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量配置
REPO_DIR = "."
DB_PATH = "stocks.db"
TICKERS_CSV = "Tickers.csv"
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"
US_EASTERN = timezone('US/Eastern')
YF_BATCH_SIZE = 20  # yfinance批次大小
MS_BATCH_SIZE = 20  # Marketstack批次大小
MONTHLY_REQUEST_LIMIT = 100  # Marketstack免费版每月限制
MAX_HISTORY_DAYS = 90  # 最大历史数据天数

# Streamlit日志显示
def log_to_page(message: str, level: str = "INFO") -> None:
    """在Streamlit页面显示日志消息"""
    if level == "INFO":
        st.info(message)
    elif level == "WARNING":
        st.warning(message)
    elif level == "ERROR":
        st.error(message)
    elif level == "DEBUG":
        st.write(f"DEBUG: {message}")

# 数据验证函数
def safe_float(value: Any, column_name: str, ticker: str, date: str) -> Optional[float]:
    """安全转换为浮点数"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (ValueError, TypeError) as e:
        logger.error(f"转换失败 {column_name}, 股票: {ticker}, 日期: {date}, 值: {repr(value)}")
        return None

def safe_int(value: Any, column_name: str, ticker: str, date: str) -> int:
    """安全转换为整数"""
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (ValueError, TypeError) as e:
        logger.error(f"转换失败 {column_name}, 股票: {ticker}, 日期: {date}, 值: {repr(value)}")
        return 0

# 数据缓存检查
def check_data_exists(db_conn: sqlite3.Connection, ticker: str, date: str) -> bool:
    """检查数据是否已存在"""
    try:
        cursor = db_conn.cursor()
        cursor.execute(
            "SELECT 1 FROM stocks WHERE Ticker = ? AND Date = ?",
            (ticker, date))
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"检查数据存在失败: {str(e)}")
        return False

# Marketstack数据处理
def process_marketstack_data(api_data: List[Dict]) -> Optional[pd.DataFrame]:
    """处理Marketstack API返回的原始数据"""
    if not api_data:
        return None

    records = []
    for item in api_data:
        # 验证必要字段
        required_fields = {'date', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        if not all(field in item for field in required_fields):
            continue

        try:
            records.append({
                "Date": item["date"].split("T")[0],
                "Ticker": item["symbol"],
                "Open": item["open"],
                "High": item["high"],
                "Low": item["low"],
                "Close": item["close"],
                "Volume": item["volume"],
                "Adj_Close": item.get("adj_close", item["close"])
            })
        except Exception as e:
            logger.error(f"处理数据项失败: {str(e)}")
            continue

    return pd.DataFrame(records) if records else None

# 优化的Marketstack下载函数
def download_with_marketstack(
    tickers: List[str],
    start: datetime,
    end: datetime,
    api_key: str,
    request_count: List[int],
    db_conn: sqlite3.Connection
) -> Optional[pd.DataFrame]:
    """优化的Marketstack数据下载"""
    if request_count[0] >= MONTHLY_REQUEST_LIMIT:
        logger.error(f"已达每月请求限制 {MONTHLY_REQUEST_LIMIT} 次")
        return None

    try:
        # 过滤掉已存在的数据（只检查最后一天）
        needed_tickers = [
            t for t in tickers 
            if not check_data_exists(db_conn, t, end.strftime('%Y-%m-%d'))
        ]
        
        if not needed_tickers:
            logger.info(f"所有 {len(tickers)} 只股票数据已存在，跳过下载")
            return pd.DataFrame()

        symbols = ','.join(needed_tickers)
        url = f"http://api.marketstack.com/v1/eod?access_key={api_key}&symbols={symbols}&date_from={start.strftime('%Y-%m-%d')}&date_to={end.strftime('%Y-%m-%d')}"
        
        logger.info(f"请求Marketstack API: {url.replace(api_key, '***')}")
        response = requests.get(url, timeout=10)
        
        # 只在成功响应时计数
        if response.status_code == 200:
            request_count[0] += 1
            logger.info(f"剩余API请求次数: {MONTHLY_REQUEST_LIMIT - request_count[0]}")
        else:
            logger.error(f"请求失败，状态码: {response.status_code}")
            return None

        data = response.json()
        
        if "error" in data:
            logger.error(f"API返回错误: {data['error']}")
            return None
            
        return process_marketstack_data(data.get("data", []))
        
    except Exception as e:
        logger.error(f"Marketstack请求异常: {str(e)}")
        return None

# 数据下载函数（带重试）
def download_with_retry(
    tickers: List[str],
    start: datetime,
    end: datetime,
    api_key: str,
    request_count: List[int],
    db_conn: sqlite3.Connection,
    retries: int = 2,
    delay: int = 5
) -> Optional[pd.DataFrame]:
    """带重试的数据下载"""
    for attempt in range(retries):
        # 强制使用yfinance失败（測試用）
        try:
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
            if data.empty:
                logger.warning(f"yfinance数据为空: {tickers}")
                return None
            return data
        except Exception as e:
            logger.warning(f"yfinance下载失败 {tickers}, 尝试 {attempt+1}/{retries}: {str(e)}")
            time.sleep(delay)

    # 回退到Marketstack
    return download_with_marketstack(tickers, start, end, api_key, request_count, db_conn)

# 批量插入数据到数据库
def bulk_insert_data(df: pd.DataFrame, db_conn: sqlite3.Connection) -> bool:
    """高效批量插入数据到数据库"""
    if df.empty:
        return True

    try:
        df.to_sql('stocks', db_conn, if_exists='append', index=False, 
                 dtype={
                     'Date': 'TEXT',
                     'Ticker': 'TEXT',
                     'Open': 'REAL',
                     'High': 'REAL',
                     'Low': 'REAL',
                     'Close': 'REAL',
                     'Adj_Close': 'REAL',
                     'Volume': 'INTEGER'
                 })
        db_conn.commit()
        return True
    except Exception as e:
        logger.error(f"批量插入失败: {str(e)}")
        db_conn.rollback()
        return False

# 初始化Git仓库
def init_repo() -> bool:
    """初始化Git仓库"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
            log_to_page("初始化Git仓库", "INFO")

        if "TOKEN" not in st.secrets:
            st.error("未找到'TOKEN'，请在Secrets中配置")
            return False

        token = st.secrets["TOKEN"]
        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        
        subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True, text=True)
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'config', 'user.name', 'KellifizW'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'your.email@example.com'], check=True)
        
        log_to_page("Git仓库初始化完成", "INFO")
        return True
    except Exception as e:
        st.error(f"初始化Git仓库失败: {str(e)}")
        return False

# 推送到GitHub
def push_to_github(message: str = "Update stocks.db") -> bool:
    """推送更改到GitHub"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists(DB_PATH):
            st.error(f"{DB_PATH} 不存在")
            return False

        # 检查是否有变更
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status.stdout.strip():
            st.write("无变更需要推送")
            return True

        subprocess.run(['git', 'add', DB_PATH], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True, text=True)
        
        token = st.secrets["TOKEN"]
        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        branch = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip() or 'main'
        
        subprocess.run(['git', 'push', remote_url, branch], check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        st.error(f"推送失败: {str(e)}")
        return False

# 初始化数据库
def init_database() -> bool:
    """初始化数据库"""
    if 'db_initialized' not in st.session_state:
        try:
            if "TOKEN" not in st.secrets:
                st.error("未找到GitHub TOKEN")
                return False

            # 尝试从GitHub下载现有数据库
            url = "https://raw.githubusercontent.com/KellifizW/Q-MagV1/main/stocks.db"
            headers = {"Authorization": f"token {st.secrets['TOKEN']}"}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                with open(DB_PATH, "wb") as f:
                    f.write(response.content)
                st.write("从GitHub下载stocks.db成功")
            else:
                # 创建新数据库
                conn = sqlite3.connect(DB_PATH)
                conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
                    Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, 
                    Close REAL, Adj_Close REAL, Volume INTEGER,
                    PRIMARY KEY (Date, Ticker))''')
                conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
                conn.commit()
                conn.close()
                st.write("创建新的stocks.db数据库")
            
            st.session_state['db_initialized'] = True
            return True
        except Exception as e:
            st.error(f"初始化数据库失败: {str(e)}")
            st.session_state['db_initialized'] = False
            return False
    return True

# 主更新函数
def update_database(
    tickers_file: str = TICKERS_CSV,
    db_path: str = DB_PATH,
    repo: Optional[bool] = None,
    check_percentage: float = 0.1
) -> bool:
    """主数据库更新函数"""
    if repo is None:
        st.error("未提供Git仓库对象")
        return False

    try:
        # 读取股票代码
        tickers_df = pd.read_csv(tickers_file)
        tickers = tickers_df['Ticker'].tolist()
        log_to_page(f"从 {tickers_file} 读取 {len(tickers)} 只股票", "INFO")

        # 连接数据库
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        
        # 确保表结构存在
        conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, 
            Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        conn.commit()

        # 获取当前日期
        current_date = datetime.now(US_EASTERN).date()
        end_date = current_date - timedelta(days=1)
        
        # 获取各股票最后更新日期
        ticker_dates = pd.read_sql_query(
            "SELECT Ticker, MAX(Date) as last_date FROM stocks GROUP BY Ticker", 
            conn)
        existing_tickers = dict(zip(ticker_dates['Ticker'], pd.to_datetime(ticker_dates['last_date']).dt.date))

        # 确定需要更新的股票
        num_to_check = max(1, int(len(tickers) * check_percentage))
        tickers_to_check = tickers[-num_to_check:]
        tickers_to_update = []
        default_start_date = end_date - timedelta(days=MAX_HISTORY_DAYS)

        for ticker in tickers_to_check:
            last_date = existing_tickers.get(ticker)
            if not last_date or (end_date - last_date).days > 0:
                tickers_to_update.append(ticker)

        if not tickers_to_update:
            if len(existing_tickers) >= len(tickers):
                st.write("数据库已是最新且完整，无需更新")
                conn.close()
                return True
            else:
                st.write(f"数据库缺少部分股票数据（现有 {len(existing_tickers)} / 共 {len(tickers)}），将更新缺失部分")
                tickers_to_update = [t for t in tickers if t not in existing_tickers]

        # 获取API密钥
        try:
            api_key = st.secrets["MARKETSTACK_API_KEY"]
            if not api_key:
                st.error("Marketstack API Key为空")
                return False
        except KeyError:
            st.error("未找到MARKETSTACK_API_KEY")
            return False

        # 设置进度显示
        st.write("下载进度")
        progress_bar = st.progress(0)
        status_text = st.empty()
        request_count = [0]
        success_count = [0]
        fail_count = [0]
        
        # 显示API使用情况
        api_usage = st.sidebar.empty()
        
        # 分批处理
        total_batches = (len(tickers_to_update) + MS_BATCH_SIZE - 1) // MS_BATCH_SIZE
        
        for i in range(0, len(tickers_to_update), MS_BATCH_SIZE):
            batch_tickers = tickers_to_update[i:i + MS_BATCH_SIZE]
            batch_num = i // MS_BATCH_SIZE + 1
            
            # 更新进度
            progress = min((i + MS_BATCH_SIZE) / len(tickers_to_update), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"处理批次 {batch_num}/{total_batches}: {', '.join(batch_tickers[:3])}...")
            
            # 检查哪些股票需要更新
            need_update = []
            for ticker in batch_tickers:
                if not check_data_exists(conn, ticker, end_date.strftime('%Y-%m-%d')):
                    need_update.append(ticker)
            
            if not need_update:
                logger.info(f"批次 {batch_num} 数据已是最新，跳过")
                continue
                
            # 计算批次开始日期
            batch_start_dates = [
                existing_tickers.get(t, default_start_date) - timedelta(days=1)
                for t in need_update
            ]
            start_date = min(batch_start_dates)
            
            # 下载数据
            data = download_with_retry(
                need_update, 
                start=start_date,
                end=end_date,
                api_key=api_key,
                request_count=request_count,
                db_conn=conn
            )
            
            if data is not None and not data.empty:
                if bulk_insert_data(data, conn):
                    success_count[0] += 1
                else:
                    fail_count[0] += 1
            
            # 更新API使用情况显示
            api_usage.markdown(f"""
                **API使用情况**  
                成功: {success_count[0]}  
                失败: {fail_count[0]}  
                剩余: {MONTHLY_REQUEST_LIMIT - request_count[0]}
            """)

        # 更新元数据
        conn.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)", 
                    (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        conn.close()

        # 推送到GitHub
        if push_to_github(f"更新 {len(tickers_to_update)} 只股票数据"):
            st.success("数据库更新完成并推送至GitHub")
        else:
            st.warning("数据库更新完成但推送至GitHub失败")
        
        # 添加下载按钮
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                st.download_button(
                    label="下载stocks.db",
                    data=f,
                    file_name="stocks.db",
                    mime="application/octet-stream"
                )
        
        return True

    except Exception as e:
        st.error(f"数据库更新失败: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return False

def fetch_stock_data(
    tickers: List[str], 
    db_path: str = DB_PATH, 
    trading_days: int = 140
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """从数据库获取股票数据
    
    Args:
        tickers: 股票代码列表
        db_path: 数据库路径 (默认: DB_PATH)
        trading_days: 要获取的交易天数 (默认: 140)
        
    Returns:
        Tuple[可选DataFrame, 原始股票代码列表]:
            - DataFrame: 包含股票数据 (None表示失败)
            - List[str]: 原始输入的股票代码列表
    """
    try:
        # 验证数据库存在
        if not os.path.exists(db_path):
            st.error(f"数据库文件 {db_path} 不存在")
            return None, tickers
            
        # 计算开始日期
        start_date = (datetime.now(US_EASTERN).date() - timedelta(days=trading_days * 1.5)).strftime('%Y-%m-%d')
        
        # 使用参数化查询防止SQL注入
        placeholders = ','.join(['?'] * len(tickers))
        query = f"""
        SELECT * FROM stocks 
        WHERE Ticker IN ({placeholders}) 
        AND Date >= ?
        ORDER BY Date
        """
        
        # 执行查询
        conn = sqlite3.connect(db_path)
        data = pd.read_sql_query(
            query, 
            conn, 
            params=tickers + [start_date],
            parse_dates=['Date']
        )
        conn.close()
        
        # 检查空数据
        if data.empty:
            st.error(f"没有找到数据: {tickers}")
            return None, tickers
            
        # 转换为pivot格式
        pivoted_data = data.pivot(index='Date', columns='Ticker')
        
        # 记录统计信息
        st.write(
            f"获取数据 - 股票数: {len(tickers)}, "
            f"数据条目: {len(data)}, "
            f"日期范围: {data['Date'].min().date()} 至 {data['Date'].max().date()}"
        )
        
        return pivoted_data, tickers
        
    except sqlite3.Error as e:
        st.error(f"数据库查询失败: {str(e)}")
        return None, tickers
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        return None, tickers

# 主程序
if __name__ == "__main__":
    st.title("股票数据库更新工具")
    
    if init_repo() and init_database():
        if st.button("初始化并更新数据库"):
            update_database()
        
        if st.button("仅更新数据库"):
            update_database()
