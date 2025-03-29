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
import warnings
import urllib3
from yfinance import utils

# 配置日誌記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 優化 yfinance 設置
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
utils.get_session().close()
utils.set_session(requests.Session())

# 調整連接池參數
session = utils.get_session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
    max_retries=3
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# 常數配置
REPO_DIR = "."
DB_PATH = "stocks.db"
TICKERS_CSV = "Tickers.csv"
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"
US_EASTERN = timezone('US/Eastern')
YF_BATCH_SIZE = 10
MS_BATCH_SIZE = 100
MONTHLY_REQUEST_LIMIT = 100
MAX_HISTORY_DAYS = 90

# Streamlit 日誌顯示
def log_to_page(message: str, level: str = "INFO") -> None:
    """在 Streamlit 頁面上顯示日誌消息"""
    if level == "INFO":
        st.info(message)
    elif level == "WARNING":
        st.warning(message)
    elif level == "ERROR":
        st.error(message)
    elif level == "DEBUG":
        st.write(f"除錯: {message}")

# 數據驗證函數
def safe_float(value: Any, column_name: str, ticker: str, date: str) -> Optional[float]:
    """安全轉換為浮點數"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (ValueError, TypeError) as e:
        logger.error(f"轉換失敗 {column_name}, 股票代碼: {ticker}, 日期: {date}, 值: {repr(value)}")
        return None

def safe_int(value: Any, column_name: str, ticker: str, date: str) -> int:
    """安全轉換為整數"""
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (ValueError, TypeError) as e:
        logger.error(f"轉換失敗 {column_name}, 股票代碼: {ticker}, 日期: {date}, 值: {repr(value)}")
        return 0

# 數據緩存檢查
def check_data_exists(db_conn: sqlite3.Connection, ticker: str, date: str) -> bool:
    """檢查數據是否已存在"""
    cursor = None
    try:
        cursor = db_conn.cursor()
        cursor.execute(
            "SELECT 1 FROM stocks WHERE Ticker = ? AND Date = ?",
            (ticker, date))
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"檢查數據存在失敗: {str(e)}")
        return False
    finally:
        if cursor:
            cursor.close()

# Marketstack 數據處理
def process_marketstack_data(api_data: List[Dict]) -> Optional[pd.DataFrame]:
    """處理 Marketstack API 返回的原始數據"""
    if not api_data:
        return None

    records = []
    for item in api_data:
        # 驗證必填字段
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
            logger.error(f"處理數據項失敗: {str(e)}")
            continue

    return pd.DataFrame(records) if records else None

# 優化的 Marketstack 下載函數
def download_with_marketstack(
    tickers: List[str],
    start: datetime,
    end: datetime,
    api_key: str,
    request_count: List[int],
    db_conn: sqlite3.Connection
) -> Optional[pd.DataFrame]:
    """優化的 Marketstack 數據下載"""
    if request_count[0] >= MONTHLY_REQUEST_LIMIT:
        logger.error(f"已達到每月請求限制 {MONTHLY_REQUEST_LIMIT}")
        return None

    try:
        # 過濾已存在的數據（僅檢查最後一天）
        needed_tickers = [
            t for t in tickers 
            if not check_data_exists(db_conn, t, end.strftime('%Y-%m-%d'))
        ]
        
        if not needed_tickers:
            logger.info(f"所有 {len(tickers)} 支股票數據已存在，跳過下載")
            return pd.DataFrame()

        symbols = ','.join(needed_tickers)
        url = f"http://api.marketstack.com/v1/eod?access_key={api_key}&symbols={symbols}&date_from={start.strftime('%Y-%m-%d')}&date_to={end.strftime('%Y-%m-%d')}"
        
        logger.info(f"請求 Marketstack API: {url.replace(api_key, '***')}")
        response = requests.get(url, timeout=10)
        
        # 僅在成功響應時增加計數器
        if response.status_code == 200:
            request_count[0] += 1
            logger.info(f"剩餘 API 請求次數: {MONTHLY_REQUEST_LIMIT - request_count[0]}")
        else:
            logger.error(f"請求失敗，狀態碼: {response.status_code}")
            return None

        data = response.json()
        
        if "error" in data:
            logger.error(f"API 返回錯誤: {data['error']}")
            return None
            
        return process_marketstack_data(data.get("data", []))
        
    except Exception as e:
        logger.error(f"Marketstack 請求異常: {str(e)}")
        return None

# 帶重試的數據下載函數
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
    """帶重試機制的數據下載"""
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
            if data.empty:
                logger.warning(f"yfinance 數據為空: {tickers}")
                return None
            
            # 處理多級列（如果存在）
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[1] if col[0] == '' else '_'.join(col) for col in data.columns]
            
            # 確保列結構一致
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in expected_columns:
                if col not in data.columns:
                    if col == 'Adj Close':
                        data['Adj Close'] = data['Close']
                    else:
                        data[col] = 0.0
            
            # 添加日期和股票代碼列
            data.reset_index(inplace=True)
            data['Ticker'] = tickers[0] if len(tickers) == 1 else 'Multiple'
            
            # 標準化列名
            data.rename(columns={'Adj Close': 'Adj_Close', 'Date': 'Date'}, inplace=True)
            
            return data
        except Exception as e:
            logger.warning(f"yfinance 下載失敗 {tickers}, 嘗試 {attempt+1}/{retries}: {str(e)}")
            time.sleep(delay)

    # 回退到 Marketstack
    return download_with_marketstack(tickers, start, end, api_key, request_count, db_conn)

# 批量插入數據到數據庫
def bulk_insert_data(df: pd.DataFrame, db_conn: sqlite3.Connection) -> bool:
    """使用 INSERT OR IGNORE 高效批量插入數據"""
    if df.empty:
        return True

    try:
        # 標準化列順序和名稱
        expected_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        df = df[expected_columns].copy()  # 顯式copy避免碎片化警告
        
        # 使用臨時表+INSERT OR IGNORE方案
        df.to_sql('temp_stocks', db_conn, if_exists='replace', index=False)
        
        # 執行批量INSERT OR IGNORE
        db_conn.execute("""
            INSERT OR IGNORE INTO stocks
            SELECT * FROM temp_stocks
        """)
        
        # 清理臨時表
        db_conn.execute("DROP TABLE temp_stocks")
        db_conn.commit()
        
        logger.info(f"成功插入 {db_conn.total_changes} 條記錄")
        return True
        
    except sqlite3.IntegrityError as e:
        logger.error(f"唯一性約束錯誤: {str(e)}")
        db_conn.rollback()
        return False
    except Exception as e:
        logger.error(f"批量插入失敗: {str(e)}")
        db_conn.rollback()
        return False

# 初始化 Git 存儲庫
def init_repo() -> bool:
    """初始化 Git 存儲庫"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
            log_to_page("已初始化 Git 存儲庫", "INFO")

        if "TOKEN" not in st.secrets:
            st.error("未找到 GitHub TOKEN，請在 Secrets 中配置")
            return False

        token = st.secrets["TOKEN"]
        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        
        subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True, text=True)
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'config', 'user.name', 'KellifizW'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'your.email@example.com'], check=True)
        
        log_to_page("Git 存儲庫初始化完成", "INFO")
        return True
    except Exception as e:
        st.error(f"初始化 Git 存儲庫失敗: {str(e)}")
        return False

# 推送到 GitHub
def push_to_github(message: str = "更新 stocks.db") -> bool:
    """推送更改到 GitHub"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists(DB_PATH):
            st.error(f"{DB_PATH} 不存在")
            return False

        # 檢查更改
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status.stdout.strip():
            st.write("沒有更改需要推送")
            return True

        subprocess.run(['git', 'add', DB_PATH], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True, text=True)
        
        token = st.secrets["TOKEN"]
        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        branch = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip() or 'main'
        
        subprocess.run(['git', 'push', remote_url, branch], check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        st.error(f"推送失敗: {str(e)}")
        return False

# 初始化數據庫
def init_database() -> bool:
    """初始化數據庫"""
    if 'db_initialized' not in st.session_state:
        try:
            if "TOKEN" not in st.secrets:
                st.error("未找到 GitHub TOKEN")
                return False

            # 嘗試從 GitHub 下載現有數據庫
            url = "https://raw.githubusercontent.com/KellifizW/Q-MagV1/main/stocks.db"
            headers = {"Authorization": f"token {st.secrets['TOKEN']}"}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                with open(DB_PATH, "wb") as f:
                    f.write(response.content)
                st.write("成功從 GitHub 下載 stocks.db")
            else:
                # 創建新數據庫
                conn = sqlite3.connect(DB_PATH)
                conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
                    Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, 
                    Close REAL, Adj_Close REAL, Volume INTEGER,
                    PRIMARY KEY (Date, Ticker))''')
                conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
                conn.commit()
                conn.close()
                st.write("創建了新的 stocks.db 數據庫")
            
            st.session_state['db_initialized'] = True
            return True
        except Exception as e:
            st.error(f"初始化數據庫失敗: {str(e)}")
            st.session_state['db_initialized'] = False
            return False
    return True

# 主更新函數
def update_database(
    tickers_file: str = TICKERS_CSV,
    db_path: str = DB_PATH,
    repo: Optional[bool] = None,
    check_percentage: float = 0.1
) -> bool:
    """主數據庫更新函數"""
    if repo is None:
        st.error("未提供 Git 存儲庫對象")
        return False

    try:
        # 讀取股票代碼
        tickers_df = pd.read_csv(tickers_file)
        tickers = tickers_df['Ticker'].tolist()
        log_to_page(f"從 {tickers_file} 讀取了 {len(tickers)} 支股票", "INFO")

        # 連接到數據庫
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        
        # 確保表結構存在
        conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, 
            Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        conn.commit()

        # 獲取當前日期
        current_date = datetime.now(US_EASTERN).date()
        end_date = current_date - timedelta(days=1)
        
        # 獲取每個股票代碼的最後更新日期
        ticker_dates = pd.read_sql_query(
            "SELECT Ticker, MAX(Date) as last_date FROM stocks GROUP BY Ticker", 
            conn)
        existing_tickers = dict(zip(ticker_dates['Ticker'], pd.to_datetime(ticker_dates['last_date']).dt.date))

        # 確定需要更新的股票
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
                st.write("數據庫已是最新且完整，無需更新")
                conn.close()
                return True
            else:
                st.write(f"數據庫缺少部分股票數據(現有 {len(existing_tickers)} / 總計 {len(tickers)})，正在更新缺失部分")
                tickers_to_update = [t for t in tickers if t not in existing_tickers]

        # 獲取 API 密鑰
        try:
            api_key = st.secrets["MARKETSTACK_API_KEY"]
            if not api_key:
                st.error("Marketstack API 密鑰為空")
                return False
        except KeyError:
            st.error("未找到 MARKETSTACK_API_KEY")
            return False

        # 設置進度顯示
        st.write("下載進度")
        progress_bar = st.progress(0)
        status_text = st.empty()
        request_count = [0]
        success_count = [0]
        fail_count = [0]
        
        # 顯示 API 使用情況
        api_usage = st.sidebar.empty()
        
        # 分批處理
        total_batches = (len(tickers_to_update) + MS_BATCH_SIZE - 1) // MS_BATCH_SIZE
        
        for i in range(0, len(tickers_to_update), MS_BATCH_SIZE):
            batch_tickers = tickers_to_update[i:i + MS_BATCH_SIZE]
            batch_num = i // MS_BATCH_SIZE + 1
            
            # 更新進度
            progress = min((i + MS_BATCH_SIZE) / len(tickers_to_update), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"處理批次 {batch_num}/{total_batches}: {', '.join(batch_tickers[:3])}...")
            
            # 檢查哪些股票需要更新
            need_update = []
            for ticker in batch_tickers:
                if not check_data_exists(conn, ticker, end_date.strftime('%Y-%m-%d')):
                    need_update.append(ticker)
            
            if not need_update:
                logger.info(f"批次 {batch_num} 數據已是最新，跳過")
                continue
                
            # 計算批次開始日期
            batch_start_dates = [
                existing_tickers.get(t, default_start_date) - timedelta(days=1)
                for t in need_update
            ]
            start_date = min(batch_start_dates)
            
            # 下載數據
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
            
            # 更新 API 使用情況顯示
            api_usage.markdown(f"""
                **API 使用情況**  
                成功: {success_count[0]}  
                失敗: {fail_count[0]}  
                剩餘次數: {MONTHLY_REQUEST_LIMIT - request_count[0]}
            """)

        # 更新元數據
        conn.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)", 
                    (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        conn.close()

        # 推送到 GitHub
        if push_to_github(f"更新了 {len(tickers_to_update)} 支股票的數據"):
            st.success("數據庫更新完成並推送至 GitHub")
        else:
            st.warning("數據庫已更新但推送至 GitHub 失敗")
        
        # 添加下載按鈕
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                st.download_button(
                    label="下載 stocks.db",
                    data=f,
                    file_name="stocks.db",
                    mime="application/octet-stream"
                )
        
        return True

    except Exception as e:
        st.error(f"數據庫更新失敗: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return False

def fetch_stock_data(
    tickers: List[str], 
    db_path: str = DB_PATH, 
    trading_days: int = 140
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """從數據庫獲取股票數據
    
    參數:
        tickers: 股票代碼列表
        db_path: 數據庫路徑 (默認: DB_PATH)
        trading_days: 要獲取的交易日天數 (默認: 140)
        
    返回:
        元組[可選 DataFrame, 原始股票代碼列表]:
            - DataFrame: 包含股票數據 (失敗時為 None)
            - List[str]: 原始輸入股票代碼列表
    """
    try:
        # 驗證數據庫是否存在
        if not os.path.exists(db_path):
            st.error(f"數據庫文件 {db_path} 不存在")
            return None, tickers
            
        # 計算開始日期
        start_date = (datetime.now(US_EASTERN).date() - timedelta(days=trading_days * 1.5)).strftime('%Y-%m-%d')
        
        # 使用參數化查詢防止 SQL 注入
        placeholders = ','.join(['?'] * len(tickers))
        query = f"""
        SELECT * FROM stocks 
        WHERE Ticker IN ({placeholders}) 
        AND Date >= ?
        ORDER BY Date
        """
        
        # 執行查詢
        conn = sqlite3.connect(db_path)
        data = pd.read_sql_query(
            query, 
            conn, 
            params=tickers + [start_date],
            parse_dates=['Date']
        )
        conn.close()
        
        # 檢查空數據
        if data.empty:
            st.error(f"未找到數據: {tickers}")
            return None, tickers
            
        # 轉換為透視格式
        pivoted_data = data.pivot(index='Date', columns='Ticker')
        
        # 記錄統計信息
        st.write(
            f"已獲取數據 - 股票數量: {len(tickers)}, "
            f"條目數: {len(data)}, "
            f"日期範圍: {data['Date'].min().date()} 至 {data['Date'].max().date()}"
        )
        
        return pivoted_data, tickers
        
    except sqlite3.Error as e:
        st.error(f"數據庫查詢失敗: {str(e)}")
        return None, tickers
    except Exception as e:
        st.error(f"獲取數據失敗: {str(e)}")
        return None, tickers

# 主程序
if __name__ == "__main__":
    st.title("股票數據庫更新工具")
    
    if init_repo() and init_database():
        if st.button("初始化並更新數據庫"):
            update_database()
        
        if st.button("僅更新數據庫"):
            update_database()
