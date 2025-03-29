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

# 設定後台日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定義
REPO_DIR = "."
DB_PATH = "stocks.db"
TICKERS_CSV = "Tickers.csv"
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"
US_EASTERN = timezone('US/Eastern')
YF_BATCH_SIZE = 20  # yfinance 批次大小
MS_BATCH_SIZE = 100  # Marketstack 批次大小
MONTHLY_REQUEST_LIMIT = 100  # Marketstack 免費版每月限制

# 使用 Streamlit 顯示日誌
def log_to_page(message, level="INFO"):
    if level == "INFO":
        st.info(message)
    elif level == "WARNING":
        st.warning(message)
    elif level == "ERROR":
        st.error(message)
    elif level == "DEBUG":
        st.write(f"DEBUG: {message}")

def safe_float(value, column_name, ticker, date):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (ValueError, TypeError) as e:
        logger.error(f"無法將 {column_name} 轉換為浮點數，股票：{ticker}，日期：{date}，值：{repr(value)}，錯誤：{str(e)}")
        raise ValueError(f"Invalid {column_name} value for {ticker} on {date}: {repr(value)}")

def safe_int(value, column_name, ticker, date):
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (ValueError, TypeError) as e:
        logger.error(f"無法將 {column_name} 轉換為整數，股票：{ticker}，日期：{date}，值：{repr(value)}，錯誤：{str(e)}")
        raise ValueError(f"Invalid {column_name} value for {ticker} on {date}: {repr(value)}")

def download_with_retry(tickers, start, end, retries=2, delay=5, api_key=None, request_count=[0], success_count=[0], fail_count=[0]):
    """下載股票數據，優先 yfinance，失敗則用 Marketstack"""
    # 嘗試 yfinance
    for attempt in range(retries):
        try:
            raise Exception("強制 yfinance 失敗")  # 測試時啟用
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
            if data.empty:
                logger.warning(f"批次數據為空，股票：{tickers}")
                return None
            success_count[0] += 1
            return data
        except Exception as e:
            logger.warning(f"yfinance 下載失敗，股票：{tickers}，錯誤：{str(e)}，重試 {attempt + 1}/{retries}")
            time.sleep(delay)

    # 切換到 Marketstack
    if not api_key:
        logger.error(f"未提供 Marketstack API Key，下載 {tickers} 失敗")
        fail_count[0] += 1
        return None

    logger.info(f"yfinance 重試失敗，切換到 Marketstack 嘗試下載 {tickers}")
    if request_count[0] >= MONTHLY_REQUEST_LIMIT:
        logger.error(f"已達每月請求限制 {MONTHLY_REQUEST_LIMIT} 次，停止下載")
        fail_count[0] += 1
        return None

    try:
        symbols = ','.join(tickers)
        url = f"http://api.marketstack.com/v1/eod?access_key={api_key}&symbols={symbols}&date_from={start.strftime('%Y-%m-%d')}&date_to={end.strftime('%Y-%m-%d')}"
        logger.debug(f"請求 Marketstack，股票：{tickers}，URL：{url}")
        response = requests.get(url).json()
        logger.debug(f"Marketstack API 回應: {response}")
        request_count[0] += 1

        if "data" not in response:
            logger.error(f"Marketstack 無數據返回，股票：{tickers}，回應：{response}")
            fail_count[0] += 1
            return None

        all_data = []
        for item in response["data"]:
            df = pd.DataFrame([{
                "Date": item["date"].split("T")[0],
                "Open": item["open"],
                "High": item["high"],
                "Low": item["low"],
                "Close": item["close"],
                "Volume": item["volume"],
                "Adj Close": item.get("adj_close", item["close"]),
                "Ticker": item["symbol"]
            }])
            all_data.append(df)

        if not all_data:
            logger.error(f"Marketstack 下載 {tickers} 失敗，無有效數據")
            fail_count[0] += 1
            return None

        combined_df = pd.concat(all_data)
        # 修改：使用 pivot 確保與 yfinance 的多層索引格式一致
        combined_df = combined_df.pivot(index='Date', columns='Ticker', values=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']).reset_index()
        combined_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in combined_df.columns]
        logger.info(f"成功從 Marketstack 下載 {tickers} 的數據")
        success_count[0] += 1
        return combined_df

    except Exception as e:
        logger.error(f"Marketstack 下載失敗，股票：{tickers}，錯誤：{str(e)}")
        fail_count[0] += 1
        return None

def init_repo():
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
            log_to_page("初始化 Git 倉庫", "INFO")

        if "TOKEN" not in st.secrets:
            st.error("未找到 'TOKEN'，請在 Streamlit Cloud 的 Secrets 中配置")
            return None
        token = st.secrets["TOKEN"]

        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True, text=True)
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True, capture_output=True, text=True)
        
        subprocess.run(['git', 'config', 'user.name', 'KellifizW'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'your.email@example.com'], check=True)
        log_to_page("Git 倉庫初始化完成", "INFO")
        return True
    except Exception as e:
        st.error(f"初始化 Git 倉庫失敗：{str(e)}")
        return None

def push_to_github(repo, message="Update stocks.db"):
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists(DB_PATH):
            st.error(f"stocks.db 不存在於路徑：{DB_PATH}")
            return False

        subprocess.run(['git', 'add', DB_PATH], check=True, capture_output=True, text=True)
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status.stdout:
            st.write("無變更需要推送")
            return True

        subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True, text=True)
        token = st.secrets["TOKEN"]
        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        branch = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip() or 'main'
        subprocess.run(['git', 'push', remote_url, branch], check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        st.error(f"推送至 GitHub 失敗：{str(e)}")
        return False

def init_database():
    if 'db_initialized' not in st.session_state:
        try:
            token = st.secrets["TOKEN"]
            url = "https://raw.githubusercontent.com/KellifizW/Q-MagV1/main/stocks.db"
            response = requests.get(url, headers={"Authorization": f"token {token}"})
            if response.status_code == 200:
                with open(DB_PATH, "wb") as f:
                    f.write(response.content)
                st.write("已從 GitHub 下載 stocks.db")
            else:
                st.write("未找到遠端 stocks.db，將創建新資料庫")
                conn = sqlite3.connect(DB_PATH)
                conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
                    Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
                    PRIMARY KEY (Date, Ticker))''')
                conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
                conn.commit()
                conn.close()
            st.session_state['db_initialized'] = True
        except Exception as e:
            st.error(f"初始化資料庫失敗：{str(e)}")
            st.session_state['db_initialized'] = False

def update_database(tickers_file=TICKERS_CSV, db_path=DB_PATH, yf_batch_size=YF_BATCH_SIZE, ms_batch_size=MS_BATCH_SIZE, repo=None, check_percentage=0.1, lookback_days=30):
    if repo is None:
        st.error("未提供 Git 倉庫物件")
        return False

    try:
        tickers_df = pd.read_csv(tickers_file)
        tickers = tickers_df['Ticker'].tolist()
        log_to_page(f"從 {tickers_file} 讀取 {len(tickers)} 檔股票", "INFO")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        conn.commit()

        cursor.execute("SELECT last_updated FROM metadata")
        last_updated = cursor.fetchone()
        current_date = datetime.now(US_EASTERN).date()
        end_date = current_date - timedelta(days=1)
        
        ticker_dates = pd.read_sql_query("SELECT Ticker, MAX(Date) as last_date FROM stocks GROUP BY Ticker", conn)
        existing_tickers = dict(zip(ticker_dates['Ticker'], pd.to_datetime(ticker_dates['last_date']).dt.date))

        num_to_check = max(1, int(len(tickers) * check_percentage))
        tickers_to_check = tickers[-num_to_check:]
        tickers_to_update = []
        default_start_date = end_date - timedelta(days=210)

        for ticker in tickers_to_check:
            last_date = existing_tickers.get(ticker)
            if not last_date:
                tickers_to_update.append(ticker)
            elif (end_date - last_date).days > 0:
                tickers_to_update.append(ticker)

        if not tickers_to_update:
            if last_updated and pd.to_datetime(last_updated[0]).date() >= end_date and len(existing_tickers) >= len(tickers):
                st.write("資料庫已是最新且完整，無需更新")
                conn.close()
                return True
            elif len(existing_tickers) < len(tickers):
                st.write(f"資料庫缺少部分股票數據（現有 {len(existing_tickers)} / 共 {len(tickers)}），將更新缺失部分")
                tickers_to_update = [t for t in tickers if t not in existing_tickers]
            else:
                st.write("資料庫數據已是最新，無需更新")
                cursor.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
                conn.commit()
                conn.close()
                return True

        try:
            api_key = st.secrets["MARKETSTACK_API_KEY"]
            log_to_page(f"獲取的 Marketstack API Key: {api_key}", "INFO")
            if not api_key:
                st.error("Marketstack API Key 是空的，請檢查 st.secrets 配置")
                return False
        except KeyError as e:
            st.error(f"未找到 MARKETSTACK_API_KEY 於 st.secrets 中：{str(e)}")
            return False

        total_batches = (len(tickers_to_update) + ms_batch_size - 1) // ms_batch_size
        total_sub_batches = sum((len(tickers_to_update[i:i + ms_batch_size]) + yf_batch_size - 1) // yf_batch_size for i in range(0, len(tickers_to_update), ms_batch_size))
        
        st.write("下載進度")
        success_bar = st.progress(0)
        success_text = st.empty()
        fail_bar = st.progress(0)
        fail_text = st.empty()
        
        success_count = [0]
        fail_count = [0]
        request_count = [0]

        for i in range(0, len(tickers_to_update), ms_batch_size):
            batch_tickers = tickers_to_update[i:i + ms_batch_size]
            batch_start_dates = [
                existing_tickers.get(ticker, default_start_date) - timedelta(days=lookback_days)
                for ticker in batch_tickers
            ]
            start_date = min(batch_start_dates)

            for j in range(0, len(batch_tickers), yf_batch_size):
                yf_batch = batch_tickers[j:j + yf_batch_size]
                data = download_with_retry(yf_batch, start=start_date, end=end_date, api_key=api_key, request_count=request_count, success_count=success_count, fail_count=fail_count)
                if data is None:
                    logger.warning(f"批次 {i // ms_batch_size + 1}/{total_batches} (子批次 {j // yf_batch_size + 1}) 下載失敗，跳過")
                
                # 更新進度條
                success_bar.progress(min(success_count[0] / total_sub_batches, 1.0))
                success_text.text(f"成功下載批次：{success_count[0]} 次")
                fail_bar.progress(min(fail_count[0] / total_sub_batches, 1.0))
                fail_text.text(f"失敗下載批次：{fail_count[0]} 次")

                if data is not None:
                    df = data.reset_index()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df.columns]

                    for ticker in yf_batch:
                        try:
                            ticker_df = df[[col for col in df.columns if col.startswith(f"{ticker}_") or col == 'Date']].copy()
                            if ticker_df.empty:
                                logger.warning(f"股票 {ticker} 的數據為空，跳過")
                                continue
                            ticker_df.columns = [col.replace(f"{ticker}_", "") for col in ticker_df.columns]
                            ticker_df['Ticker'] = ticker
                            ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.strftime('%Y-%m-%d')
                            
                            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                            missing_cols = [col for col in required_cols if col not in ticker_df.columns]
                            if missing_cols:
                                logger.error(f"股票 {ticker} 缺少必要欄位：{missing_cols}")
                                continue

                            for _, row in ticker_df.iterrows():
                                try:
                                    cursor.execute('''INSERT OR IGNORE INTO stocks 
                                        (Date, Ticker, Open, High, Low, Close, Adj_Close, Volume)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                                        row['Date'], ticker,
                                        safe_float(row['Open'], 'Open', ticker, row['Date']),
                                        safe_float(row['High'], 'High', ticker, row['Date']),
                                        safe_float(row['Low'], 'Low', ticker, row['Date']),
                                        safe_float(row['Close'], 'Close', ticker, row['Date']),
                                        safe_float(row['Close'], 'Close', ticker, row['Date']),
                                        safe_int(row['Volume'], 'Volume', ticker, row['Date'])
                                    ))
                                except ValueError as e:
                                    logger.error(f"數據轉換失敗：{str(e)}")
                                    continue
                                except sqlite3.Error as e:
                                    logger.error(f"SQLite 插入失敗，股票：{ticker}，日期：{row['Date']}，錯誤：{str(e)}")
                                    continue
                        except Exception as e:
                            logger.error(f"處理股票 {ticker} 的數據時失敗：{str(e)}")
                            continue

            conn.commit()

        cursor.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        conn.close()

        push_success = push_to_github(repo, f"Updated stocks.db with new data")
        if push_success:
            st.success("資料庫更新完成並成功推送至 GitHub")
        else:
            st.warning("資料庫更新完成，但推送至 GitHub 失敗")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("手動推送至 GitHub", key="manual_push"):
                manual_push_success = push_to_github(repo, "Manual push after update")
                if manual_push_success:
                    st.success("手動推送至 GitHub 成功")
                else:
                    st.error("手動推送至 GitHub 失敗")
        with col2:
            if os.path.exists(DB_PATH):
                with open(DB_PATH, "rb") as file:
                    st.download_button(
                        label="下載 stocks.db",
                        data=file,
                        file_name="stocks.db",
                        mime="application/octet-stream",
                        key="download_db"
                    )
            else:
                st.error("stocks.db 不存在，無法下載")
        
        return True

    except Exception as e:
        st.error(f"資料庫更新失敗：{str(e)}")
        if 'conn' in locals():
            conn.close()
        return False

def fetch_stock_data(tickers, db_path=DB_PATH, trading_days=140):  # 調整為 140 天
    try:
        if not os.path.exists(db_path):
            st.error(f"資料庫檔案 {db_path} 不存在，請先初始化或更新資料庫")
            return None, tickers
        conn = sqlite3.connect(db_path)
        start_date = (datetime.now(US_EASTERN).date() - timedelta(days=trading_days * 1.5)).strftime('%Y-%m-%d')
        query = f"SELECT * FROM stocks WHERE Ticker IN ({','.join(['?']*len(tickers))}) AND Date >= ?"
        data = pd.read_sql_query(query, conn, params=tickers + [start_date], index_col='Date', parse_dates=['Date'])
        conn.close()
        
        if data.empty:
            st.error(f"無數據：{tickers}")
            return None, tickers
        
        st.write(f"提取數據 - 股票數: {len(tickers)}, 數據長度: {len(data)}, 日期範圍: {data.index.min()} 至 {data.index.max()}")
        
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)
        
        pivoted_data = data.pivot(columns='Ticker')
        return pivoted_data, tickers
    except Exception as e:
        st.error(f"提取數據失敗：{str(e)}")
        return None, tickers

# 主程式
if __name__ == "__main__":
    st.title("股票資料庫更新工具")
    repo = init_repo()
    
    if st.button("初始化並更新資料庫"):
        init_database()
        update_database(repo=repo)
    
    if st.button("更新資料庫"):
        update_database(repo=repo)
