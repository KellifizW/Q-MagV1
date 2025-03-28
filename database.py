import os
import subprocess
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import streamlit as st
import logging
import time
import requests

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 定義路徑和常量
REPO_DIR = "."
DB_PATH = os.path.join(REPO_DIR, "stocks.db")
nasdaq = mcal.get_calendar('NASDAQ')
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"
MIN_TRADING_DAYS = 130  # 最小交易日數要求
MAX_NEW_TICKERS_PER_UPDATE = 100  # 每次更新最多新增股票數

# 設置 yfinance 快取位置
yf.set_tz_cache_location("/tmp/yfinance_cache")

# 自訂 requests Session，擴大連線池
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
session.mount('http://', adapter)
session.mount('https://', adapter)

def init_repo():
    """初始化 Git 倉庫，使用 st.secrets 獲取 GitHub Token"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
            logger.info("初始化 Git 倉庫")

        st.write("調試：檢查 st.secrets 內容")
        st.write(f"st.secrets 可用鍵：{list(st.secrets.keys())}")
        if "TOKEN" not in st.secrets:
            logger.error("st.secrets 中未找到 TOKEN")
            st.error("未找到 'TOKEN'，請在 Streamlit Cloud 的 Secrets 中配置為：TOKEN = \"your_token\"")
            return None
        token = st.secrets["TOKEN"]
        st.write("成功從 st.secrets 獲取 TOKEN")

        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True, text=True)
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True, capture_output=True, text=True)
        
        subprocess.run(['git', 'config', 'user.name', 'KellifizW'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'your.email@example.com'], check=True)
        subprocess.run(['git', 'config', 'core.autocrlf', 'true'], check=True)
        
        logger.info("Git 倉庫初始化完成")
        st.write("成功初始化 Git 倉庫")
        return True
    except Exception as e:
        logger.error(f"Git 倉庫初始化失敗：{str(e)}")
        st.error(f"初始化 Git 倉庫失敗：{str(e)}")
        return None

def push_to_github(repo, message="Update stocks.db"):
    """推送變更到 GitHub"""
    try:
        os.chdir(REPO_DIR)
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status.stdout.strip():
            logger.info("沒有需要提交的變更")
            st.write("沒有變更需要推送")
            return True
        
        subprocess.run(['git', 'add', 'stocks.db'], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True, text=True)
        
        if "TOKEN" not in st.secrets:
            logger.error("st.secrets 中未找到 TOKEN")
            st.error("未找到 'TOKEN'，請在 Streamlit Cloud 的 Secrets 中配置為：TOKEN = \"your_token\"")
            return False
        token = st.secrets["TOKEN"]
        st.write("成功從 st.secrets 獲取 TOKEN 用於推送")
        
        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        branch = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip() or 'main'
        subprocess.run(['git', 'push', remote_url, branch], check=True, capture_output=True, text=True)
        
        logger.info(f"成功推送至 GitHub，提交訊息：{message}")
        st.write(f"已推送至 GitHub: {message}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git 推送失敗：{e.stderr}")
        st.error(f"推送至 GitHub 失敗: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Git 推送發生未知錯誤：{str(e)}")
        st.error(f"推送至 GitHub 發生未知錯誤: {e}")
        return False

def download_with_retry(tickers, start, end, retries=5, delay=10):
    """帶重試機制的股票數據下載，增強除錯信息"""
    for attempt in range(retries):
        try:
            logger.info(f"嘗試下載 {tickers}，第 {attempt + 1} 次，日期範圍：{start} 至 {end}")
            st.write(f"嘗試下載 {len(tickers)} 檔股票：{tickers}，第 {attempt + 1} 次，日期範圍：{start} 至 {end}")
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False, threads=False, session=session)
            if data.empty:
                logger.warning(f"批次數據為空，可能是無效日期或股票已下市，股票：{tickers}，重試 {attempt + 1}/{retries}")
                st.write(f"批次數據為空，股票：{tickers}，重試 {attempt + 1}/{retries}")
            else:
                logger.info(f"成功下載 {len(tickers)} 檔股票數據，股票：{tickers}，列數：{len(data.columns)}，行數：{len(data)}")
                st.write(f"成功下載 {len(tickers)} 檔股票數據，股票：{tickers}，列數：{len(data.columns)}，行數：{len(data)}")
                return data
        except Exception as e:
            logger.warning(f"下載失敗，股票：{tickers}，錯誤：{str(e)}，重試 {attempt + 1}/{retries}")
            st.write(f"下載失敗，股票：{tickers}，錯誤：{str(e)}，重試 {attempt + 1}/{retries}")
            if "YFPricesMissingError" in str(e) and "no price data found" in str(e):
                logger.error(f"確定為無數據錯誤，可能股票已下市或日期範圍無效，股票：{tickers}，日期範圍：{start} 至 {end}")
                st.error(f"確定為無數據錯誤，可能股票已下市或日期範圍無效，股票：{tickers}，日期範圍：{start} 至 {end}")
                break
            time.sleep(delay * (attempt + 1))
    logger.error(f"下載 {tickers} 失敗，已達最大重試次數，日期範圍：{start} 至 {end}")
    st.error(f"下載 {len(tickers)} 檔股票失敗，已達最大重試次數，股票：{tickers}，日期範圍：{start} 至 {end}")
    return None

def initialize_database(tickers, db_path=DB_PATH, batch_size=20, repo=None):
    """初始化資料庫，檢查現有數據並增量添加，最多新增100個新股票"""
    if repo is None:
        logger.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        st.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 創建表（若不存在）
        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        
        logger.info(f"初始化資料庫，檢查所有股票：{len(tickers)} 筆")
        st.write(f"初始化資料庫，檢查所有股票：{len(tickers)} 筆")
        
        current_date = datetime.now().date()
        end_date = current_date - timedelta(days=1)  # 確保結束日期不過未來
        schedule = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date)
        if schedule.empty:
            logger.error("NASDAQ 日曆無效，無法確定交易日")
            st.error("NASDAQ 日曆無效，無法初始化資料庫")
            return False
        start_date = schedule.index[0].date()
        logger.info(f"初始化資料庫，日期範圍：{start_date} 至 {end_date}")
        st.write(f"初始化資料庫，日期範圍：{start_date} 至 {end_date}")
        
        # 檢查現有股票及其交易日數
        ticker_days = pd.read_sql_query("SELECT Ticker, COUNT(DISTINCT Date) as days FROM stocks GROUP BY Ticker", conn)
        existing_tickers = {row['Ticker']: row['days'] for _, row in ticker_days.iterrows()}
        
        tickers_to_download = []
        new_ticker_count = 0
        for ticker in tickers:
            if ticker in existing_tickers and existing_tickers[ticker] >= MIN_TRADING_DAYS:
                logger.info(f"股票 {ticker} 已有多於 {MIN_TRADING_DAYS} 天的數據，跳過")
                st.write(f"股票 {ticker} 已有多於 {MIN_TRADING_DAYS} 天的數據，跳過")
                continue
            if new_ticker_count >= MAX_NEW_TICKERS_PER_UPDATE:
                break
            tickers_to_download.append(ticker)
            new_ticker_count += 1
        
        if not tickers_to_download:
            logger.info("無需下載新股票，所有股票已有足夠數據或已達限制")
            st.write("無需下載新股票，所有股票已有足夠數據或已達限制")
            conn.close()
            return False
        
        logger.info(f"需下載的股票：{tickers_to_download}")
        st.write(f"需下載的股票：{tickers_to_download}")
        
        total_batches = (len(tickers_to_download) + batch_size - 1) // batch_size
        for i in range(0, len(tickers_to_download), batch_size):
            batch_tickers = tickers_to_download[i:i + batch_size]
            batch_num = i // batch_size + 1
            st.write(f"初始化：下載批次 {batch_num}/{total_batches} ({len(batch_tickers)} 檔股票：{batch_tickers})")
            
            data = download_with_retry(batch_tickers, start=start_date, end=end_date)
            if data is None:
                logger.warning(f"批次 {batch_num} 下載失敗，跳過：{batch_tickers}")
                continue
            
            df = data.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]
            
            tickers_in_batch = list({col.split('_')[0] for col in df.columns if '_' in col})
            all_data = []
            for ticker in tickers_in_batch:
                ticker_cols = [col for col in df.columns if col.startswith(f"{ticker}_") or col == 'Date']
                ticker_df = df[ticker_cols].copy()
                ticker_df.columns = [col.replace(f"{ticker}_", "") for col in ticker_df.columns]
                ticker_df['Ticker'] = ticker
                all_data.append(ticker_df)
            
            pivoted_df = pd.concat(all_data, ignore_index=True)
            pivoted_df['Date'] = pd.to_datetime(pivoted_df['Date']).dt.strftime('%Y-%m-%d')
            
            for _, row in pivoted_df.iterrows():
                cursor.execute('''INSERT OR IGNORE INTO stocks 
                    (Date, Ticker, Open, High, Low, Close, Adj_Close, Volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                    str(row['Date']), str(row['Ticker']),
                    float(row['Open']) if pd.notna(row['Open']) else None,
                    float(row['High']) if pd.notna(row['High']) else None,
                    float(row['Low']) if pd.notna(row['Low']) else None,
                    float(row['Close']) if pd.notna(row['Close']) else None,
                    float(row['Close']) if pd.notna(row['Close']) else None,
                    int(row['Volume']) if pd.notna(row['Volume']) else 0))
            
            conn.commit()
            if batch_num % 10 == 0 or batch_num == total_batches:
                push_to_github(repo, f"Initialized {batch_num} batches of stock data")
            time.sleep(5)
        
        # 更新 metadata
        cursor.execute("SELECT last_updated FROM metadata")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
        else:
            cursor.execute("UPDATE metadata SET last_updated = ?", (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        push_to_github(repo, "Final initialization of stocks.db")
        conn.close()
        
        logger.info("資料庫初始化完成")
        st.write("資料庫初始化完成")
        return True
    except Exception as e:
        logger.error(f"資料庫初始化失敗：{str(e)}")
        st.error(f"資料庫初始化失敗：{str(e)}")
        return False

def update_database(tickers, db_path=DB_PATH, batch_size=20, repo=None):
    """更新資料庫，檢查130個交易日數據，每次最多新增100個新股票"""
    if repo is None:
        logger.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        st.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 確保表存在
        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        
        logger.info(f"更新資料庫，檢查所有股票：{len(tickers)} 筆")
        st.write(f"更新資料庫，檢查所有股票：{len(tickers)} 筆")
        
        current_date = datetime.now().date()
        end_date = current_date - timedelta(days=1)  # 結束日期為昨天
        
        # 檢查 metadata 是否有記錄
        cursor.execute("SELECT last_updated FROM metadata")
        last_updated = cursor.fetchone()
        
        if last_updated:
            try:
                last_updated_date = datetime.strptime(last_updated[0], '%Y-%m-%d %H:%M:%S').date()
            except ValueError:
                last_updated_date = datetime.strptime(last_updated[0], '%Y-%m-%d').date()
            st.write(f"上次更新日期：{last_updated_date}")
        else:
            # 若無記錄，從 180 天前開始
            schedule = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date)
            if schedule.empty:
                logger.error("NASDAQ 日曆無效，無法確定交易日")
                st.error("NASDAQ 日曆無效，無法更新資料庫")
                return False
            last_updated_date = schedule.index[0].date()
            st.write(f"無上次更新記錄，從 180 天前開始：{last_updated_date}")
        
        # 設置 start_date，若上次更新日期晚於或等於昨天，則回溯 180 天
        start_date = last_updated_date
        if start_date >= end_date:
            schedule = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date)
            if schedule.empty:
                logger.error("NASDAQ 日曆無效，無法確定交易日")
                st.error("NASDAQ 日曆無效，無法更新資料庫")
                return False
            start_date = schedule.index[0].date()
            logger.info(f"上次更新日期晚於或等於昨天，調整開始日期為：{start_date}")
            st.write(f"上次更新日期晚於或等於昨天，調整開始日期為：{start_date}")
        
        schedule = nasdaq.schedule(start_date=start_date, end_date=end_date)
        if schedule.empty:
            logger.info("無新交易日數據，已跳過更新")
            st.write("無新交易日數據，已跳過更新")
            conn.close()
            return False
        
        logger.info(f"更新資料庫，日期範圍：{start_date} 至 {end_date}")
        st.write(f"更新資料庫，日期範圍：{start_date} 至 {end_date}")
        
        # 檢查現有股票及其交易日數
        ticker_days = pd.read_sql_query("SELECT Ticker, COUNT(DISTINCT Date) as days FROM stocks GROUP BY Ticker", conn)
        existing_tickers = {row['Ticker']: row['days'] for _, row in ticker_days.iterrows()}
        
        tickers_to_download = []
        new_ticker_count = 0
        for ticker in tickers:
            if ticker in existing_tickers and existing_tickers[ticker] >= MIN_TRADING_DAYS:
                logger.info(f"股票 {ticker} 已有多於 {MIN_TRADING_DAYS} 天的數據，跳過")
                st.write(f"股票 {ticker} 已有多於 {MIN_TRADING_DAYS} 天的數據，跳過")
                continue
            if new_ticker_count >= MAX_NEW_TICKERS_PER_UPDATE:
                break
            tickers_to_download.append(ticker)
            new_ticker_count += 1
        
        if not tickers_to_download:
            logger.info("無需下載新股票，已達限制或所有股票已有足夠數據")
            st.write("無需下載新股票，已達限制或所有股票已有足夠數據")
            conn.close()
            return False
        
        logger.info(f"需下載的股票：{tickers_to_download}")
        st.write(f"需下載的股票：{tickers_to_download}")
        
        total_batches = (len(tickers_to_download) + batch_size - 1) // batch_size
        for i in range(0, len(tickers_to_download), batch_size):
            batch_tickers = tickers_to_download[i:i + batch_size]
            batch_num = i // batch_size + 1
            st.write(f"更新：下載批次 {batch_num}/{total_batches} ({len(batch_tickers)} 檔股票：{batch_tickers})")
            
            data = download_with_retry(batch_tickers, start=start_date, end=end_date)
            if data is None:
                logger.warning(f"批次 {batch_num} 下載失敗，跳過：{batch_tickers}")
                continue
            
            df = data.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]
            
            tickers_in_batch = list({col.split('_')[0] for col in df.columns if '_' in col})
            all_data = []
            for ticker in tickers_in_batch:
                ticker_cols = [col for col in df.columns if col.startswith(f"{ticker}_") or col == 'Date']
                ticker_df = df[ticker_cols].copy()
                ticker_df.columns = [col.replace(f"{ticker}_", "") for col in ticker_df.columns]
                ticker_df['Ticker'] = ticker
                all_data.append(ticker_df)
            
            pivoted_df = pd.concat(all_data, ignore_index=True)
            pivoted_df['Date'] = pd.to_datetime(pivoted_df['Date']).dt.strftime('%Y-%m-%d')
            
            for _, row in pivoted_df.iterrows():
                cursor.execute('''INSERT OR IGNORE INTO stocks 
                    (Date, Ticker, Open, High, Low, Close, Adj_Close, Volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                    str(row['Date']), str(row['Ticker']),
                    float(row['Open']) if pd.notna(row['Open']) else None,
                    float(row['High']) if pd.notna(row['High']) else None,
                    float(row['Low']) if pd.notna(row['Low']) else None,
                    float(row['Close']) if pd.notna(row['Close']) else None,
                    float(row['Close']) if pd.notna(row['Close']) else None,
                    int(row['Volume']) if pd.notna(row['Volume']) else 0))
            
            conn.commit()
            if batch_num % 10 == 0 or batch_num == total_batches:
                push_to_github(repo, f"Updated {batch_num} batches of stock data")
            time.sleep(5)
        
        cursor.execute("UPDATE metadata SET last_updated = ?", (end_date.strftime('%Y-%m-%d'),))
        if not last_updated:
            cursor.execute("INSERT INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        push_to_github(repo, "Final update of stocks.db")
        conn.close()
        
        logger.info("資料庫更新完成")
        st.write("資料庫更新完成")
        return True
    except Exception as e:
        logger.error(f"資料庫更新失敗：{str(e)}")
        st.error(f"資料庫更新失敗：{str(e)}")
        return False

def fetch_stock_data(tickers, db_path=DB_PATH, trading_days=70):
    """從資料庫中提取股票數據"""
    try:
        conn = sqlite3.connect(db_path)
        end_date = datetime.now().date()
        start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[-trading_days].date()
        
        query = f"SELECT * FROM stocks WHERE Ticker IN ({','.join(['?']*len(tickers))}) AND Date >= ?"
        data = pd.read_sql_query(query, conn, params=tickers + [start_date.strftime('%Y-%m-%d')], index_col='Date', parse_dates=True)
        conn.close()
        
        if data.empty:
            logger.error(f"資料庫中無符合條件的數據，股票：{tickers}")
            st.error(f"資料庫中無符合條件的數據，請檢查初始化是否成功，股票：{tickers}")
            return None
        logger.info(f"成功從資料庫提取數據，股票：{tickers}，行數：{len(data)}")
        st.write(f"成功從資料庫提取數據，股票：{tickers}，行數：{len(data)}")
        return data.pivot(columns='Ticker')
    except Exception as e:
        logger.error(f"提取股票數據失敗，股票：{tickers}，錯誤：{str(e)}")
        st.error(f"提取股票數據失敗，股票：{tickers}，錯誤：{str(e)}")
        return None

def extend_sp500(tickers_sp500, db_path=DB_PATH, batch_size=20, repo=None):
    """擴展 S&P 500 股票數據，檢查130個交易日數據，每次最多新增100個新股票"""
    if repo is None:
        logger.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        st.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 確保表存在
        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        
        logger.info(f"檢查 S&P 500 股票：{len(tickers_sp500)} 筆")
        st.write(f"檢查 S&P 500 股票：{len(tickers_sp500)} 筆")
        
        end_date = datetime.now().date() - timedelta(days=1)
        start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[0].date()
        
        # 檢查現有股票及其交易日數
        ticker_days = pd.read_sql_query("SELECT Ticker, COUNT(DISTINCT Date) as days FROM stocks GROUP BY Ticker", conn)
        existing_tickers = {row['Ticker']: row['days'] for _, row in ticker_days.iterrows()}
        
        tickers_to_download = []
        new_ticker_count = 0
        for ticker in tickers_sp500:
            if ticker in existing_tickers and existing_tickers[ticker] >= MIN_TRADING_DAYS:
                logger.info(f"股票 {ticker} 已有多於 {MIN_TRADING_DAYS} 天的數據，跳過")
                st.write(f"股票 {ticker} 已有多於 {MIN_TRADING_DAYS} 天的數據，跳過")
                continue
            if new_ticker_count >= MAX_NEW_TICKERS_PER_UPDATE:
                break
            tickers_to_download.append(ticker)
            new_ticker_count += 1
        
        if not tickers_to_download:
            logger.info(f"無需下載新 S&P 500 股票，已達限制或所有股票已有足夠數據")
            st.write(f"無需下載新 S&P 500 股票，已達限制或所有股票已有足夠數據")
            conn.close()
            return False
        
        st.write(f"檢測到 {len(tickers_to_download)} 隻 S&P 500 股票需下載，股票：{tickers_to_download}")
        logger.info(f"補充 S&P 500 股票，日期範圍：{start_date} 至 {end_date}")
        st.write(f"補充 S&P 500 股票，日期範圍：{start_date} 至 {end_date}")
        
        total_batches = (len(tickers_to_download) + batch_size - 1) // batch_size
        for i in range(0, len(tickers_to_download), batch_size):
            batch_tickers = tickers_to_download[i:i + batch_size]
            batch_num = i // batch_size + 1
            st.write(f"補充 S&P 500：下載批次 {batch_num}/{total_batches} ({len(batch_tickers)} 檔股票：{batch_tickers})")
            
            data = download_with_retry(batch_tickers, start=start_date, end=end_date)
            if data is None:
                logger.warning(f"批次 {batch_num} 下載失敗，跳過：{batch_tickers}")
                continue
            
            df = data.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]
            
            tickers_in_batch = list({col.split('_')[0] for col in df.columns if '_' in col})
            all_data = []
            for ticker in tickers_in_batch:
                ticker_cols = [col for col in df.columns if col.startswith(f"{ticker}_") or col == 'Date']
                ticker_df = df[ticker_cols].copy()
                ticker_df.columns = [col.replace(f"{ticker}_", "") for col in ticker_df.columns]
                ticker_df['Ticker'] = ticker
                all_data.append(ticker_df)
            
            pivoted_df = pd.concat(all_data, ignore_index=True)
            pivoted_df['Date'] = pd.to_datetime(pivoted_df['Date']).dt.strftime('%Y-%m-%d')
            
            for _, row in pivoted_df.iterrows():
                cursor.execute('''INSERT OR IGNORE INTO stocks 
                    (Date, Ticker, Open, High, Low, Close, Adj_Close, Volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                    str(row['Date']), str(row['Ticker']),
                    float(row['Open']) if pd.notna(row['Open']) else None,
                    float(row['High']) if pd.notna(row['High']) else None,
                    float(row['Low']) if pd.notna(row['Low']) else None,
                    float(row['Close']) if pd.notna(row['Close']) else None,
                    float(row['Close']) if pd.notna(row['Close']) else None,
                    int(row['Volume']) if pd.notna(row['Volume']) else 0))
            
            conn.commit()
            if batch_num % 10 == 0 or batch_num == total_batches:
                push_to_github(repo, f"Extended S&P 500 with {batch_num} batches")
            time.sleep(5)
        
        cursor.execute("UPDATE metadata SET last_updated = ?", (end_date.strftime('%Y-%m-%d'),))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        push_to_github(repo, "Final S&P 500 extension")
        conn.close()
        
        logger.info("S&P 500 股票補充完成")
        st.write("S&P 500 股票補充完成")
        return True
    except Exception as e:
        logger.error(f"S&P 500 股票補充失敗：{str(e)}")
        st.error(f"S&P 500 股票補充失敗：{str(e)}")
        return False
