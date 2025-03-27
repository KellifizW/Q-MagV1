import os
import subprocess
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import streamlit as st
import logging

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

def init_repo():
    """初始化 Git 倉庫，使用 st.secrets 獲取 GitHub Token"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
            logger.info("初始化 Git 倉庫")

        # 調試：檢查 st.secrets 是否包含 github_token
        st.write("調試：檢查 st.secrets 內容")
        st.write(f"st.secrets 可用鍵：{list(st.secrets.keys())}")
        if "github_token" not in st.secrets:
            logger.error("st.secrets 中未找到 github_token")
            st.error("未找到 'github_token'，請在 Streamlit Cloud 的 Secrets 中配置為：github_token = \"your_token\"")
            return None
        token = st.secrets["github_token"]
        st.write("成功從 st.secrets 獲取 github_token")

        # 配置遠端倉庫
        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True, text=True)
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True, capture_output=True, text=True)
        
        # 配置 Git 用戶信息
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
        
        # 檢查是否有變更
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status.stdout.strip():
            logger.info("沒有需要提交的變更")
            st.write("沒有變更需要推送")
            return True
        
        # 添加並提交變更
        subprocess.run(['git', 'add', 'stocks.db'], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'add', 'tickers.csv'], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True, text=True)
        
        # 使用 st.secrets 中的 Token 推送
        if "github_token" not in st.secrets:
            logger.error("st.secrets 中未找到 github_token")
            st.error("未找到 'github_token'，請在 Streamlit Cloud 的 Secrets 中配置為：github_token = \"your_token\"")
            return False
        token = st.secrets["github_token"]
        st.write("成功從 st.secrets 獲取 github_token 用於推送")
        
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

def download_with_retry(tickers, start, end, retries=3, delay=5):
    """帶重試機制的股票數據下載"""
    for attempt in range(retries):
        try:
            logger.info(f"嘗試下載 {tickers}，第 {attempt + 1} 次")
            st.write(f"嘗試下載 {len(tickers)} 檔股票，第 {attempt + 1} 次")
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False, threads=True)
            if not data.empty:
                return data
            else:
                logger.warning(f"批次數據為空，重試 {attempt + 1}/{retries}")
                st.write(f"批次數據為空，重試 {attempt + 1}/{retries}")
        except Exception as e:
            logger.warning(f"下載失敗: {e}，重試 {attempt + 1}/{retries}")
            st.write(f"下載失敗: {e}，重試 {attempt + 1}/{retries}")
            time.sleep(delay * (attempt + 1))
    logger.error(f"下載 {tickers} 失敗，已達最大重試次數")
    st.error(f"下載 {len(tickers)} 檔股票失敗，已達最大重試次數")
    return None

def initialize_database(tickers, db_path=DB_PATH, batch_size=50, repo=None):
    """初始化資料庫"""
    if repo is None:
        logger.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        st.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 創建表格
        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        cursor.execute("DELETE FROM stocks")
        cursor.execute("DELETE FROM metadata")
        
        # 計算日期範圍
        end_date = datetime.now().date()
        start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[0].date()
        
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            batch_num = i // batch_size + 1
            st.write(f"初始化：下載批次 {batch_num}/{total_batches} ({len(batch_tickers)} 檔股票)")
            
            data = download_with_retry(batch_tickers, start=start_date, end=end_date)
            if data is None:
                continue
            
            # 處理多層索引數據
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
            
            # 插入數據
            for _, row in pivoted_df.iterrows():
                cursor.execute('''INSERT OR REPLACE INTO stocks 
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
            time.sleep(2)
        
        # 更新元數據
        cursor.execute("INSERT INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
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

def update_database(tickers, db_path=DB_PATH, batch_size=50, repo=None):
    """更新資料庫"""
    if repo is None:
        logger.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        st.error("未提供 Git 倉庫物件， compléter至 GitHub")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 檢查最後更新時間
        current_date = datetime.now().date()
        cursor.execute("SELECT last_updated FROM metadata")
        last_updated = cursor.fetchone()
        if last_updated:
            last_updated_date = datetime.strptime(last_updated[0], '%Y-%m-%d').date()
        else:
            last_updated_date = current_date - timedelta(days=1)
        
        schedule = nasdaq.schedule(start_date=last_updated_date, end_date=current_date)
        if len(schedule) <= 1:
            logger.info("今日無新數據，已跳過更新")
            st.write("今日無新數據，已跳過更新")
            conn.close()
            return False
        
        start_date = last_updated_date + timedelta(days=1)
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            batch_num = i // batch_size + 1
            st.write(f"更新：下載批次 {batch_num}/{total_batches} ({len(batch_tickers)} 檔股票)")
            
            data = download_with_retry(batch_tickers, start=start_date, end=current_date)
            if data is None:
                continue
            
            # 處理多層索引數據
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
            
            # 插入數據
            for _, row in pivoted_df.iterrows():
                cursor.execute('''INSERT OR REPLACE INTO stocks 
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
            time.sleep(2)
        
        # 更新元數據
        cursor.execute("UPDATE metadata SET last_updated = ?", (current_date.strftime('%Y-%m-%d'),))
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

def fetch_stock_data(tickers, stock_pool=None, db_path=DB_PATH, trading_days=70):
    """從資料庫中提取股票數據"""
    try:
        conn = sqlite3.connect(db_path)
        end_date = datetime.now().date()
        start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[-trading_days].date()
        
        query = f"SELECT * FROM stocks WHERE Ticker IN ({','.join(['?']*len(tickers))}) AND Date >= ?"
        data = pd.read_sql_query(query, conn, params=tickers + [start_date.strftime('%Y-%m-%d')], index_col='Date', parse_dates=True)
        conn.close()
        
        if data.empty:
            logger.error("資料庫中無符合條件的數據")
            st.error("資料庫中無符合條件的數據，請檢查初始化是否成功")
            return None
        return data.pivot(columns='Ticker')
    except Exception as e:
        logger.error(f"提取股票數據失敗：{str(e)}")
        st.error(f"提取股票數據失敗：{str(e)}")
        return None

def extend_sp500(tickers_sp500, db_path=DB_PATH, batch_size=50, repo=None):
    """擴展 S&P 500 股票數據"""
    if repo is None:
        logger.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        st.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        return False

    try:
        conn = sqlite3.connect(db_path)
        existing_tickers = pd.read_sql_query("SELECT DISTINCT Ticker FROM stocks", conn)['Ticker'].tolist()
        missing_tickers = [ticker for ticker in tickers_sp500 if ticker not in existing_tickers]
        
        if not missing_tickers:
            logger.info("無缺失的 S&P 500 股票")
            st.write("無缺失的 S&P 500 股票")
            conn.close()
            return False
        
        st.write(f"檢測到 {len(missing_tickers)} 隻 S&P 500 股票缺失，正在補充...")
        end_date = datetime.now().date()
        start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[0].date()
        
        total_batches = (len(missing_tickers) + batch_size - 1) // batch_size
        for i in range(0, len(missing_tickers), batch_size):
            batch_tickers = missing_tickers[i:i + batch_size]
            batch_num = i // batch_size + 1
            st.write(f"補充 S&P 500：下載批次 {batch_num}/{total_batches} ({len(batch_tickers)} 檔股票)")
            
            data = download_with_retry(batch_tickers, start=start_date, end=end_date)
            if data is None:
                continue
            
            # 處理多層索引數據
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
            
            # 插入數據
            cursor = conn.cursor()
            for _, row in pivoted_df.iterrows():
                cursor.execute('''INSERT OR REPLACE INTO stocks 
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
            time.sleep(2)
        
        conn.close()
        if batch_num % 10 != 0:
            push_to_github(repo, "Final S&P 500 extension")
        
        logger.info("S&P 500 股票補充完成")
        st.write("S&P 500 股票補充完成")
        return True
    except Exception as e:
        logger.error(f"S&P 500 股票補充失敗：{str(e)}")
        st.error(f"S&P 500 股票補充失敗：{str(e)}")
        return False
