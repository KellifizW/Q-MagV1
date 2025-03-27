import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import os
import git
import shutil
import streamlit as st
import time

REPO_DIR = "repo"
DB_PATH = os.path.join(REPO_DIR, "stocks.db")
REPO_URL = f"https://{st.secrets['TOKEN']}@github.com/KellifizW/Q-MagV1.git"
nasdaq = mcal.get_calendar('NASDAQ')

def clone_repo():
    try:
        if os.path.exists(REPO_DIR):
            shutil.rmtree(REPO_DIR)
        repo = git.Repo.clone_from(REPO_URL, REPO_DIR)
        return repo
    except KeyError:
        st.error("未找到 TOKEN 秘密，請在 Streamlit Cloud 的 Secrets 中配置")
        return None
    except Exception as e:
        st.error(f"克隆倉庫失敗：{str(e)}")
        return None

def push_to_github(repo, message="Update stocks.db"):
    repo.git.add("stocks.db")
    repo.git.commit(m=message)
    repo.git.push()

def download_with_retry(tickers, start, end, retries=3, delay=5):
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, group_by="ticker", progress=False, threads=True)
            if not data.empty:
                return data
            else:
                st.write(f"批次數據為空，重試 {attempt + 1}/{retries}")
        except Exception as e:
            st.write(f"下載失敗: {e}，重試 {attempt + 1}/{retries}")
            time.sleep(delay * (attempt + 1))
    st.error(f"下載 {tickers} 失敗，已達最大重試次數")
    return pd.DataFrame()

def initialize_database(tickers, db_path=DB_PATH, batch_size=50):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 定義表結構
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT,
            Ticker TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Adj_Close REAL,
            Volume INTEGER,
            PRIMARY KEY (Date, Ticker)
        )
    """)
    conn.commit()

    end_date = datetime.now().date()
    start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[0].date()
    
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        st.write(f"初始化：下載批次 {i//batch_size + 1}/{len(tickers)//batch_size + 1} ({len(batch_tickers)} 檔股票)...")
        data = download_with_retry(batch_tickers, start=start_date, end=end_date)
        if not data.empty:
            if len(batch_tickers) == 1:
                data = pd.DataFrame(data).assign(Ticker=batch_tickers[0])
            else:
                data = data.stack(level=1, future_stack=True).reset_index().rename(columns={'level_1': 'Ticker'})
            
            # 打印數據欄位以調試
            st.write(f"批次 {i//batch_size + 1} 的數據欄位: {list(data.columns)}")
            
            # 確保數據欄位與表結構匹配
            expected_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            data = data.rename(columns={'Adj Close': 'Adj_Close'})  # 修正可能的欄位名稱
            if 'Price' in data.columns:
                st.warning("發現意外的 'Price' 欄位，將其映射為 'Close'")
                data = data.rename(columns={'Price': 'Close'})
            data = data[[col for col in expected_columns if col in data.columns]]  # 只保留匹配的欄位
            
            # 插入數據庫
            data.to_sql('stocks', conn, if_exists='append', index=False)
        time.sleep(2)
    
    pd.DataFrame({'last_updated': [end_date.strftime('%Y-%m-%d')]}).to_sql('metadata', conn, if_exists='replace', index=False)
    conn.close()

def update_database(tickers, db_path=DB_PATH, batch_size=50):
    conn = sqlite3.connect(db_path)
    current_date = datetime.now().date()
    last_updated = pd.read_sql_query("SELECT last_updated FROM metadata", conn).iloc[0, 0]
    last_updated_date = datetime.strptime(last_updated, '%Y-%m-%d').date()
    
    schedule = nasdaq.schedule(start_date=last_updated_date, end_date=current_date)
    if len(schedule) <= 1:
        st.write("今日無新數據，已跳過更新")
        conn.close()
        return False
    
    start_date = last_updated_date + timedelta(days=1)
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        st.write(f"更新：下載批次 {i//batch_size + 1}/{len(tickers)//batch_size + 1} ({len(batch_tickers)} 檔股票)...")
        data = download_with_retry(batch_tickers, start=start_date, end=current_date)
        if not data.empty:
            if len(batch_tickers) == 1:
                data = pd.DataFrame(data).assign(Ticker=batch_tickers[0])
            else:
                data = data.stack(level=1, future_stack=True).reset_index().rename(columns={'level_1': 'Ticker'})
            
            st.write(f"批次 {i//batch_size + 1} 的數據欄位: {list(data.columns)}")
            expected_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            data = data.rename(columns={'Adj Close': 'Adj_Close'})
            if 'Price' in data.columns:
                st.warning("發現意外的 'Price' 欄位，將其映射為 'Close'")
                data = data.rename(columns={'Price': 'Close'})
            data = data[[col for col in expected_columns if col in data.columns]]
            
            data.to_sql('stocks', conn, if_exists='append', index=False)
        time.sleep(2)
    
    pd.DataFrame({'last_updated': [current_date.strftime('%Y-%m-%d')]}).to_sql('metadata', conn, if_exists='replace', index=False)
    conn.close()
    return True

def extend_sp500(tickers_sp500, db_path=DB_PATH, batch_size=50):
    conn = sqlite3.connect(db_path)
    existing_tickers = pd.read_sql_query("SELECT DISTINCT Ticker FROM stocks", conn)['Ticker'].tolist()
    missing_tickers = [ticker for ticker in tickers_sp500 if ticker not in existing_tickers]
    
    if missing_tickers:
        st.write(f"檢測到 {len(missing_tickers)} 隻 S&P 500 股票缺失，正在補充...")
        end_date = datetime.now().date()
        start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[0].date()
        
        for i in range(0, len(missing_tickers), batch_size):
            batch_tickers = missing_tickers[i:i + batch_size]
            st.write(f"補充 S&P 500：下載批次 {i//batch_size + 1}/{len(missing_tickers)//batch_size + 1} ({len(batch_tickers)} 檔股票)...")
            data = download_with_retry(batch_tickers, start=start_date, end=end_date)
            if not data.empty:
                if len(batch_tickers) == 1:
                    data = pd.DataFrame(data).assign(Ticker=batch_tickers[0])
                else:
                    data = data.stack(level=1, future_stack=True).reset_index().rename(columns={'level_1': 'Ticker'})
                
                st.write(f"批次 {i//batch_size + 1} 的數據欄位: {list(data.columns)}")
                expected_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                data = data.rename(columns={'Adj Close': 'Adj_Close'})
                if 'Price' in data.columns:
                    st.warning("發現意外的 'Price' 欄位，將其映射為 'Close'")
                    data = data.rename(columns={'Price': 'Close'})
                data = data[[col for col in expected_columns if col in data.columns]]
                
                data.to_sql('stocks', conn, if_exists='append', index=False)
            time.sleep(2)
    
    conn.close()
    return len(missing_tickers) > 0

def fetch_stock_data(tickers, stock_pool=None, db_path=DB_PATH, trading_days=70):
    conn = sqlite3.connect(db_path)
    if stock_pool == "S&P 500":
        from screening import get_sp500
        tickers_sp500 = get_sp500()
        if extend_sp500(tickers_sp500):
            st.write("S&P 500 股票補充完成，更新資料庫...")
    
    end_date = datetime.now().date()
    start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[-trading_days].date()
    
    query = f"SELECT * FROM stocks WHERE Ticker IN ({','.join(['?']*len(tickers))}) AND Date >= ?"
    data = pd.read_sql_query(query, conn, params=tickers + [start_date.strftime('%Y-%m-%d')], index_col='Date', parse_dates=True)
    conn.close()
    
    if data.empty:
        return None
    return data.pivot(columns='Ticker')
