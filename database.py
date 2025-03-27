import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import os
import git
import streamlit as st
import time

REPO_DIR = "."  # Streamlit Cloud 中，檔案位於應用根目錄
DB_PATH = os.path.join(REPO_DIR, "stocks.db")  # 即 ./stocks.db
REPO_URL = f"https://{st.secrets['TOKEN']}@github.com/KellifizW/Q-MagV1.git"
nasdaq = mcal.get_calendar('NASDAQ')

def init_repo():
    try:
        # 初始化現有的 Git 倉庫，而不是重新克隆
        repo = git.Repo(REPO_DIR)
        with repo.config_writer() as git_config:
            git_config.set_value('credential', 'helper', 'store --file=.git-credentials')
        with open('.git-credentials', 'w') as f:
            f.write(f"https://{st.secrets['TOKEN']}:@github.com")
        st.write("成功初始化現有 Git 倉庫")
        return repo
    except KeyError:
        st.error("未找到 TOKEN 秘密，請在 Streamlit Cloud 的 Secrets 中配置")
        return None
    except Exception as e:
        st.error(f"初始化 Git 倉庫失敗：{str(e)}")
        return None

def push_to_github(repo, message="Update stocks.db"):
    try:
        repo.git.config('user.name', 'KellifizW')
        repo.git.config('user.email', 'kellyindabox@gmail.com')
        repo.git.add("stocks.db")
        repo.git.add("tickers.csv")  # 如果 tickers.csv 可能變更，也推送
        if repo.is_dirty():
            repo.git.commit(m=message)
            repo.git.push()
            st.write(f"已推送至 GitHub: {message}")
        else:
            st.write("沒有變更需要推送")
    except git.GitCommandError as e:
        st.error(f"推送至 GitHub 失敗: {e}")
        st.error(f"命令: {e.command}")
        st.error(f"錯誤輸出: {e.stderr}")
    except Exception as e:
        st.error(f"推送至 GitHub 發生未知錯誤: {e}")

def download_with_retry(tickers, start, end, retries=3, delay=5):
    for attempt in range(retries):
        try:
            st.write(f"嘗試下載 {tickers}，第 {attempt + 1} 次")
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

def initialize_database(tickers, db_path=DB_PATH, batch_size=50, repo=None):
    if repo is None:
        st.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
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
    
    batch_count = 0
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        batch_count += 1
        st.write(f"初始化：下載批次 {batch_count}/{total_batches} ({len(batch_tickers)} 檔股票): {batch_tickers}")
        data = download_with_retry(batch_tickers, start=start_date, end=end_date)
        if not data.empty:
            if len(batch_tickers) == 1:
                data = pd.DataFrame(data).assign(Ticker=batch_tickers[0])
            else:
                data = data.stack(level=1, future_stack=True).reset_index().rename(columns={'level_1': 'Ticker'})
            
            data.columns = [col.replace('Adj Close', 'Adj_Close') for col in data.columns]
            expected_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
            data = data[[col for col in expected_columns if col in data.columns]]
            st.write(f"批次 {batch_count} 下載成功，數據筆數: {len(data)}")
            data.to_sql('stocks', conn, if_exists='append', index=False)
        else:
            st.warning(f"批次 {batch_count} 無數據，下載的股票: {batch_tickers}")
        
        if batch_count % 10 == 0 or batch_count == total_batches:
            conn.commit()
            st.write(f"批次 {batch_count} 已提交至資料庫")
            push_to_github(repo, f"Initialized {batch_count} batches of stock data")
        
        time.sleep(2)
    
    pd.DataFrame({'last_updated': [end_date.strftime('%Y-%m-%d')]}).to_sql('metadata', conn, if_exists='replace', index=False)
    conn.commit()
    push_to_github(repo, "Final initialization of stocks.db")
    conn.close()
    st.write("資料庫初始化完成")

def update_database(tickers, db_path=DB_PATH, batch_size=50, repo=None):
    if repo is None:
        st.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        return False

    conn = sqlite3.connect(db_path)
    current_date = datetime.now().date()
    try:
        last_updated = pd.read_sql_query("SELECT last_updated FROM metadata", conn).iloc[0, 0]
        last_updated_date = datetime.strptime(last_updated, '%Y-%m-%d').date()
    except Exception:
        st.write("無法讀取上次更新日期，假設需要初始化")
        last_updated_date = current_date - timedelta(days=1)
    
    schedule = nasdaq.schedule(start_date=last_updated_date, end_date=current_date)
    if len(schedule) <= 1:
        st.write("今日無新數據，已跳過更新")
        conn.close()
        return False
    
    start_date = last_updated_date + timedelta(days=1)
    batch_count = 0
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        batch_count += 1
        st.write(f"更新：下載批次 {batch_count}/{total_batches} ({len(batch_tickers)} 檔股票): {batch_tickers}")
        data = download_with_retry(batch_tickers, start=start_date, end=current_date)
        if not data.empty:
            if len(batch_tickers) == 1:
                data = pd.DataFrame(data).assign(Ticker=batch_tickers[0])
            else:
                data = data.stack(level=1, future_stack=True).reset_index().rename(columns={'level_1': 'Ticker'})
            
            data.columns = [col.replace('Adj Close', 'Adj_Close') for col in data.columns]
            expected_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
            data = data[[col for col in expected_columns if col in data.columns]]
            st.write(f"批次 {batch_count} 下載成功，數據筆數: {len(data)}")
            data.to_sql('stocks', conn, if_exists='append', index=False)
        else:
            st.warning(f"批次 {batch_count} 無數據，下載的股票: {batch_tickers}")
        
        if batch_count % 10 == 0 or batch_count == total_batches:
            conn.commit()
            st.write(f"批次 {batch_count} 已提交至資料庫")
            push_to_github(repo, f"Updated {batch_count} batches of stock data")
        
        time.sleep(2)
    
    pd.DataFrame({'last_updated': [current_date.strftime('%Y-%m-%d')]}).to_sql('metadata', conn, if_exists='replace', index=False)
    conn.commit()
    push_to_github(repo, "Final update of stocks.db")
    conn.close()
    st.write("資料庫更新完成")
    return True

def fetch_stock_data(tickers, stock_pool=None, db_path=DB_PATH, trading_days=70):
    conn = sqlite3.connect(db_path)
    if stock_pool == "S&P 500":
        from screening import get_sp500
        tickers_sp500 = get_sp500()
        repo = init_repo()
        if repo and extend_sp500(tickers_sp500, repo=repo):
            st.write("S&P 500 股票補充完成，更新資料庫...")
    
    end_date = datetime.now().date()
    start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[-trading_days].date()
    
    query = f"SELECT * FROM stocks WHERE Ticker IN ({','.join(['?']*len(tickers))}) AND Date >= ?"
    data = pd.read_sql_query(query, conn, params=tickers + [start_date.strftime('%Y-%m-%d')], index_col='Date', parse_dates=True)
    conn.close()
    
    if data.empty and stock_pool != "S&P 500":
        st.error("資料庫中無符合條件的數據，請檢查初始化是否成功")
        return None
    return data.pivot(columns='Ticker')

def extend_sp500(tickers_sp500, db_path=DB_PATH, batch_size=50, repo=None):
    if repo is None:
        st.error("未提供 Git 倉庫物件，無法推送至 GitHub")
        return False

    conn = sqlite3.connect(db_path)
    existing_tickers = pd.read_sql_query("SELECT DISTINCT Ticker FROM stocks", conn)['Ticker'].tolist()
    missing_tickers = [ticker for ticker in tickers_sp500 if ticker not in existing_tickers]
    
    if not missing_tickers:
        conn.close()
        return False
    
    st.write(f"檢測到 {len(missing_tickers)} 隻 S&P 500 股票缺失，正在補充...")
    end_date = datetime.now().date()
    start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[0].date()
    
    batch_count = 0
    total_batches = (len(missing_tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(missing_tickers), batch_size):
        batch_tickers = missing_tickers[i:i + batch_size]
        batch_count += 1
        st.write(f"補充 S&P 500：下載批次 {batch_count}/{total_batches} ({len(batch_tickers)} 檔股票)...")
        data = download_with_retry(batch_tickers, start=start_date, end=end_date)
        if not data.empty:
            if len(batch_tickers) == 1:
                data = pd.DataFrame(data).assign(Ticker=batch_tickers[0])
            else:
                data = data.stack(level=1, future_stack=True).reset_index().rename(columns={'level_1': 'Ticker'})
            
            data.columns = [col.replace('Adj Close', 'Adj_Close') for col in data.columns]
            expected_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
            data = data[[col for col in expected_columns if col in data.columns]]
            data.to_sql('stocks', conn, if_exists='append', index=False)
        
        if batch_count % 10 == 0 or batch_count == total_batches:
            conn.commit()
            push_to_github(repo, f"Extended S&P 500 with {batch_count} batches")
        
        time.sleep(2)
    
    conn.commit()
    if batch_count % 10 != 0:
        push_to_github(repo, "Final S&P 500 extension")
    conn.close()
    return True
