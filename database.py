import time
import os
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import logging
from pytz import timezone
import requests
from git_utils import GitRepoManager  # 新增模組
from file_utils import check_and_fetch_lfs_file, diagnose_db_file  # 新增模組

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定義
REPO_DIR = "."
DB_PATH = "stocks.db"
TICKERS_CSV = "Tickers.csv"
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"
US_EASTERN = timezone('US/Eastern')
BATCH_SIZE = 10

def download_with_retry(tickers, start, end, retries=2, delay=60):
    """使用 yfinance 下載數據，失敗後重試"""
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
            if data.empty:
                logger.warning(f"批次數據為空，股票：{tickers}")
                return None
            logger.info(f"成功下載 {len(tickers)} 檔股票數據")
            return data
        except Exception as e:
            logger.warning(f"yfinance 下載失敗，股票：{tickers}，錯誤：{str(e)}，重試 {attempt + 1}/{retries}")
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error(f"yfinance 下載 {tickers} 最終失敗，經過 {retries} 次嘗試")
    return None

def init_database(repo_manager):
    """初始化資料庫，支援 LFS"""
    if 'db_initialized' not in st.session_state:
        token = st.secrets.get("TOKEN", "")
        try:
            if os.path.exists(DB_PATH):
                check_and_fetch_lfs_file(DB_PATH, REPO_URL, token)
            
            if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 100:
                url = f"https://raw.githubusercontent.com/KellifizW/Q-MagV1/main/{DB_PATH}"
                response = requests.get(url, headers={"Authorization": f"token {token}"})
                if response.status_code == 200:
                    with open(DB_PATH, "wb") as f:
                        f.write(response.content)
                    repo_manager.track_lfs(DB_PATH)
                    st.write("已從 GitHub 下載 stocks.db 並配置為 LFS 檔案")
                else:
                    st.write("未找到遠端 stocks.db，將創建新資料庫")
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
                            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
                            PRIMARY KEY (Date, Ticker))''')
                        conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
                    repo_manager.track_lfs(DB_PATH)
            
            diagnostics = diagnose_db_file(DB_PATH)
            st.write("檔案診斷資訊：")
            for line in diagnostics:
                st.write(line)
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            st.session_state['db_initialized'] = True
        except sqlite3.DatabaseError as e:
            st.error(f"資料庫無效：{str(e)}，是否重建資料庫？")
            if st.button("確認重建"):
                if os.path.exists(DB_PATH):
                    os.remove(DB_PATH)
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
                        Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
                        PRIMARY KEY (Date, Ticker))''')
                    conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
                repo_manager.track_lfs(DB_PATH)
                st.session_state['db_initialized'] = True
        except Exception as e:
            st.error(f"初始化資料庫失敗：{str(e)}")
            st.session_state['db_initialized'] = False

def update_database(tickers_file=TICKERS_CSV, db_path=DB_PATH, batch_size=BATCH_SIZE, repo_manager=None, check_percentage=0.1, lookback_days=30):
    """增量更新資料庫"""
    if repo_manager is None:
        st.error("未提供 Git 倉庫管理物件")
        return False

    token = st.secrets.get("TOKEN", "")
    check_and_fetch_lfs_file(db_path, REPO_URL, token)
    diagnostics = diagnose_db_file(db_path)
    st.write("資料庫檔案診斷資訊：")
    for line in diagnostics:
        st.write(line)

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
                Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
                PRIMARY KEY (Date, Ticker))''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')

            tickers_df = pd.read_csv(tickers_file)
            tickers = tickers_df['Ticker'].tolist()
            logger.info(f"從 {tickers_file} 讀取 {len(tickers)} 檔股票")

            cursor.execute("SELECT last_updated FROM metadata")
            last_updated = cursor.fetchone()
            current_date = datetime.now(US_EASTERN).date()
            end_date = current_date - timedelta(days=1)

            ticker_dates = pd.read_sql_query("SELECT Ticker, MAX(Date) as last_date FROM stocks GROUP BY Ticker", conn)
            ticker_dates['last_date'] = pd.to_datetime(ticker_dates['last_date'], errors='coerce').dt.strftime('%Y-%m-%d')
            existing_tickers = dict(zip(ticker_dates['Ticker'], pd.to_datetime(ticker_dates['last_date']).dt.date))

            num_to_check = max(1, int(len(tickers) * check_percentage))
            tickers_to_check = tickers[-num_to_check:]
            tickers_to_update = []
            default_start_date = end_date - timedelta(days=210)

            for ticker in tickers_to_check:
                last_date = existing_tickers.get(ticker)
                if not last_date or (end_date - last_date).days > 0:
                    tickers_to_update.append(ticker)

            if not tickers_to_update and last_updated and pd.to_datetime(last_updated[0]).date() >= end_date and len(existing_tickers) >= len(tickers):
                st.write("資料庫已是最新且完整，無需更新")
                return True

            if len(existing_tickers) < len(tickers):
                tickers_to_update = [t for t in tickers if t not in existing_tickers]

            logger.info(f"需更新的股票數量：{len(tickers_to_update)}")
            total_batches = (len(tickers_to_update) + batch_size - 1) // batch_size
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            for i in range(0, len(tickers_to_update), batch_size):
                batch_tickers = tickers_to_update[i:i + batch_size]
                batch_start_dates = [existing_tickers.get(ticker, default_start_date) - timedelta(days=lookback_days) for ticker in batch_tickers]
                start_date = min(batch_start_dates)
                
                data = download_with_retry(batch_tickers, start=start_date, end=end_date)
                if data is None:
                    continue

                df = data.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]

                for ticker in batch_tickers:
                    ticker_df = df[[col for col in df.columns if col.startswith(f"{ticker}_") or col == 'Date']].copy()
                    ticker_df.columns = [col.replace(f"{ticker}_", "") for col in ticker_df.columns]
                    ticker_df['Ticker'] = ticker
                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.strftime('%Y-%m-%d')
                    records = ticker_df.to_records(index=False)
                    cursor.executemany('''INSERT OR IGNORE INTO stocks 
                        (Date, Ticker, Open, High, Low, Close, Adj_Close, Volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                        [(r.Date, r.Ticker, float(r.Open) if pd.notna(r.Open) else None, 
                          float(r.High) if pd.notna(r.High) else None, float(r.Low) if pd.notna(r.Low) else None, 
                          float(r.Close) if pd.notna(r.Close) else None, float(r.Close) if pd.notna(r.Close) else None, 
                          int(r.Volume) if pd.notna(r.Volume) else 0) for r in records])

                conn.commit()
                elapsed = time.time() - start_time
                progress = (i + batch_size) / len(tickers_to_update)
                remaining_batches = total_batches - (i // batch_size + 1)
                eta = (elapsed / (i + batch_size)) * (len(tickers_to_update) - (i + batch_size)) if i > 0 else 0
                status_text.text(f"處理中：{batch_tickers}，進度：{int(progress*100)}%，預估剩餘時間：{int(eta)}秒")
                progress_bar.progress(min(progress, 1.0))

            cursor.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
            conn.commit()

            push_success = repo_manager.push("Updated stocks.db with new data")
            if push_success:
                st.success("資料庫更新完成並成功推送至 GitHub")
            else:
                st.warning("資料庫更新完成，但推送至 GitHub 失敗，詳情請查看日誌")
                if st.button("手動推送至 GitHub"):
                    if repo_manager.push("Manual push after update"):
                        st.success("手動推送成功")
                    else:
                        st.error("手動推送失敗，請檢查網絡或認證設置")

            if os.path.exists(DB_PATH):
                with open(DB_PATH, "rb") as file:
                    st.download_button(label="下載 stocks.db", data=file, file_name="stocks.db", mime="application/octet-stream")
            return True

    except sqlite3.DatabaseError as e:
        st.error(f"資料庫錯誤：{str(e)}\n診斷資訊：\n{' '.join(diagnose_db_file(db_path))}")
        return False
    except Exception as e:
        st.error(f"資料庫更新失敗：{str(e)}")
        return False

def fetch_stock_data(tickers, db_path=DB_PATH, trading_days=70):
    """提取股票數據"""
    token = st.secrets.get("TOKEN", "")
    check_and_fetch_lfs_file(db_path, REPO_URL, token)
    
    if not os.path.exists(db_path):
        st.error(f"資料庫檔案 {db_path} 不存在，請先初始化或更新資料庫")
        return None, tickers
    
    diagnostics = diagnose_db_file(db_path)
    st.write("提取數據前的檔案診斷資訊：")
    for line in diagnostics:
        st.write(line)
    
    try:
        with sqlite3.connect(db_path) as conn:
            start_date = (datetime.now(US_EASTERN).date() - timedelta(days=trading_days * 1.5)).strftime('%Y-%m-%d')
            query = f"SELECT * FROM stocks WHERE Ticker IN ({','.join(['?']*len(tickers))}) AND Date >= ?"
            data = pd.read_sql_query(query, conn, params=tickers + [start_date], index_col='Date', parse_dates=['Date'])
        
        if data.empty:
            st.error(f"無數據：{tickers}")
            return None, tickers
        
        pivoted_data = data.pivot(columns='Ticker')
        return pivoted_data, tickers
    except sqlite3.DatabaseError as e:
        st.error(f"提取數據失敗：{str(e)}\n診斷資訊：\n{' '.join(diagnose_db_file(db_path))}")
        return None, tickers
    except Exception as e:
        st.error(f"提取數據失敗：{str(e)}")
        return None, tickers

if __name__ == "__main__":
    repo_manager = GitRepoManager(REPO_DIR, REPO_URL, st.secrets.get("TOKEN", ""))
    init_database(repo_manager)
    update_database(repo=repo_manager)
