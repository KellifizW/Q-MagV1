import time
import os
import sqlite3
import pandas as pd
import yfinance as yf
yf.set_tz_cache_location("./yfinance_cache")
from datetime import datetime, timedelta
import streamlit as st
import logging
from pytz import timezone
import requests
from git_utils import GitRepoManager
from file_utils import check_and_fetch_lfs_file, diagnose_db_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_DIR = "."
DB_PATH = "stocks.db"
TICKERS_CSV = "Tickers.csv"
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"
US_EASTERN = timezone('US/Eastern')
BATCH_SIZE = 10

def get_last_trading_day(current_date):
    current_date = current_date.date() if isinstance(current_date, datetime) else current_date
    while current_date.weekday() >= 5:
        current_date -= timedelta(days=1)
    return current_date

def get_next_trading_day(date):
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day

def download_with_retry(tickers, start, end, retries=2, delay=60):
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
            if data.empty:
                logger.warning(f"批次數據為空，股票：{tickers}")
                return None
            logger.info(f"成功下載股票數據：{tickers}")
            return data
        except Exception as e:
            logger.warning(f"yfinance 下載失敗，股票：{tickers}，錯誤：{str(e)}，重試 {attempt + 1}/{retries}")
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error(f"yfinance 下載 {tickers} 最終失敗，經過 {retries} 次嘗試")
    return None

def fetch_yfinance_data(ticker, trading_days=136):
    try:
        end_date = get_last_trading_day(datetime.now(US_EASTERN))
        start_date = (end_date - timedelta(days=trading_days * 1.5)).strftime('%Y-%m-%d')
        end_date = get_next_trading_day(end_date).strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            logger.warning(f"無法從 yfinance 獲取 {ticker} 的數據")
            return None
        data = data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
        data['Ticker'] = ticker
        data = data.reset_index()
        # 設置日期索引並指定時區
        data.set_index('Date', inplace=True)
        data.index = pd.to_datetime(data.index).tz_localize('US/Eastern')  # 明確指定時區為 US/Eastern
        data_pivoted = data.pivot(columns='Ticker')
        data_pivoted.columns = pd.MultiIndex.from_tuples([(col[0], col[1]) for col in data_pivoted.columns], names=[None, 'Ticker'])
        return data_pivoted
    except Exception as e:
        logger.error(f"查詢 {ticker} 數據失敗：{str(e)}")
        return None

def init_database(repo_manager):
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
                conn.execute('''CREATE INDEX IF NOT EXISTS idx_ticker_date ON stocks (Ticker, Date)''')
                conn.commit()
            
            diagnostics = diagnose_db_file(DB_PATH)
            st.write("資料庫診斷資訊：")
            if diagnostics:
                file_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
                st.write(f"檢查檔案：{DB_PATH}")
                st.write(f"檔案大小：{file_size_mb:.2f} MB")
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
                    conn.execute('''CREATE INDEX IF NOT EXISTS idx_ticker_date ON stocks (Ticker, Date)''')
                repo_manager.track_lfs(DB_PATH)
                st.session_state['db_initialized'] = True
        except Exception as e:
            st.error(f"初始化資料庫失敗：{str(e)}")
            st.session_state['db_initialized'] = False

def update_database(tickers_file=TICKERS_CSV, db_path=DB_PATH, batch_size=BATCH_SIZE, repo_manager=None,
                    check_percentage=0.1, lookback_days=30):
    if repo_manager is None:
        st.error("未提供 Git 倉庫管理物件")
        return False

    token = st.secrets.get("TOKEN", "")
    check_and_fetch_lfs_file(db_path, REPO_URL, token)
    diagnostics = diagnose_db_file(db_path)
    st.write("資料庫診斷資訊：")
    if diagnostics:
        file_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
        st.write(f"檢查檔案：{DB_PATH}")
        st.write(f"檔案大小：{file_size_mb:.2f} MB")

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
                Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
                PRIMARY KEY (Date, Ticker))''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
            cursor.execute('''CREATE INDEX IF NOT EXISTS idx_ticker_date ON stocks (Ticker, Date)''')

            tickers_df = pd.read_csv(tickers_file)
            tickers = tickers_df['Ticker'].tolist()
            logger.info(f"從 {tickers_file} 讀取 {len(tickers)} 檔股票")

            cursor.execute("SELECT last_updated FROM metadata")
            last_updated = cursor.fetchone()
            current_date = datetime.now(US_EASTERN)
            end_date = get_last_trading_day(current_date)
            actual_current_date = datetime.now(US_EASTERN).date()
            if actual_current_date < end_date:
                end_date = get_last_trading_day(actual_current_date)
            end_date_for_download = get_next_trading_day(end_date)

            ticker_dates = pd.read_sql_query("SELECT Ticker, MAX(Date) as last_date FROM stocks GROUP BY Ticker", conn)
            ticker_dates['last_date'] = pd.to_datetime(ticker_dates['last_date'], errors='coerce').dt.date
            existing_tickers = dict(zip(ticker_dates['Ticker'], ticker_dates['last_date']))

            num_to_check = max(1, int(len(tickers) * check_percentage))
            tickers_to_check = tickers[-num_to_check:]
            tickers_to_update = []
            ticker_start_dates = {}
            default_start_date = end_date - timedelta(days=210)
            required_start_date = end_date - timedelta(days=210 + lookback_days)

            for ticker in tickers_to_check:
                last_date = existing_tickers.get(ticker)
                if not last_date or last_date < end_date:
                    tickers_to_update.append(ticker)
                    ticker_start_dates[ticker] = last_date + timedelta(days=1) if last_date else default_start_date

            if not tickers_to_update and last_updated and pd.to_datetime(last_updated[0]).date() >= end_date and len(existing_tickers) >= len(tickers):
                st.write(f"資料庫已是最新至 {end_date}，無需更新")
                return True

            if len(existing_tickers) < len(tickers):
                for ticker in tickers:
                    if ticker not in existing_tickers:
                        tickers_to_update.append(ticker)
                        ticker_start_dates[ticker] = default_start_date

            logger.info(f"需更新的股票數量：{len(tickers_to_update)}")

            total_batches = (len(tickers_to_update) + batch_size - 1) // batch_size
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            for i in range(0, len(tickers_to_update), batch_size):
                batch_tickers = tickers_to_update[i:i + batch_size]
                batch_start_dates = [ticker_start_dates[ticker] for ticker in batch_tickers]
                start_date = min(batch_start_dates)
                if start_date >= end_date:
                    start_date = end_date - timedelta(days=1)
                data = download_with_retry(batch_tickers, start=start_date, end=end_date_for_download)

                if data is not None:
                    if data.empty:
                        st.warning(f"批次 {batch_tickers} 返回空數據，嘗試單獨下載")
                        for ticker in batch_tickers:
                            single_data = yf.download(ticker, start=start_date, end=end_date_for_download, progress=False)
                            if not single_data.empty:
                                single_df = single_data.reset_index()
                                single_df['Ticker'] = ticker
                                single_df['Date'] = pd.to_datetime(single_df['Date']).dt.strftime('%Y-%m-%d')
                                records = single_df.to_records(index=False)
                                cursor.executemany('''INSERT OR IGNORE INTO stocks 
                                    (Date, Ticker, Open, High, Low, Close, Adj_Close, Volume)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                                                   [(r.Date, r.Ticker, float(r.Open) if pd.notna(r.Open) else None,
                                                     float(r.High) if pd.notna(r.High) else None,
                                                     float(r.Low) if pd.notna(r.Low) else None,
                                                     float(r.Close) if pd.notna(r.Close) else None,
                                                     float(r.Close) if pd.notna(r.Close) else None,
                                                     int(r.Volume) if pd.notna(r.Volume) else 0) for r in records])
                    else:
                        df = data.reset_index()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]

                        for ticker in batch_tickers:
                            ticker_df = df[[col for col in df.columns if col.startswith(f"{ticker}_") or col == 'Date']].copy()
                            ticker_df.columns = [col.replace(f"{ticker}_", "") for col in ticker_df.columns]
                            expected_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
                            if not expected_columns.issubset(ticker_df.columns):
                                st.error(f"錯誤：ticker_df 缺少必要列，當前列名：{ticker_df.columns.tolist()}")
                                continue

                            ticker_df['Ticker'] = ticker
                            ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.strftime('%Y-%m-%d')
                            records = ticker_df.to_records(index=False)
                            cursor.executemany('''INSERT OR IGNORE INTO stocks 
                                (Date, Ticker, Open, High, Low, Close, Adj_Close, Volume)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                                               [(r.Date, r.Ticker, float(r.Open) if pd.notna(r.Open) else None,
                                                 float(r.High) if pd.notna(r.High) else None,
                                                 float(r.Low) if pd.notna(r.Low) else None,
                                                 float(r.Close) if pd.notna(r.Close) else None,
                                                 float(r.Close) if pd.notna(r.Close) else None,
                                                 int(r.Volume) if pd.notna(r.Volume) else 0) for r in records])

                conn.commit()
                elapsed = time.time() - start_time
                progress = min((i + batch_size) / len(tickers_to_update), 1.0)
                eta = (elapsed / (i + batch_size)) * (len(tickers_to_update) - (i + batch_size)) if i > 0 else 0
                status_text.text(f"處理中：{batch_tickers}，進度：{int(progress * 100)}%，預估剩餘時間：{int(eta)}秒")
                progress_bar.progress(progress)

            cursor.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)",
                           (end_date.strftime('%Y-%m-%d'),))
            conn.commit()

            if not os.path.exists(DB_PATH):
                st.error(f"資料庫檔案 {DB_PATH} 不存在，無法推送")
                return False

            push_success = repo_manager.push(DB_PATH, f"Updated stocks.db to {end_date}")
            if push_success:
                st.success(f"資料庫更新至 {end_date} 並成功推送至 GitHub")
            else:
                st.warning("資料庫更新完成，但推送至 GitHub 失敗")
            return True

    except sqlite3.DatabaseError as e:
        st.error(f"資料庫錯誤：{str(e)}")
        return False
    except Exception as e:
        st.error(f"資料庫更新失敗：{str(e)}")
        return False

def fetch_stock_data(tickers, db_path=DB_PATH, trading_days=70, batch_size=50):
    token = st.secrets.get("TOKEN", "")
    check_and_fetch_lfs_file(db_path, REPO_URL, token)

    if not os.path.exists(db_path):
        st.error(f"資料庫檔案 {db_path} 不存在")
        return None, tickers

    try:
        current_date = datetime.now(US_EASTERN)
        end_date = get_last_trading_day(current_date)  # 確保不超過當前日期
        start_date = (end_date - timedelta(days=trading_days * 1.5)).strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        all_data = []

        with sqlite3.connect(db_path) as conn:
            for i in range(0, len(tickers), batch_size):
                batch_tickers = tickers[i:i + batch_size]
                query = f"SELECT * FROM stocks WHERE Ticker IN ({','.join(['?'] * len(batch_tickers))}) AND Date >= ? AND Date <= ?"
                batch_data = pd.read_sql_query(query, conn, params=batch_tickers + [start_date, end_date_str],
                                               index_col='Date', parse_dates=['Date'])
                if not batch_data.empty:
                    all_data.append(batch_data)

        if not all_data:
            st.error(f"無數據：{tickers}")
            return None, tickers

        data = pd.concat(all_data)
        pivoted_data = data.pivot(columns='Ticker')
        st.write(f"數據提取至最後交易日：{end_date}")
        return pivoted_data, tickers
    except sqlite3.DatabaseError as e:
        st.error(f"提取數據失敗：{str(e)}")
        return None, tickers
    except Exception as e:
        st.error(f"提取數據失敗：{str(e)}")
        return None, tickers

if __name__ == "__main__":
    repo_manager = GitRepoManager(REPO_DIR, REPO_URL, st.secrets.get("TOKEN", ""))
    init_database(repo_manager)
    update_database(repo_manager=repo_manager)
