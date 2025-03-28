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
from pytz import timezone

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
TICKERS_CSV = "Tickers.csv"
nasdaq = mcal.get_calendar('NASDAQ')
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"
MIN_TRADING_DAYS = 130
MAX_NEW_TICKERS_PER_UPDATE = 100
US_EASTERN = timezone('US/Eastern')
BATCH_SIZE = 20
INITIAL_CHECK_PERCENTAGE = 0.10  # 首次檢查 10% 股票

def init_repo():
    """初始化 Git 倉庫（適配 Streamlit 環境）"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
            logger.info("初始化 Git 倉庫")

        if "TOKEN" not in st.secrets:
            logger.error("st.secrets 中未找到 TOKEN")
            st.error("未找到 'TOKEN'，請在 Streamlit Cloud 的 Secrets 中配置為：TOKEN = \"your_token\"")
            return None
        token = st.secrets["TOKEN"]

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
    """單次推送變更到 GitHub，若失敗則提供下載選項"""
    try:
        os.chdir(REPO_DIR)
        logger.info(f"準備推送至 GitHub，訊息：{message}")
        st.write(f"調試：準備推送至 GitHub，訊息：{message}")

        if not os.path.exists(DB_PATH):
            logger.error(f"stocks.db 不存在於路徑：{DB_PATH}")
            st.error(f"stocks.db 不存在於路徑：{DB_PATH}")
            return False

        # 添加並檢查變更
        subprocess.run(['git', 'add', DB_PATH], check=True, capture_output=True, text=True)
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status.stdout:
            logger.info("無變更需要推送")
            st.write("調試：無變更需要推送")
            return True

        # 提交變更
        commit_result = subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True, text=True)
        logger.info(f"提交成功 - stdout: {commit_result.stdout}, stderr: {commit_result.stderr}")

        # 推送
        token = st.secrets["TOKEN"]
        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        branch = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip() or 'main'
        push_result = subprocess.run(['git', 'push', remote_url, branch], check=True, capture_output=True, text=True)
        logger.info(f"推送成功 - stdout: {push_result.stdout}, stderr: {push_result.stderr}")
        st.write(f"已推送至 GitHub: {message}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git 操作失敗 - 命令：{e.cmd}, stdout: {e.stdout}, stderr: {e.stderr}")
        st.error(f"推送至 GitHub 失敗：{e.stderr}")
        # 提供下載選項
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as file:
                st.download_button(
                    label="下載當前 stocks.db",
                    data=file,
                    file_name="stocks.db",
                    mime="application/octet-stream"
                )
        return False
    except Exception as e:
        logger.error(f"推送至 GitHub 發生未知錯誤：{str(e)}")
        st.error(f"推送至 GitHub 發生未知錯誤: {str(e)}")
        # 提供下載選項
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as file:
                st.download_button(
                    label="下載當前 stocks.db",
                    data=file,
                    file_name="stocks.db",
                    mime="application/octet-stream"
                )
        return False

def download_with_retry(tickers, start, end, retries=5, delay=10):
    """帶重試機制的股票數據下載"""
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False, threads=False)
            if data.empty:
                logger.warning(f"批次數據為空，股票：{tickers}，重試 {attempt + 1}/{retries}")
            else:
                logger.info(f"成功下載 {len(tickers)} 檔股票數據，股票：{tickers}，行數：{len(data)}")
                return data
        except Exception as e:
            logger.warning(f"下載失敗，股票：{tickers}，錯誤：{str(e)}，重試 {attempt + 1}/{retries}")
            time.sleep(delay * (attempt + 1))
    logger.error(f"下載 {tickers} 失敗，已達最大重試次數")
    return None

def update_database(tickers_file=TICKERS_CSV, db_path=DB_PATH, batch_size=BATCH_SIZE, repo=None):
    """增量更新資料庫並提供中斷下載選項，首次檢查 10% 股票"""
    if repo is None:
        logger.error("未提供 Git 倉庫物件")
        st.error("未提供 Git 倉庫物件")
        return False

    try:
        # 讀取股票清單
        if not os.path.exists(tickers_file):
            logger.error(f"找不到 {tickers_file}")
            st.error(f"找不到 {tickers_file}")
            return False
        tickers_df = pd.read_csv(tickers_file)
        tickers = tickers_df['Ticker'].tolist()
        logger.info(f"從 {tickers_file} 讀取股票數量：{len(tickers)} 筆")
        st.write(f"從 {tickers_file} 讀取股票數量：{len(tickers)} 筆")

        # 連接到資料庫
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 確保表存在並建立索引
        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_ticker_date ON stocks (Ticker, Date)''')

        # 獲取現有股票的最新日期
        ticker_dates = pd.read_sql_query("SELECT Ticker, MAX(Date) as last_date FROM stocks GROUP BY Ticker", conn)
        existing_tickers = dict(zip(ticker_dates['Ticker'], pd.to_datetime(ticker_dates['last_date'])))
        logger.info(f"資料庫中已有股票數量：{len(existing_tickers)} 筆")
        st.write(f"調試：資料庫中已有股票數量：{len(existing_tickers)} 筆")

        # 確定需要更新的股票（首次檢查 10%）
        current_date_et = datetime.now(US_EASTERN).date()
        end_date = current_date_et - timedelta(days=1)  # 昨天
        tickers_to_check = tickers[:max(1, int(len(tickers) * INITIAL_CHECK_PERCENTAGE))]  # 檢查 10%
        tickers_to_update = []
        for ticker in tickers_to_check:
            last_date = existing_tickers.get(ticker)
            if not last_date or (end_date - last_date.date()).days > 0:
                tickers_to_update.append(ticker)
            elif (end_date - last_date.date()).days + MIN_TRADING_DAYS > 180:
                tickers_to_update.append(ticker)  # 若數據不足 130 天，重新下載

        if not tickers_to_update:
            logger.info("測試範圍內所有股票數據已是最新，無需更新")
            st.write("測試範圍內所有股票數據已是最新，無需更新")
            conn.close()
            return True

        logger.info(f"需更新的股票數量（10% 測試）：{len(tickers_to_update)}")
        st.write(f"需更新的股票數量（10% 測試）：{len(tickers_to_update)}")

        # 分批更新
        total_batches = (len(tickers_to_update) + batch_size - 1) // batch_size
        rows_inserted_total = 0
        for i in range(0, len(tickers_to_update), batch_size):
            batch_tickers = tickers_to_update[i:i + batch_size]
            batch_num = i // batch_size + 1
            st.write(f"更新：處理批次 {batch_num}/{total_batches} ({len(batch_tickers)} 檔股票)")

            # 計算該批次的開始日期
            batch_start_dates = {t: existing_tickers.get(t, end_date - timedelta(days=180)) for t in batch_tickers}
            earliest_start = min(batch_start_dates.values()).date()
            schedule = nasdaq.schedule(start_date=earliest_start, end_date=end_date)
            if schedule.empty:
                logger.error("NASDAQ 日曆無效")
                continue
            start_date = schedule.index[0].date()

            # 下載數據
            data = download_with_retry(batch_tickers, start=start_date, end=end_date)
            if data is None:
                logger.warning(f"批次 {batch_num} 下載失敗，提供當前 stocks.db 下載")
                st.error(f"批次 {batch_num} 下載失敗，已中斷更新")
                if os.path.exists(DB_PATH):
                    with open(DB_PATH, "rb") as file:
                        st.download_button(
                            label="下載當前 stocks.db（中斷時）",
                            data=file,
                            file_name="stocks.db",
                            mime="application/octet-stream"
                        )
                conn.close()
                return False

            # 處理並插入數據
            df = data.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]
            
            all_data = []
            for ticker in batch_tickers:
                ticker_cols = [col for col in df.columns if col.startswith(f"{ticker}_") or col == 'Date']
                ticker_df = df[ticker_cols].copy()
                ticker_df.columns = [col.replace(f"{ticker}_", "") for col in ticker_df.columns]
                ticker_df['Ticker'] = ticker
                if ticker in existing_tickers:
                    ticker_df = ticker_df[pd.to_datetime(ticker_df['Date']) > existing_tickers[ticker]]
                all_data.append(ticker_df)

            if not all_data:
                continue

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
                rows_inserted_total += cursor.rowcount

            conn.commit()
            logger.info(f"批次 {batch_num}/{total_batches} 完成，插入 {rows_inserted_total} 行")
            st.write(f"調試：批次 {batch_num}/{total_batches} 完成，插入 {rows_inserted_total} 行")

        # 更新 metadata
        cursor.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        conn.close()

        # 單次推送
        if rows_inserted_total > 0:
            push_success = push_to_github(repo, f"Updated stocks.db with {rows_inserted_total} new rows (10% test)")
            if not push_success:
                st.write("推送失敗，但更新已完成")
        else:
            logger.info("無新數據插入，跳過推送")
            st.write("調試：無新數據插入，跳過推送")

        logger.info("資料庫更新完成（10% 測試）")
        st.write("資料庫更新完成（10% 測試）")
        return True

    except Exception as e:
        logger.error(f"資料庫更新失敗：{str(e)}")
        st.error(f"資料庫更新失敗：{str(e)}")
        # 提供下載選項
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as file:
                st.download_button(
                    label="下載當前 stocks.db（異常時）",
                    data=file,
                    file_name="stocks.db",
                    mime="application/octet-stream"
                )
        return False

def fetch_stock_data(tickers, db_path=DB_PATH, trading_days=70):
    """從資料庫中提取股票數據"""
    try:
        conn = sqlite3.connect(db_path)
        end_date = datetime.now(US_EASTERN).date()
        start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[-trading_days].date()
        
        query = f"SELECT * FROM stocks WHERE Ticker IN ({','.join(['?']*len(tickers))}) AND Date >= ?"
        data = pd.read_sql_query(query, conn, params=tickers + [start_date.strftime('%Y-%m-%d')], index_col='Date', parse_dates=True)
        conn.close()
        
        if data.empty:
            logger.error(f"資料庫中無符合條件的數據，股票：{tickers}")
            st.error(f"資料庫中無符合條件的數據，股票：{tickers}")
            return None
        logger.info(f"成功從資料庫提取數據，股票：{tickers}，行數：{len(data)}")
        st.write(f"調試：成功從資料庫提取數據，股票：{tickers}，行數：{len(data)}")
        return data.pivot(columns='Ticker')
    except Exception as e:
        logger.error(f"提取股票數據失敗，股票：{tickers}，錯誤：{str(e)}")
        st.error(f"提取股票數據失敗，股票：{tickers}，錯誤：{str(e)}")
        return None

if __name__ == "__main__":
    repo = init_repo()
    if repo:
        update_database(repo=repo)
