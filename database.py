import time
import os
import subprocess
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import logging
from pytz import timezone

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定義
REPO_DIR = "."
DB_PATH = "stocks.db"
TICKERS_CSV = "Tickers.csv"
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"
US_EASTERN = timezone('US/Eastern')
BATCH_SIZE = 20

def download_with_retry(tickers, start, end, retries=2, delay=60):
    """使用 yfinance 下載數據，失敗後等待指定秒數後重試"""
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
            if data.empty:
                logger.warning(f"批次數據為空，股票：{tickers}")
                return None
            logger.info(f"成功下載 {len(tickers)} 檔股票數據 (yfinance)")
            return data
        except Exception as e:
            logger.warning(f"yfinance 下載失敗，股票：{tickers}，錯誤：{str(e)}，重試 {attempt + 1}/{retries}")
            if attempt < retries - 1:  # 最後一次失敗不等待
                time.sleep(delay)
    logger.error(f"yfinance 下載 {tickers} 最終失敗，經過 {retries} 次嘗試")
    return None

def init_repo():
    """初始化 Git 倉庫"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
            logger.info("初始化 Git 倉庫")

        if "TOKEN" not in st.secrets:
            st.error("未找到 'TOKEN'，請在 Streamlit Cloud 的 Secrets 中配置")
            return None
        token = st.secrets["TOKEN"]

        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True, text=True)
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True, capture_output=True, text=True)
        
        subprocess.run(['git', 'config', 'user.name', 'KellifizW'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'your.email@example.com'], check=True)
        logger.info("Git 倉庫初始化完成")
        return True
    except Exception as e:
        st.error(f"初始化 Git 倉庫失敗：{str(e)}")
        return None

def push_to_github(repo, message="Update stocks.db"):
    """推送變更到 GitHub"""
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
    """從 GitHub 下載初始資料庫或創建新資料庫"""
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

def update_database(tickers_file=TICKERS_CSV, db_path=DB_PATH, batch_size=BATCH_SIZE, repo=None, check_percentage=0.1, lookback_days=30):
    """增量更新資料庫，包含完整性檢查"""
    if repo is None:
        st.error("未提供 Git 倉庫物件")
        return False

    try:
        # 讀取股票清單
        tickers_df = pd.read_csv(tickers_file)
        tickers = tickers_df['Ticker'].tolist()
        logger.info(f"從 {tickers_file} 讀取 {len(tickers)} 檔股票")

        # 連接到資料庫
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        conn.commit()

        # 檢查最後更新日期
        cursor.execute("SELECT last_updated FROM metadata")
        last_updated = cursor.fetchone()
        current_date = datetime.now(US_EASTERN).date()
        end_date = current_date - timedelta(days=1)
        logger.info(f"當前日期：{current_date}，結束日期：{end_date}")
        logger.info(f"資料庫最後更新日期：{last_updated[0] if last_updated else '無記錄'}")

        # 獲取現有股票的最後日期
        ticker_dates = pd.read_sql_query("SELECT Ticker, MAX(Date) as last_date FROM stocks GROUP BY Ticker", conn)
        logger.info(f"從資料庫中讀取 {len(ticker_dates)} 檔股票的最後日期")
        # 確保日期格式一致
        ticker_dates['last_date'] = pd.to_datetime(ticker_dates['last_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        logger.info(f"日期樣本（前5）：{ticker_dates['last_date'].head().tolist()}")
        existing_tickers = dict do_not_track=True
        existing_tickers = dict(zip(ticker_dates['Ticker'], pd.to_datetime(ticker_dates['last_date']).dt.date))

        # 檢查完整性：根據指定比例從末尾開始檢查
        num_to_check = max(1, int(len(tickers) * check_percentage))
        tickers_to_check = tickers[-num_to_check:]
        tickers_to_update = []
        default_start_date = end_date - timedelta(days=210)

        for ticker in tickers_to_check:
            last_date = existing_tickers.get(ticker)
            if not last_date:
                logger.info(f"股票 {ticker} 無數據，將加入更新清單")
                tickers_to_update.append(ticker)
            elif (end_date - last_date).days > 0:
                logger.info(f"股票 {ticker} 最後日期 {last_date}，需更新至 {end_date}")
                tickers_to_update.append(ticker)

        if not tickers_to_update:
            if last_updated and pd.to_datetime(last_updated[0], errors='coerce').date() >= end_date and len(existing_tickers) >= len(tickers):
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

        logger.info(f"需更新的股票數量：{len(tickers_to_update)}")

        # 分批下載並更新
        total_batches = (len(tickers_to_update) + batch_size - 1) // batch_size
        for i in range(0, len(tickers_to_update), batch_size):
            batch_tickers = tickers_to_update[i:i + batch_size]
            batch_start_dates = [
                existing_tickers.get(ticker, default_start_date) - timedelta(days=lookback_days)
                for ticker in batch_tickers
            ]
            start_date = min(batch_start_dates)
            logger.info(f"批次 {i // batch_size + 1}/{total_batches}：股票 {batch_tickers}，開始日期 {start_date}")
            
            data = download_with_retry(batch_tickers, start=start_date, end=end_date)
            if data is None:
                logger.warning(f"批次 {batch_tickers} 下載失敗，跳過")
                continue

            df = data.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]

            for ticker in batch_tickers:
                ticker_df = df[[col for col in df.columns if col.startswith(f"{ticker}_") or col == 'Date']].copy()
                ticker_df.columns = [col.replace(f"{ticker}_", "") for col in ticker_df.columns]
                ticker_df['Ticker'] = ticker
                # 確保日期格式為純日期
                ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                logger.info(f"股票 {ticker} 下載數據日期樣本（前5）：{ticker_df['Date'].head().tolist()}")

                # 驗證日期格式
                if not ticker_df['Date'].str.match(r'^\d{4}-\d{2}-\d{2}$').all():
                    logger.error(f"股票 {ticker} 的日期格式不正確：{ticker_df['Date'].head().tolist()}")
                    st.error(f"股票 {ticker} 的日期格式不正確，請檢查數據來源")
                    continue

                for _, row in ticker_df.iterrows():
                    cursor.execute('''INSERT OR IGNORE INTO stocks 
                        (Date, Ticker, Open, High, Low, Close, Adj_Close, Volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                        row['Date'], ticker,
                        float(row['Open']) if pd.notna(row['Open']) else None,
                        float(row['High']) if pd.notna(row['High']) else None,
                        float(row['Low']) if pd.notna(row['Low']) else None,
                        float(row['Close']) if pd.notna(row['Close']) else None,
                        float(row['Close']) if pd.notna(row['Close']) else None,
                        int(row['Volume']) if pd.notna(row['Volume']) else 0))

            conn.commit()
            st.write(f"批次 {i // batch_size + 1}/{total_batches} 完成")
            logger.info(f"批次 {i // batch_size + 1}/{total_batches} 已寫入資料庫")

        # 更新 metadata
        cursor.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        conn.close()
        logger.info(f"資料庫 metadata 更新，最後更新日期：{end_date}")

        # 推送至 GitHub 並顯示結果
        push_success = push_to_github(repo, f"Updated stocks.db with new data")
        if push_success:
            st.success("資料庫更新完成並成功推送至 GitHub")
        else:
            st.warning("資料庫更新完成，但推送至 GitHub 失敗")
        
        # 提供手動推送和下載選項
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
        logger.error(f"資料庫更新失敗，詳細錯誤：{str(e)}")
        if 'conn' in locals():
            conn.close()
        return False

def fetch_stock_data(tickers, db_path=DB_PATH, trading_days=70):
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
        
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = pd.to_datetime(data.index)
        
        pivoted_data = data.pivot(columns='Ticker')
        return pivoted_data, tickers
    except Exception as e:
        st.error(f"提取數據失敗：{str(e)}")
        return None, tickers
