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
import requests
import hashlib
import magic  # 需要安裝 python-magic-bin (Windows)

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

def check_and_fetch_lfs_file(file_path, repo_url, token):
    """檢查並從 Git LFS 下載檔案"""
    try:
        # 檢查檔案是否存在且不是 LFS 指標檔案
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 檢查是否為 LFS 指標檔案
                if content.startswith('version https://git-lfs.github.com'):
                    logger.info(f"檢測到 {file_path} 是 LFS 指標檔案，開始下載實際內容")
                    subprocess.run(['git', 'lfs', 'pull', '--include', file_path], 
                                 check=True, capture_output=True, text=True)
                    logger.info(f"已從 LFS 下載 {file_path}")
                return True
        else:
            # 如果檔案不存在，嘗試從遠端獲取
            logger.info(f"{file_path} 不存在，嘗試從 GitHub LFS 下載")
            raw_url = f"https://raw.githubusercontent.com/KellifizW/Q-MagV1/main/{os.path.basename(file_path)}"
            response = requests.get(raw_url, headers={"Authorization": f"token {token}"})
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                # 配置 LFS 追蹤
                subprocess.run(['git', 'lfs', 'track', file_path], 
                             check=True, capture_output=True, text=True)
                logger.info(f"已下載並配置 {file_path} 為 LFS 檔案")
            else:
                logger.warning(f"無法從 {raw_url} 下載檔案，狀態碼：{response.status_code}")
                return False
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"從 LFS 下載 {file_path} 失敗：{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"處理 LFS 檔案 {file_path} 時出錯：{str(e)}")
        return False

def diagnose_db_file(db_path):
    """診斷資料庫檔案，包含 LFS 檢查"""
    diagnostics = [f"檢查檔案：{os.path.abspath(db_path)}"]
    
    # 檢查並獲取 LFS 檔案
    token = st.secrets.get("TOKEN", "")
    if not check_and_fetch_lfs_file(db_path, REPO_URL, token):
        diagnostics.append("錯誤：無法從 LFS 獲取檔案")
        return diagnostics
        
    if not os.path.exists(db_path):
        diagnostics.append("錯誤：檔案不存在")
        return diagnostics
    
    # 檢查檔案大小
    file_size = os.path.getsize(db_path)
    diagnostics.append(f"檔案大小：{file_size} 位元組")
    if file_size == 0:
        diagnostics.append("警告：檔案為空")

    # 檢查檔案類型
    try:
        file_type = magic.from_file(db_path, mime=True)
        diagnostics.append(f"檔案類型：{file_type}")
        if file_type != "application/x-sqlite3":
            diagnostics.append("警告：檔案不是 SQLite 資料庫格式")
    except ImportError:
        diagnostics.append("警告：未安裝 python-magic，無法檢查檔案類型")
    except Exception as e:
        diagnostics.append(f"檢查檔案類型失敗：{str(e)}")

    # 計算檔案哈希值（檢查完整性）
    with open(db_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    diagnostics.append(f"檔案 MD5 哈希值：{file_hash}")

    return diagnostics

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
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error(f"yfinance 下載 {tickers} 最終失敗，經過 {retries} 次嘗試")
    return None

def init_repo():
    """初始化 Git 倉庫"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
            subprocess.run(['git', 'lfs', 'install'], check=True, capture_output=True, text=True)
            logger.info("初始化 Git 倉庫並啟用 LFS")

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
        
        # 先拉取遠端變更
        pull_result = subprocess.run(['git', 'pull', remote_url, branch], capture_output=True, text=True)
        if pull_result.returncode != 0:
            st.error(f"拉取遠端變更失敗：{pull_result.stderr}")
            return False

        # 推送至 GitHub，包括 LFS 檔案
        subprocess.run(['git', 'push', remote_url, branch], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'lfs', 'push', '--all', remote_url, branch], check=True, capture_output=True, text=True)
        st.write("成功推送至 GitHub（包含 LFS 檔案）")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"推送至 GitHub 失敗：{str(e)}\n錯誤輸出：{e.stderr}")
        return False
    except Exception as e:
        st.error(f"推送至 GitHub 失敗：{str(e)}")
        return False

def init_database():
    """從 GitHub 下載初始資料庫或創建新資料庫，支援 LFS"""
    if 'db_initialized' not in st.session_state:
        try:
            token = st.secrets["TOKEN"]
            # 先檢查本地檔案並處理 LFS
            if os.path.exists(DB_PATH):
                check_and_fetch_lfs_file(DB_PATH, REPO_URL, token)
            
            # 如果檔案仍不存在或無效，從遠端下載
            if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 100:
                url = "https://raw.githubusercontent.com/KellifizW/Q-MagV1/main/stocks.db"
                response = requests.get(url, headers={"Authorization": f"token {token}"})
                if response.status_code == 200:
                    with open(DB_PATH, "wb") as f:
                        f.write(response.content)
                    subprocess.run(['git', 'lfs', 'track', DB_PATH], 
                                 check=True, capture_output=True, text=True)
                    st.write("已從 GitHub 下載 stocks.db 並配置為 LFS 檔案")
                else:
                    st.write("未找到遠端 stocks.db，將創建新資料庫")
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
                        Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
                        PRIMARY KEY (Date, Ticker))''')
                    conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
                    conn.commit()
                    conn.close()
                    subprocess.run(['git', 'lfs', 'track', DB_PATH], 
                                 check=True, capture_output=True, text=True)
            
            # 驗證資料庫
            diagnostics = diagnose_db_file(DB_PATH)
            st.write("檔案診斷資訊：")
            for line in diagnostics:
                st.write(line)
            conn = sqlite3.connect(DB_PATH)
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            conn.close()
            st.session_state['db_initialized'] = True
        except sqlite3.DatabaseError as e:
            st.error(f"下載的 stocks.db 無效：{str(e)}")
            st.write("將創建新資料庫")
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            conn = sqlite3.connect(DB_PATH)
            conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
                Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
                PRIMARY KEY (Date, Ticker))''')
            conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
            conn.commit()
            conn.close()
            subprocess.run(['git', 'lfs', 'track', DB_PATH], 
                         check=True, capture_output=True, text=True)
            st.session_state['db_initialized'] = True
        except Exception as e:
            st.error(f"初始化資料庫失敗：{str(e)}")
            st.session_state['db_initialized'] = False

def update_database(tickers_file=TICKERS_CSV, db_path=DB_PATH, batch_size=BATCH_SIZE, repo=None, check_percentage=0.1, lookback_days=30):
    """增量更新資料庫，包含 LFS 支援"""
    if repo is None:
        st.error("未提供 Git 倉庫物件")
        return False

    try:
        # 確保檔案從 LFS 正確獲取
        check_and_fetch_lfs_file(db_path, REPO_URL, st.secrets.get("TOKEN", ""))
        
        diagnostics = diagnose_db_file(db_path)
        st.write("資料庫檔案診斷資訊：")
        for line in diagnostics:
            st.write(line)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        conn.commit()

        tickers_df = pd.read_csv(tickers_file)
        tickers = tickers_df['Ticker'].tolist()
        logger.info(f"從 {tickers_file} 讀取 {len(tickers)} 檔股票")

        cursor.execute("SELECT last_updated FROM metadata")
        last_updated = cursor.fetchone()
        current_date = datetime.now(US_EASTERN).date()
        end_date = current_date - timedelta(days=1)
        logger.info(f"當前日期：{current_date}，結束日期：{end_date}")

        ticker_dates = pd.read_sql_query("SELECT Ticker, MAX(Date) as last_date FROM stocks GROUP BY Ticker", conn)
        ticker_dates['last_date'] = pd.to_datetime(ticker_dates['last_date'], errors='coerce').dt.strftime('%Y-%m-%d')
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

        total_batches = (len(tickers_to_update) + batch_size - 1) // batch_size
        progress_bar = st.progress(0)
        for i in range(0, len(tickers_to_update), batch_size):
            batch_tickers = tickers_to_update[i:i + batch_size]
            batch_start_dates = [
                existing_tickers.get(ticker, default_start_date) - timedelta(days=lookback_days)
                for ticker in batch_tickers
            ]
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
                ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')

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
            progress = (i + batch_size) / len(tickers_to_update)
            progress_bar.progress(min(progress, 1.0))

        cursor.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        conn.close()

        push_success = push_to_github(repo, f"Updated stocks.db with new data")
        if push_success:
            st.success("資料庫更新完成並成功推送至 GitHub (含 LFS)")
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
        return True

    except sqlite3.DatabaseError as e:
        diagnostics = diagnose_db_file(db_path)
        error_msg = f"資料庫錯誤：{str(e)}\n診斷資訊：\n" + "\n".join(diagnostics)
        st.error(error_msg)
        logger.error(error_msg)
        if 'conn' in locals():
            conn.close()
        return False
    except Exception as e:
        st.error(f"資料庫更新失敗：{str(e)}")
        logger.error(f"資料庫更新失敗，詳細錯誤：{str(e)}")
        if 'conn' in locals():
            conn.close()
        return False

def fetch_stock_data(tickers, db_path=DB_PATH, trading_days=70):
    try:
        check_and_fetch_lfs_file(db_path, REPO_URL, st.secrets.get("TOKEN", ""))
        
        if not os.path.exists(db_path):
            st.error(f"資料庫檔案 {db_path} 不存在，請先初始化或更新資料庫")
            return None, tickers
        diagnostics = diagnose_db_file(db_path)
        st.write("提取數據前的檔案診斷資訊：")
        for line in diagnostics:
            st.write(line)
        
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
    except sqlite3.DatabaseError as e:
        diagnostics = diagnose_db_file(db_path)
        error_msg = f"提取數據失敗：{str(e)}\n診斷資訊：\n" + "\n".join(diagnostics)
        st.error(error_msg)
        return None, tickers
    except Exception as e:
        st.error(f"提取數據失敗：{str(e)}")
        return None, tickers

if __name__ == "__main__":
    repo = init_repo()
    init_database()
    update_database(repo=repo)