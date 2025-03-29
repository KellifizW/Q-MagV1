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

# 常量定義
REPO_DIR = "."
DB_PATH = "stocks.db"
TICKERS_CSV = "Tickers.csv"
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"
US_EASTERN = timezone('US/Eastern')
BATCH_SIZE = 5  # 減少批次大小以避免 Alpha Vantage 速率限制

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
    """安全轉換為浮點數，記錄詳細錯誤"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (ValueError, TypeError) as e:
        log_to_page(f"無法將 {column_name} 轉換為浮點數，股票：{ticker}，日期：{date}，值：{repr(value)}，錯誤：{str(e)}", "ERROR")
        raise ValueError(f"Invalid {column_name} value for {ticker} on {date}: {repr(value)}")

def safe_int(value, column_name, ticker, date):
    """安全轉換為整數，記錄詳細錯誤"""
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (ValueError, TypeError) as e:
        log_to_page(f"無法將 {column_name} 轉換為整數，股票：{ticker}，日期：{date}，值：{repr(value)}，錯誤：{str(e)}", "ERROR")
        raise ValueError(f"Invalid {column_name} value for {ticker} on {date}: {repr(value)}")

def download_with_retry(tickers, start, end, retries=2, delay=5, api_key=None):
    """下載股票數據，優先 yfinance，失敗則切換 Alpha Vantage"""
    # 首先嘗試 yfinance，並強制失敗
    for attempt in range(retries):
        try:
            raise Exception("強制 yfinance 失敗")  # 強制 yfinance 失敗以測試 Alpha Vantage
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
            if data.empty:
                log_to_page(f"批次數據為空，股票：{tickers}", "WARNING")
                return None
            log_to_page(f"成功下載 {len(tickers)} 檔股票數據 (yfinance)", "INFO")
            return data
        except Exception as e:
            log_to_page(f"yfinance 下載失敗，股票：{tickers}，錯誤：{str(e)}，重試 {attempt + 1}/{retries}", "WARNING")
            time.sleep(delay)

    # 如果 yfinance 失敗，切換到 Alpha Vantage
    if not api_key:
        log_to_page(f"未提供 Alpha Vantage API Key，下載 {tickers} 失敗", "ERROR")
        return None

    log_to_page(f"yfinance 重試失敗，切換到 Alpha Vantage 嘗試下載 {tickers}", "INFO")
    all_data = []
    cutoff_date = start.strftime('%Y-%m-%d')
    
    for ticker in tickers:
        for attempt in range(retries):
            try:
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=full"
                log_to_page(f"請求 Alpha Vantage，股票：{ticker}，URL：{url}", "DEBUG")
                response = requests.get(url).json()
                log_to_page(f"股票 {ticker} 的 API 回應: {response}", "DEBUG")
                
                if "Time Series (Daily)" not in response:
                    if "Note" in response:
                        log_to_page(f"Alpha Vantage API 限制，股票：{ticker}，訊息：{response['Note']}", "WARNING")
                    elif "Error Message" in response:
                        log_to_page(f"Alpha Vantage API 錯誤，股票：{ticker}，錯誤：{response['Error Message']}", "ERROR")
                    else:
                        log_to_page(f"Alpha Vantage 無數據返回，股票：{ticker}，未知回應：{response}", "WARNING")
                    break
                
                time_series = response["Time Series (Daily)"]
                df = pd.DataFrame.from_dict(time_series, orient='index').reset_index()
                
                df = df.rename(columns={
                    'index': 'Date',
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
                
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    log_to_page(f"Alpha Vantage 返回的數據缺少欄位：{missing_cols}，股票：{ticker}", "ERROR")
                    break
                
                df['Ticker'] = ticker
                df['Adj Close'] = df['Close']
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                
                # 過濾日期範圍
                df = df[df['Date'] >= cutoff_date]
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isna().any():
                        log_to_page(f"股票 {ticker} 的 {col} 欄位包含無效值，已轉換為 NaN", "WARNING")
                
                all_data.append(df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Ticker']])
                log_to_page(f"成功下載股票 {ticker} 的數據 (Alpha Vantage)", "INFO")
                time.sleep(12)  # 每分鐘 5 次請求，每 12 秒一次
                break
            except Exception as e:
                log_to_page(f"Alpha Vantage 下載股票 {ticker} 失敗，錯誤：{str(e)}，重試 {attempt + 1}/{retries}", "WARNING")
                time.sleep(delay)
        else:
            log_to_page(f"Alpha Vantage 下載股票 {ticker} 失敗，已達最大重試次數", "ERROR")
    
    if not all_data:
        log_to_page(f"Alpha Vantage 下載 {tickers} 失敗，無有效數據", "ERROR")
        return None
    
    combined_df = pd.concat(all_data)
    combined_df = combined_df.set_index(['Date', 'Ticker']).unstack('Ticker')
    log_to_page(f"成功下載 {len(tickers)} 檔股票數據 (Alpha Vantage)", "INFO")
    return combined_df

def init_repo():
    """初始化 Git 倉庫"""
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
        log_to_page(f"從 {tickers_file} 讀取 {len(tickers)} 檔股票", "INFO")

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
        
        # 獲取現有股票的最後日期
        ticker_dates = pd.read_sql_query("SELECT Ticker, MAX(Date) as last_date FROM stocks GROUP BY Ticker", conn)
        existing_tickers = dict(zip(ticker_dates['Ticker'], pd.to_datetime(ticker_dates['last_date']).dt.date))

        # 檢查完整性
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

        # 分批下載並更新
        try:
            api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
            log_to_page(f"獲取的 Alpha Vantage API Key: {api_key}", "INFO")
            if not api_key:
                st.error("Alpha Vantage API Key 是空的，請檢查 st.secrets 配置")
                return False
        except KeyError as e:
            st.error(f"未找到 ALPHA_VANTAGE_API_KEY 於 st.secrets 中：{str(e)}")
            return False

        total_batches = (len(tickers_to_update) + batch_size - 1) // batch_size
        for i in range(0, len(tickers_to_update), batch_size):
            batch_tickers = tickers_to_update[i:i + batch_size]
            batch_start_dates = [
                existing_tickers.get(ticker, default_start_date) - timedelta(days=lookback_days)
                for ticker in batch_tickers
            ]
            start_date = min(batch_start_dates)
            data = download_with_retry(batch_tickers, start=start_date, end=end_date, api_key=api_key)
            if data is None:
                log_to_page(f"批次 {i // batch_size + 1}/{total_batches} 下載失敗，跳過", "WARNING")
                continue

            # 重置索引並處理 MultiIndex
            df = data.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df.columns]

            for ticker in batch_tickers:
                try:
                    ticker_df = df[[col for col in df.columns if col.startswith(f"{ticker}_") or col == 'Date']].copy()
                    if ticker_df.empty:
                        log_to_page(f"股票 {ticker} 的數據為空，跳過", "WARNING")
                        continue
                    ticker_df.columns = [col.replace(f"{ticker}_", "") for col in ticker_df.columns]
                    ticker_df['Ticker'] = ticker
                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.strftime('%Y-%m-%d')
                    
                    # 檢查必要欄位
                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    missing_cols = [col for col in required_cols if col not in ticker_df.columns]
                    if missing_cols:
                        log_to_page(f"股票 {ticker} 缺少必要欄位：{missing_cols}", "ERROR")
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
                            log_to_page(f"數據轉換失敗：{str(e)}", "ERROR")
                            continue
                        except sqlite3.Error as e:
                            log_to_page(f"SQLite 插入失敗，股票：{ticker}，日期：{row['Date']}，錯誤：{str(e)}", "ERROR")
                            continue
                except Exception as e:
                    log_to_page(f"處理股票 {ticker} 的數據時失敗：{str(e)}", "ERROR")
                    continue

            conn.commit()
            st.write(f"批次 {i // batch_size + 1}/{total_batches} 完成")

        # 更新 metadata
        cursor.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)", (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        conn.close()

        # 推送至 GitHub
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
        if 'conn' in locals():
            conn.close()
        return False

def fetch_stock_data(tickers, db_path=DB_PATH, trading_days=70):
    """從資料庫提取股票數據"""
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

# 主程式
if __name__ == "__main__":
    st.title("股票資料庫更新工具")
    repo = init_repo()
    init_database()
    if st.button("更新資料庫"):
        update_database(repo=repo)
