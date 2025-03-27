import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import os
import git
import shutil
import streamlit as st

# 配置
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"  # 替換為你的存儲庫 URL
TOKEN = "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"  # 替換為你的 GitHub Token
REPO_DIR = "Q-MagV1"
DB_PATH = os.path.join(REPO_DIR, "stocks.db")
nasdaq = mcal.get_calendar('NASDAQ')

def clone_repo():
    """從 GitHub 複製存儲庫到本地"""
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    repo = git.Repo.clone_from(f"https://{TOKEN}@github.com/KellifizW/Q-MagV1.git", REPO_DIR)
    return repo

def push_to_github(repo, message="Update stocks.db"):
    """推送更新到 GitHub"""
    repo.git.add("stocks.db")
    repo.git.commit(m=message)
    repo.git.push()

def initialize_database(tickers, db_path=DB_PATH):
    """初始化資料庫，只下載 tickers.csv 中的股票"""
    conn = sqlite3.connect(db_path)
    end_date = datetime.now().date()
    start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[0].date()
    
    for i, ticker in enumerate(tickers):
        st.write(f"初始化：下載 {ticker} ({i+1}/{len(tickers)})...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                data['Ticker'] = ticker
                data.to_sql('stocks', conn, if_exists='append', index=True, index_label='Date')
        except Exception as e:
            st.write(f"無法下載 {ticker}: {e}")
    
    pd.DataFrame({'last_updated': [end_date.strftime('%Y-%m-%d')]}).to_sql('metadata', conn, if_exists='replace', index=False)
    conn.close()

def update_database(tickers, db_path=DB_PATH):
    """每日更新，只更新 tickers.csv 中的股票"""
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
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=current_date, progress=False)
            if not data.empty:
                data['Ticker'] = ticker
                data.to_sql('stocks', conn, if_exists='append', index=True, index_label='Date')
        except Exception as e:
            st.write(f"更新 {ticker} 失敗: {e}")
    
    pd.DataFrame({'last_updated': [current_date.strftime('%Y-%m-%d')]}).to_sql('metadata', conn, if_exists='replace', index=False)
    conn.close()
    return True

def extend_sp500(tickers_sp500, db_path=DB_PATH):
    """動態補充 S&P 500 缺少的股票"""
    conn = sqlite3.connect(db_path)
    existing_tickers = pd.read_sql_query("SELECT DISTINCT Ticker FROM stocks", conn)['Ticker'].tolist()
    missing_tickers = [ticker for ticker in tickers_sp500 if ticker not in existing_tickers]
    
    if missing_tickers:
        st.write(f"檢測到 {len(missing_tickers)} 隻 S&P 500 股票缺失，正在補充...")
        end_date = datetime.now().date()
        start_date = nasdaq.schedule(start_date=end_date - timedelta(days=180), end_date=end_date).index[0].date()
        
        for i, ticker in enumerate(missing_tickers):
            st.write(f"補充 S&P 500 股票：{ticker} ({i+1}/{len(missing_tickers)})...")
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    data['Ticker'] = ticker
                    data.to_sql('stocks', conn, if_exists='append', index=True, index_label='Date')
            except Exception as e:
                st.write(f"無法下載 {ticker}: {e}")
    
    conn.close()
    return len(missing_tickers) > 0

def fetch_stock_data(tickers, stock_pool=None, db_path=DB_PATH, trading_days=70):
    """從 SQLite 查詢數據，若是 S&P 500 則先擴展"""
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
