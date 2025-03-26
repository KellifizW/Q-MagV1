# screening.py
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import time

def get_nasdaq_100():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        df = tables[4]
        if 'Ticker' not in df.columns:
            raise KeyError("未找到 'Ticker' 列")
        tickers = df['Ticker'].tolist()
        return [str(ticker) for ticker in tickers if pd.notna(ticker) and str(ticker).strip()]
    except Exception as e:
        st.error(f"無法從 Wikipedia 獲取 Nasdaq-100 清單: {str(e)}")
        backup = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'ADBE', 'PYPL', 'INTC',
                  'NFLX', 'CSCO', 'PEP', 'AVGO', 'COST', 'TMUS', 'AMD', 'TXN', 'QCOM', 'AMGN']
        return backup

def get_sp500():
    try:
        return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    except Exception as e:
        st.error(f"無法從 Wikipedia 獲取 S&P 500 清單: {e}")
        return get_nasdaq_100()

def get_nasdaq_all():
    try:
        nasdaq = pd.read_csv('https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download')
        return nasdaq['Symbol'].tolist()
    except Exception as e:
        st.error(f"無法從 Nasdaq 獲取股票清單: {e}")
        return get_nasdaq_100()

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, days=90):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            end_date = datetime(2025, 3, 26)  # 固定日期以符合要求
            start_date = end_date - timedelta(days=days)
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False, timeout=15)
            if stock.empty:
                st.warning(f"無法獲取 {ticker} 的數據：數據為空")
                return None
            if 'Close' not in stock.columns or 'Volume' not in stock.columns:
                st.warning(f"無法獲取 {ticker} 的數據：缺少 'Close' 或 'Volume' 欄位")
                return None
            return stock
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"獲取 {ticker} 的數據失敗（第 {attempt + 1} 次嘗試）：{str(e)}，正在重試...")
                time.sleep(2)  # 等待 2 秒後重試
            else:
                st.error(f"無法獲取 {ticker} 的數據（重試 {max_retries} 次後仍失敗）：{str(e)}")
                return None

def analyze_stock(args):
    ticker, prior_days, consol_days, min_rise_22, min_rise_67, max_range, min_adr = args
    # 確保數據範圍足夠（至少需要 67 天數據來計算 67 日漲幅）
    stock = fetch_stock_data(ticker, days=max(prior_days + consol_days + 30, 67))
    if stock is None or len(stock) < prior_days + consol_days + 30:
        return None
    
    close = stock['Close'].squeeze()
    volume = stock['Volume'].squeeze()
    dates = stock.index
    
    results = []
    for i in range(-30, 0):
        # 計算 22 日內漲幅
        if i + 22 >= 0 or len(close) < 22:
            rise_22 = 0
        else:
            price_22_days_ago = close.iloc[i - 22]
            current_price = close.iloc[i]
            rise_22 = ((current_price - price_22_days_ago) / price_22_days_ago) * 100 if price_22_days_ago != 0 else 0

        # 計算 67 日內漲幅
        if i + 67 >= 0 or len(close) < 67:
            rise_67 = 0
        else:
            price_67_days_ago = close.iloc[i - 67]
            rise_67 = ((current_price - price_67_days_ago) / price_67_days_ago) * 100 if price_67_days_ago != 0 else 0

        # 計算前段漲幅（僅用於檢查 prior_days 期間的趨勢）
        if i < -prior_days:
            prior_rise = (close.iloc[i] / close.iloc[i - prior_days] - 1) * 100
            recent_high = close.iloc[i - consol_days:i].max()
            recent_low = close.iloc[i - consol_days:i].min()
            consolidation_range = (recent_high / recent_low - 1) * 100 if recent_low != 0 else float('inf')
            vol_decline = volume.iloc[i - consol_days:i].mean() < volume.iloc[i - prior_days:i - consol_days].mean()
            
            high = stock['High'].squeeze().iloc[i - prior_days:i]
            low = stock['Low'].squeeze().iloc[i - prior_days:i]
            prev_close = stock['Close'].shift(1).squeeze().iloc[i - prior_days:i]
            
            if high.empty or low.empty or prev_close.empty:
                adr = 0
            else:
                is_all_nan = prev_close.isna().all()
                if is_all_nan:
                    adr = 0
                else:
                    daily_range = (high - low) / prev_close
                    adr_mean = daily_range.mean()
                    adr = adr_mean * 100 if not pd.isna(adr_mean) else 0
            
            breakout = (i == -1) and (close.iloc[-1] > recent_high) and (close.iloc[-2] <= recent_high)
            breakout_volume = (i == -1) and (volume.iloc[-1] > volume.iloc[-10:].mean() * 1.5)
            
            # 使用 22 日和 67 日漲幅進行篩選
            if rise_22 > min_rise_22 and rise_67 > min_rise_67 and consolidation_range < max_range and adr > min_adr:
                results.append({
                    'Ticker': ticker,
                    'Date': dates[i].strftime('%Y-%m-%d'),
                    'Price': close.iloc[i],
                    'Prior_Rise_22_%': rise_22,
                    'Prior_Rise_67_%': rise_67,
                    'Consolidation_Range_%': consolidation_range,
                    'ADR_%': adr,
                    'Breakout': breakout,
                    'Breakout_Volume': breakout_volume
                })
    return results

def screen_stocks(tickers, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, max_range=5, min_adr=5, progress_bar=None):
    total_tickers = len(tickers)
    results = []
    for i, ticker in enumerate(tickers):
        try:
            # 調用 analyze_stock 函數
            result = analyze_stock((ticker, prior_days, consol_days, min_rise_22, min_rise_67, max_range, min_adr))
            if result:
                results.extend(result)
        except Exception as e:
            st.warning(f"處理 {ticker} 時發生錯誤：{str(e)}")
            continue  # 跳過失敗的股票，繼續處理下一隻

        # 更新進度條
        if progress_bar:
            progress_bar.progress(min((i + 1) / total_tickers, 1.0))
        time.sleep(0.05)  # 添加 50 毫秒延遲，避免過多請求

    return pd.DataFrame(results)
