# screening.py
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import pandas_market_calendars as mcal
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

def get_trading_days(end_date, num_trading_days):
    nyse = mcal.get_calendar('NYSE')
    temp_start = end_date - timedelta(days=180)
    schedule = nyse.schedule(start_date=temp_start, end_date=end_date)
    trading_days = schedule.index
    if len(trading_days) < num_trading_days:
        raise ValueError(f"無法獲取 {num_trading_days} 個交易日，僅有 {len(trading_days)} 個交易日可用")
    start_date = trading_days[-num_trading_days]
    return start_date.date(), end_date

@st.cache_data(ttl=3600)
def fetch_stock_data_batch(tickers, trading_days=70):
    """批量下載股票數據並返回完整的 DataFrame"""
    end_date = datetime(2025, 3, 26).date()
    try:
        start_date, end_date = get_trading_days(end_date, trading_days)
    except Exception as e:
        st.error(f"無法計算交易日範圍：{str(e)}")
        return None
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", progress=False, timeout=15)
        if data.empty:
            st.error("批量下載數據為空")
            return None
        st.write(f"批量下載完成，數據長度：{len(data)} 筆，股票數量：{len(tickers)}")
        return data
    except Exception as e:
        st.error(f"批量下載數據失敗：{str(e)}")
        return None

def analyze_stock_batch(data, tickers, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, max_range=5, min_adr=5):
    """批量分析股票數據，返回符合條件的結果"""
    results = []
    failed_stocks = {}
    required_days = prior_days + consol_days + 30
    
    for ticker in tickers:
        try:
            # 提取單一股票數據
            if isinstance(data.columns, pd.MultiIndex):
                stock = data[ticker]
            else:
                stock = data
            
            if stock['Close'].isna().all() or len(stock) < required_days:
                failed_stocks[ticker] = f"數據不足或無效，長度 {len(stock)}，需 {required_days}"
                continue
            
            close = stock['Close']
            volume = stock['Volume']
            high = stock['High']
            low = stock['Low']
            prev_close = close.shift(1)
            dates = stock.index
            
            # 向量計算指標
            rise_22 = (close / close.shift(22) - 1) * 100
            rise_67 = (close / close.shift(67) - 1) * 100
            recent_high = close.rolling(consol_days).max()
            recent_low = close.rolling(consol_days).min()
            consolidation_range = (recent_high / recent_low - 1) * 100
            vol_decline = volume.rolling(consol_days).mean() < volume.shift(consol_days).rolling(prior_days).mean()
            daily_range = (high - low) / prev_close
            adr = daily_range.rolling(prior_days).mean() * 100
            breakout = (close > recent_high.shift(1)) & (close.shift(1) <= recent_high.shift(1))
            breakout_volume = volume > volume.rolling(10).mean() * 1.5
            
            # 篩選條件
            mask = (rise_22 >= min_rise_22) & (rise_67 >= min_rise_67) & \
                   (consolidation_range <= max_range) & (adr >= min_adr)
            
            if mask.any():
                matched = pd.DataFrame({
                    'Ticker': ticker,
                    'Date': dates[mask],
                    'Price': close[mask],
                    'Prior_Rise_22_%': rise_22[mask],
                    'Prior_Rise_67_%': rise_67[mask],
                    'Consolidation_Range_%': consolidation_range[mask],
                    'ADR_%': adr[mask],
                    'Breakout': breakout[mask],
                    'Breakout_Volume': breakout_volume[mask]
                })
                results.append(matched)
                st.write(f"股票 {ticker} 符合條件（最新）：22 日漲幅 = {rise_22.iloc[-1]:.2f}%, "
                         f"67 日漲幅 = {rise_67.iloc[-1]:.2f}%, 盤整範圍 = {consolidation_range.iloc[-1]:.2f}%, "
                         f"ADR = {adr.iloc[-1]:.2f}%")
            else:
                st.write(f"股票 {ticker} 不符合條件：22 日漲幅 = {rise_22.iloc[-1]:.2f}% (需 >= {min_rise_22}), "
                         f"67 日漲幅 = {rise_67.iloc[-1]:.2f}% (需 >= {min_rise_67}), "
                         f"盤整範圍 = {consolidation_range.iloc[-1]:.2f}% (需 <= {max_range}), "
                         f"ADR = {adr.iloc[-1]:.2f}% (需 >= {min_adr})")
                
        except Exception as e:
            failed_stocks[ticker] = f"分析失敗：{str(e)}"
    
    if failed_stocks:
        st.warning(f"無法分析的股票：{failed_stocks}")
    
    return pd.concat(results) if results else pd.DataFrame()

def screen_stocks(tickers, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, max_range=5, min_adr=5, progress_bar=None):
    """主篩選函數，返回篩選結果並保留批量數據供可視化使用"""
    # 批量下載數據
    data = fetch_stock_data_batch(tickers, trading_days=70)
    if data is None:
        return pd.DataFrame()
    
    # 批量分析
    total_tickers = len(tickers)
    results = analyze_stock_batch(data, tickers, prior_days, consol_days, min_rise_22, min_rise_67, max_range, min_adr)
    
    if progress_bar:
        progress_bar.progress(1.0)
    
    # 將批量數據存入 session_state 供可視化使用
    st.session_state['stock_data'] = data
    
    return results

# 保留原始 fetch_stock_data 以兼容性測試（可選）
def fetch_stock_data(ticker, trading_days=70):
    max_retries = 3
    end_date = datetime(2025, 3, 26).date()
    try:
        start_date, end_date = get_trading_days(end_date, trading_days)
    except Exception as e:
        st.error(f"無法計算交易日範圍：{str(e)}")
        return None, str(e)

    for attempt in range(max_retries):
        try:
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False, timeout=15)
            if stock.empty:
                return None, "數據為空"
            if 'Close' not in stock.columns or 'Volume' not in stock.columns:
                return None, "缺少 'Close' 或 'Volume' 欄位"
            return stock, None
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"獲取 {ticker} 的數據失敗（第 {attempt + 1} 次嘗試）：{str(e)}，正在重試...")
                time.sleep(2)
            else:
                return None, f"重試 {max_retries} 次後仍失敗：{str(e)}"
