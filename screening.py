import pandas as pd
import streamlit as st
from database import fetch_stock_data

def get_nasdaq_100(csv_tickers):
    """從 tickers.csv 過濾 NASDAQ 100 股票"""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        df = tables[4]
        if 'Ticker' not in df.columns:
            raise KeyError("未找到 'Ticker' 列")
        nasdaq_100 = df['Ticker'].tolist()
        return [ticker for ticker in nasdaq_100 if ticker in csv_tickers]
    except Exception as e:
        st.error(f"無法從 Wikipedia 獲取 Nasdaq-100 清單: {str(e)}")
        return [ticker for ticker in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA'] if ticker in csv_tickers]

def get_sp500():
    """獲取完整的 S&P 500 清單"""
    try:
        return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    except Exception as e:
        st.error(f"無法從 Wikipedia 獲取 S&P 500 清單: {e}")
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA']

def get_nasdaq_all(csv_tickers):
    """直接使用 tickers.csv 的股票，不擴展"""
    return csv_tickers

def analyze_stock_batch(data, tickers, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, max_range=5, min_adr=5):
    """批量分析股票數據"""
    results = []
    failed_stocks = {}
    required_days = prior_days + consol_days + 30
    
    for ticker in tickers:
        try:
            stock = data[ticker]
            if stock['Close'].isna().all() or len(stock) < required_days:
                failed_stocks[ticker] = f"數據不足或無效，長度 {len(stock)}，需 {required_days}"
                continue
            
            close = stock['Close']
            volume = stock['Volume']
            high = stock['High']
            low = stock['Low']
            prev_close = close.shift(1)
            dates = stock.index
            
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

def screen_stocks(tickers, stock_pool, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, max_range=5, min_adr=5, progress_bar=None):
    """主篩選函數"""
    data = fetch_stock_data(tickers, stock_pool)
    if data is None:
        st.error("無法從資料庫獲取數據")
        return pd.DataFrame()
    
    results = analyze_stock_batch(data, tickers, prior_days, consol_days, min_rise_22, min_rise_67, max_range, min_adr)
    
    if progress_bar:
        progress_bar.progress(1.0)
    
    st.session_state['stock_data'] = data
    return results
