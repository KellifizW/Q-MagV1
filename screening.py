import pandas as pd
import streamlit as st
from database import fetch_stock_data, fetch_yfinance_data

def get_nasdaq_100(csv_tickers):
    """從 csv_tickers 過濾 NASDAQ 100 股票"""
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
    """直接使用 csv_tickers，不擴展"""
    return csv_tickers

def analyze_stock_batch(data, tickers, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, min_rise_126=80, max_range=5, min_adr=5, show_all=False):
    results = []
    failed_stocks = {}
    matched_count = 0
    unmatched_count = 0
    required_days = max(prior_days + consol_days + 30, 126 + consol_days)
    
    for ticker in tickers:
        try:
            # 檢查數據結構並提取股票數據
            if isinstance(data.columns, pd.MultiIndex):
                stock = data.xs(ticker, level='Ticker', axis=1)
            else:
                stock = data[[col for col in data.columns if ticker in col]].copy()
                stock.columns = [col[0] for col in stock.columns]  # 移除多層索引中的 Ticker 層
            
            if not isinstance(stock, pd.DataFrame):
                failed_stocks[ticker] = f"stock 不是 DataFrame，類型為 {type(stock)}"
                continue
            
            close = pd.Series(stock['Close']).dropna()
            volume = pd.Series(stock['Volume']).dropna()
            high = pd.Series(stock['High']).dropna()
            low = pd.Series(stock['Low']).dropna()
            prev_close = close.shift(1)
            dates = stock.index
            
            if len(close) < required_days:
                failed_stocks[ticker] = f"數據長度不足，長度 {len(close)}，需 {required_days}"
                continue
            
            close_shift_22 = close.shift(22)
            close_shift_67 = close.shift(67)
            close_shift_126 = close.shift(126)
            rise_22 = (close / close_shift_22 - 1) * 100
            rise_67 = (close / close_shift_67 - 1) * 100
            rise_126 = (close / close_shift_126 - 1) * 100
            
            st.write(f"{ticker} - 22日數據點: {close_shift_22.notna().sum()}, 67日數據點: {close_shift_67.notna().sum()}, 126日數據點: {close_shift_126.notna().sum()}")
            
            if rise_126.isna().all():
                failed_stocks[ticker] = f"無法計算 126 日漲幅，數據長度 {len(close)}，有效 126 日前數據點 {close_shift_126.notna().sum()}"
                continue
            if rise_67.isna().all():
                failed_stocks[ticker] = f"無法計算 67 日漲幅，數據長度 {len(close)}，有效 67 日前數據點 {close_shift_67.notna().sum()}"
                continue
            if rise_22.isna().all():
                failed_stocks[ticker] = f"無法計算 22 日漲幅，數據長度 {len(close)}，有效 22 日前數據點 {close_shift_22.notna().sum()}"
                continue
            
            recent_high = close.rolling(consol_days).max()
            recent_low = close.rolling(consol_days).min()
            consolidation_range = (recent_high / recent_low - 1) * 100
            vol_decline = volume.rolling(consol_days).mean() < volume.shift(consol_days).rolling(prior_days).mean()
            daily_range = (high - low) / prev_close
            adr = daily_range.rolling(prior_days).mean() * 100
            breakout = (close > recent_high.shift(1)) & (close.shift(1) <= recent_high.shift(1))
            breakout_volume = volume > volume.rolling(10).mean() * 1.5
            
            mask = (rise_22 >= min_rise_22) & (rise_67 >= min_rise_67) & (rise_126 >= min_rise_126) & \
                   (consolidation_range <= max_range) & (adr >= min_adr)
            
            # 如果 show_all=True，記錄最新一天數據（用於即時查詢）
            if show_all:
                latest_data = pd.DataFrame({
                    'Ticker': ticker,
                    'Date': dates[-1:],
                    'Price': close[-1:],
                    'Prior_Rise_22_%': rise_22[-1:],
                    'Prior_Rise_67_%': rise_67[-1:],
                    'Prior_Rise_126_%': rise_126[-1:],
                    'Consolidation_Range_%': consolidation_range[-1:],
                    'ADR_%': adr[-1:],
                    'Breakout': breakout[-1:],
                    'Breakout_Volume': breakout_volume[-1:]
                })
                results.append(latest_data)
            
            # 如果股票符合條件，記錄所有符合的數據點
            if mask.any():
                matched_count += 1
                matched = pd.DataFrame({
                    'Ticker': ticker,
                    'Date': dates[mask],
                    'Price': close[mask],
                    'Prior_Rise_22_%': rise_22[mask],
                    'Prior_Rise_67_%': rise_67[mask],
                    'Prior_Rise_126_%': rise_126[mask],
                    'Consolidation_Range_%': consolidation_range[mask],
                    'ADR_%': adr[mask],
                    'Breakout': breakout[mask],
                    'Breakout_Volume': breakout_volume[mask]
                })
                # 如果不是 show_all 模式，只記錄符合條件的數據
                if not show_all:
                    results.append(matched)
                st.write(f"股票 {ticker} 符合條件（最新）：22 日漲幅 = {rise_22.iloc[-1]:.2f}%, "
                         f"67 日漲幅 = {rise_67.iloc[-1]:.2f}%, 126 日漲幅 = {rise_126.iloc[-1]:.2f}%, "
                         f"盤整範圍 = {consolidation_range.iloc[-1]:.2f}%, ADR = {adr.iloc[-1]:.2f}%")
            else:
                unmatched_count += 1
                
        except Exception as e:
            failed_stocks[ticker] = f"分析失敗：{str(e)}"
            st.write(f"{ticker} 分析失敗：{str(e)}")
    
    if failed_stocks:
        st.warning(f"無法分析的股票：{failed_stocks}")
    
    total_analyzed = matched_count + unmatched_count + len(failed_stocks)
    st.write(f"\n分析統計：共分析 {total_analyzed} 隻股票，"
             f"符合條件 {matched_count} 隻(The number of matched tickers)，不符合條件 {unmatched_count} 隻，"
             f"無法分析 {len(failed_stocks)} 隻(The number of tickers failed to analyze)")
    
    if results:
        combined_results = pd.concat(results)
        return combined_results
    else:
        return pd.DataFrame()

def screen_stocks(tickers, stock_pool, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, min_rise_126=80, max_range=5, min_adr=5, progress_bar=None):
    """主篩選函數，從 SQLite 查詢數據"""
    required_days = max(prior_days + consol_days + 30, 126 + consol_days)
    data, all_tickers = fetch_stock_data(tickers, trading_days=required_days)
    if data is None:
        st.error("無法從資料庫獲取數據")
        return pd.DataFrame()
    
    if stock_pool == "NASDAQ 100":
        tickers = get_nasdaq_100(all_tickers)
    elif stock_pool == "S&P 500":
        tickers = [t for t in get_sp500() if t in all_tickers]
    elif stock_pool == "NASDAQ All":
        tickers = get_nasdaq_all(all_tickers)
    
    st.write(f"篩選股票池：{stock_pool}，共 {len(tickers)} 隻股票")
    
    results = analyze_stock_batch(data, tickers, prior_days, consol_days, min_rise_22, min_rise_67, min_rise_126, max_range, min_adr, show_all=False)
    
    if progress_bar:
        progress_bar.progress(1.0)
    
    st.session_state['stock_data'] = data
    return results

def screen_single_stock(ticker, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, min_rise_126=80, max_range=5, min_adr=5):
    """篩選單一股票，使用 yfinance 數據"""
    required_days = max(prior_days + consol_days + 30, 126 + consol_days)
    data = fetch_yfinance_data(ticker, trading_days=required_days)
    if data is None:
        st.error(f"無法獲取 {ticker} 的數據")
        return pd.DataFrame()
    return analyze_stock_batch(data, [ticker], prior_days, consol_days, min_rise_22, min_rise_67, min_rise_126, max_range, min_adr, show_all=True)
