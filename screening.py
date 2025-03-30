import pandas as pd
import streamlit as st
import numpy as np
from database import fetch_stock_data, fetch_yfinance_data

def get_nasdaq_100(csv_tickers):
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        df = tables[4]
        if 'Ticker' not in df.columns:
            raise KeyError("未找到 'Ticker' 列")
        nasdaq_100 = df['Ticker'].tolist()
        return [ticker for ticker in nasdaq_100 if ticker in csv_tickers]
    except Exception as e:
        return [ticker for ticker in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA'] if ticker in csv_tickers]

def get_sp500():
    try:
        return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    except Exception as e:
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA']

def get_nasdaq_all(csv_tickers):
    return csv_tickers

def analyze_stock_batch(data, tickers, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, min_rise_126=80, 
                        max_range=5, min_adr=5, use_candle_strength=True, show_all=False):
    results = []
    matched_tickers = set()  # 用於記錄最新日期符合條件的股票
    required_days = max(prior_days + consol_days + 30, 126 + consol_days)
    current_date = pd.Timestamp.now(tz='US/Eastern').normalize()  # 使用 pd.Timestamp 並確保帶時區
    
    for ticker in tickers:
        if isinstance(data.columns, pd.MultiIndex):
            stock = data.xs(ticker, level='Ticker', axis=1)
        else:
            stock = data[[col for col in data.columns if ticker in col]].copy()
            stock.columns = [col[0] for col in stock.columns]
        
        if not isinstance(stock, pd.DataFrame):
            continue
        
        # 提取並過濾數據
        close = pd.Series(stock['Close']).dropna()
        volume = pd.Series(stock['Volume']).dropna()
        high = pd.Series(stock['High']).dropna()
        low = pd.Series(stock['Low']).dropna()
        # 檢查索引是否帶時區，若無則本地化，若有則轉換
        dates = pd.to_datetime(stock.index)
        if dates.tz is None:
            dates = dates.tz_localize('US/Eastern')
        else:
            dates = dates.tz_convert('US/Eastern')
        
        if len(close) < required_days:
            continue
        
        valid_mask = dates <= current_date
        valid_dates = dates[valid_mask]
        if len(valid_dates) < required_days:
            continue
        
        common_index = valid_dates.intersection(close.index).intersection(volume.index).intersection(high.index).intersection(low.index)
        if len(common_index) < required_days:
            continue
        
        close = close.loc[common_index]
        volume = volume.loc[common_index]
        high = high.loc[common_index]
        low = low.loc[common_index]
        prev_close = close.shift(1)
        dates = common_index
        
        # 漲幅計算
        close_shift_22 = close.shift(22)
        close_shift_67 = close.shift(67)
        close_shift_126 = close.shift(126)
        rise_22 = (close / close_shift_22 - 1) * 100
        rise_67 = (close / close_shift_67 - 1) * 100
        rise_126 = (close / close_shift_126 - 1) * 100
        
        if rise_126.isna().all() or rise_67.isna().all() or rise_22.isna().all():
            continue
        
        # 盤整與突破計算
        recent_high = close.rolling(consol_days).max()
        recent_low = close.rolling(consol_days).min()
        consolidation_range = (recent_high / recent_low - 1) * 100
        vol_decline = volume.rolling(consol_days).mean() < volume.shift(consol_days).rolling(prior_days).mean()
        daily_range = (high - low) / prev_close
        adr = daily_range.rolling(prior_days).mean() * 100
        breakout = (close > recent_high.shift(1)) & (close.shift(1) <= recent_high.shift(1))
        breakout_volume = volume > volume.rolling(10).mean() * 1.5
        
        # K線強度計算
        candle_strength = (close - low) / (high - low) > 0.7 if use_candle_strength else pd.Series(True, index=close.index)
        
        # 風險管理計算
        stop_loss = recent_low
        breakout_price = close[breakout & breakout_volume & candle_strength]
        if not breakout_price.empty:
            targets = pd.DataFrame({
                '20%': breakout_price * 1.2,
                '50%': breakout_price * 1.5,
                '100%': breakout_price * 2.0
            }, index=breakout_price.index)
        else:
            targets = pd.DataFrame(index=close.index, columns=['20%', '50%', '100%'])
        
        # 篩選條件
        mask = (rise_22 >= min_rise_22) & (rise_67 >= min_rise_67) & (rise_126 >= min_rise_126) & \
               (consolidation_range <= max_range) & (adr >= min_adr)
        
        if show_all:
            mask = mask & breakout & breakout_volume & candle_strength
        
        if mask.any():
            valid_dates = dates[mask]
            valid_dates = valid_dates[valid_dates.isin(dates)]
            if not valid_dates.empty:
                targets_filtered = targets.reindex(valid_dates).fillna(method='ffill') if not targets.empty else pd.DataFrame(index=valid_dates, columns=['20%', '50%', '100%'])
                matched = pd.DataFrame({
                    'Ticker': ticker,
                    'Date': valid_dates,
                    'Price': close.loc[valid_dates],
                    'Prior_Rise_22_%': rise_22.loc[valid_dates],
                    'Prior_Rise_67_%': rise_67.loc[valid_dates],
                    'Prior_Rise_126_%': rise_126.loc[valid_dates],
                    'Consolidation_Range_%': consolidation_range.loc[valid_dates],
                    'ADR_%': adr.loc[valid_dates],
                    'Breakout': breakout.loc[valid_dates],
                    'Breakout_Volume': breakout_volume.loc[valid_dates],
                    'Candle_Strength': candle_strength.loc[valid_dates],
                    'Stop_Loss': stop_loss.loc[valid_dates],
                    'Target_20%': targets_filtered['20%'] if not targets_filtered.empty else pd.Series([None] * len(valid_dates)),
                    'Target_50%': targets_filtered['50%'] if not targets_filtered.empty else pd.Series([None] * len(valid_dates)),
                    'Target_100%': targets_filtered['100%'] if not targets_filtered.empty else pd.Series([None] * len(valid_dates))
                })
                results.append(matched)
                
                # 只顯示最新日期符合條件的股票，並記錄到 matched_tickers
                if dates[-1] in valid_dates:
                    matched_tickers.add(ticker)
                    st.write(f"股票 {ticker} 符合條件（最新）：22 日漲幅 = {rise_22.iloc[-1]:.2f}%, "
                             f"67 日漲幅 = {rise_67.iloc[-1]:.2f}%, 126 日漲幅 = {rise_126.iloc[-1]:.2f}%, "
                             f"盤整範圍 = {consolidation_range.iloc[-1]:.2f}%, ADR = {adr.iloc[-1]:.2f}%")
    
    if results:
        combined_results = pd.concat(results)
        st.write(f"找到 {len(combined_results)} 筆符合條件的記錄（{len(matched_tickers)} 隻股票）")
        return combined_results
    else:
        st.write("無符合條件的股票")
        return pd.DataFrame()

def screen_stocks(tickers, stock_pool, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, min_rise_126=80, 
                  max_range=5, min_adr=5, use_candle_strength=True, progress_bar=None):
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
    
    results = analyze_stock_batch(data, tickers, prior_days, consol_days, min_rise_22, min_rise_67, min_rise_126, 
                                  max_range, min_adr, use_candle_strength, show_all=False)
    
    if progress_bar:
        progress_bar.progress(1.0)
    
    st.session_state['stock_data'] = data
    return results

def screen_single_stock(ticker, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, min_rise_126=80, 
                        max_range=5, min_adr=5, use_candle_strength=True):
    required_days = max(prior_days + consol_days + 30, 126 + consol_days)
    data = fetch_yfinance_data(ticker, trading_days=required_days)
    if data is None:
        return pd.DataFrame()
    return analyze_stock_batch(data, [ticker], prior_days, consol_days, min_rise_22, min_rise_67, min_rise_126, 
                               max_range, min_adr, use_candle_strength, show_all=True)
