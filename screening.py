import pandas as pd
import streamlit as st
import numpy as np
import logging
from database import fetch_stock_data, fetch_yfinance_data

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_nasdaq_100(csv_tickers):
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        df = tables[4]
        if 'Ticker' not in df.columns:
            raise KeyError("未找到 'Ticker' 列")
        nasdaq_100 = df['Ticker'].tolist()
        return [ticker for ticker in nasdaq_100 if ticker in csv_tickers]
    except Exception as e:
        logger.error(f"獲取 NASDAQ 100 失敗: {str(e)}")
        return [ticker for ticker in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA'] if ticker in csv_tickers]

def get_sp500():
    try:
        return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    except Exception as e:
        logger.error(f"獲取 S&P 500 失敗: {str(e)}")
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA']

def get_nasdaq_all(csv_tickers):
    return csv_tickers

def analyze_stock_batch(data, tickers, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, min_rise_126=80, 
                        max_range=5, min_adr=5, use_candle_strength=True, show_all=False,
                        use_volume_filter=False, min_volume=1000, use_intraday_rise=False, min_intraday_rise=10):
    logger.info(f"開始分析股票批次，股票數量: {len(tickers)}")
    results = []
    matched_tickers = set()
    required_days = max(prior_days + consol_days + 30, 126 + consol_days)
    current_date = pd.Timestamp.now(tz='US/Eastern').normalize()
    logger.info(f"當前日期: {current_date}, 所需天數: {required_days}")
    
    for ticker in tickers:
        logger.info(f"處理股票: {ticker}")
        if isinstance(data.columns, pd.MultiIndex):
            stock = data.xs(ticker, level='Ticker', axis=1)
        else:
            stock = data[[col for col in data.columns if ticker in col]].copy()
            stock.columns = [col[0] for col in stock.columns]
        
        if not isinstance(stock, pd.DataFrame):
            logger.warning(f"股票 {ticker} 的數據不是 DataFrame，跳過")
            continue
        
        # 提取數據並統一索引
        logger.info(f"提取 {ticker} 的收盤價、成交量、高價、低價")
        stock.index = pd.to_datetime(stock.index)
        if stock.index.tz is None:
            stock.index = stock.index.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT')
        else:
            stock.index = stock.index.tz_convert('US/Eastern')
        logger.info(f"{ticker} 索引時區狀態: {'已帶時區' if stock.index.tz else '無時區'}")
        
        dates = stock.index.dropna()
        close = pd.Series(stock['Close']).dropna()
        volume = pd.Series(stock['Volume']).dropna()
        high = pd.Series(stock['High']).dropna()
        low = pd.Series(stock['Low']).dropna()
        
        logger.info(f"日期範圍: {dates.min()} 至 {dates.max()}, 總天數: {len(dates)}")
        
        if len(dates) < required_days:
            logger.warning(f"股票 {ticker} 的數據天數 ({len(dates)}) 小於所需天數 ({required_days})，跳過")
            continue
        
        valid_mask = dates <= current_date
        valid_dates = dates[valid_mask]
        if len(valid_dates) < required_days:
            logger.warning(f"股票 {ticker} 的有效天數 ({len(valid_dates)}) 小於所需天數 ({required_days})，跳過")
            continue
        
        # 使用 stock 的索引作為基準
        common_index = valid_dates
        close = close.reindex(common_index)
        volume = volume.reindex(common_index)
        high = high.reindex(common_index)
        low = low.reindex(common_index)
        logger.info(f"共同索引天數: {len(common_index)}")
        
        if len(common_index) < required_days:
            logger.warning(f"股票 {ticker} 的共同索引天數 ({len(common_index)}) 小於所需天數 ({required_days})，跳過")
            continue
        
        prev_close = close.shift(1)
        dates = common_index
        
        # 漲幅計算
        logger.info(f"計算 {ticker} 的漲幅")
        close_shift_22 = close.shift(22)
        close_shift_67 = close.shift(67)
        close_shift_126 = close.shift(126)
        rise_22 = (close / close_shift_22 - 1) * 100
        rise_67 = (close / close_shift_67 - 1) * 100
        rise_126 = (close / close_shift_126 - 1) * 100
        
        # 盤整與突破計算
        logger.info(f"計算 {ticker} 的盤整與突破條件")
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
        
        # 新增成交量篩選
        volume_condition = True
        if use_volume_filter:
            recent_volume = volume.tail(10)  # 最近10日成交量
            volume_condition = (recent_volume.max() >= min_volume * 10000)  # 轉換為股數
        
        # 新增日內價格上漲篩選
        intraday_rise_condition = True
        if use_intraday_rise:
            intraday_range = ((high - low) / prev_close * 100).tail(5)  # 最近5日日內幅度
            intraday_rise_condition = (intraday_range.max() >= min_intraday_rise)
        
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
               (consolidation_range <= max_range) & (adr >= min_adr) & volume_condition & intraday_rise_condition
        
        if show_all:
            mask_full = mask & breakout & breakout_volume & candle_strength
            valid_dates = dates[-1:]
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
                'Target_20%': targets_filtered['20%'],
                'Target_50%': targets_filtered['50%'],
                'Target_100%': targets_filtered['100%']
            })
            results.append(matched)
        elif mask.any():
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
                if dates[-1] in valid_dates:
                    matched_tickers.add(ticker)
                    st.write(f"股票 {ticker} 符合條件（最新）：22 日漲幅 = {rise_22.iloc[-1]:.2f}%, "
                             f"67 日漲幅 = {rise_67.iloc[-1]:.2f}%, 126 日漲幅 = {rise_126.iloc[-1]:.2f}%, "
                             f"盤整範圍 = {consolidation_range.iloc[-1]:.2f}%, ADR = {adr.iloc[-1]:.2f}%")
    
    if results:
        combined_results = pd.concat(results)
        st.write(f"找到 {len(combined_results)} 筆記錄（{len(matched_tickers)} 隻股票）")
        logger.info(f"分析完成，總記錄數: {len(combined_results)}, 符合股票數: {len(matched_tickers)}")
        return combined_results
    else:
        if show_all:
            logger.info("分析完成，無數據返回（可能是數據處理失敗）")
            return pd.DataFrame()
        else:
            st.write("無符合條件的股票")
            logger.info("分析完成，無符合條件的股票")
            return pd.DataFrame()

def screen_stocks(tickers, stock_pool, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, min_rise_126=80, 
                  max_range=5, min_adr=5, use_candle_strength=True, progress_bar=None,
                  use_volume_filter=False, min_volume=1000, use_intraday_rise=False, min_intraday_rise=10):
    required_days = max(prior_days + consol_days + 30, 126 + consol_days)
    logger.info(f"開始篩選股票，所需天數: {required_days}")
    data, all_tickers = fetch_stock_data(tickers, trading_days=required_days)
    if data is None:
        st.error("無法從資料庫獲取數據")
        logger.error("無法從資料庫獲取數據")
        return pd.DataFrame()
    
    if stock_pool == "NASDAQ 100":
        tickers = get_nasdaq_100(all_tickers)
    elif stock_pool == "S&P 500":
        tickers = [t for t in get_sp500() if t in all_tickers]
    elif stock_pool == "NASDAQ All":
        tickers = get_nasdaq_all(all_tickers)
    
    st.write(f"篩選股票池：{stock_pool}，共 {len(tickers)} 隻股票")
    logger.info(f"篩選股票池: {stock_pool}，股票數: {len(tickers)}")
    
    results = analyze_stock_batch(data, tickers, prior_days, consol_days, min_rise_22, min_rise_67, min_rise_126, 
                                  max_range, min_adr, use_candle_strength, show_all=False,
                                  use_volume_filter=use_volume_filter, min_volume=min_volume,
                                  use_intraday_rise=use_intraday_rise, min_intraday_rise=min_intraday_rise)
    
    if progress_bar:
        progress_bar.progress(1.0)
    
    st.session_state['stock_data'] = data
    return results

def screen_single_stock(ticker, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, min_rise_126=80, 
                        max_range=5, min_adr=5, use_candle_strength=True,
                        use_volume_filter=False, min_volume=1000, use_intraday_rise=False, min_intraday_rise=10):
    required_days = max(prior_days + consol_days + 30, 126 + consol_days)
    logger.info(f"開始單一股票查詢: {ticker}，所需天數: {required_days}")
    data = fetch_yfinance_data(ticker, trading_days=required_days)
    if data is None:
        logger.error(f"無法獲取 {ticker} 的數據")
        return pd.DataFrame()
    logger.info(f"成功獲取 {ticker} 的數據，開始分析")
    return analyze_stock_batch(data, [ticker], prior_days, consol_days, min_rise_22, min_rise_67, min_rise_126, 
                               max_range, min_adr, use_candle_strength, show_all=True,
                               use_volume_filter=use_volume_filter, min_volume=min_volume,
                               use_intraday_rise=use_intraday_rise, min_intraday_rise=min_intraday_rise)
