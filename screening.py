# screening.py
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import time
import pandas_market_calendars as mcal

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
def fetch_stock_data(ticker, trading_days=70):
    max_retries = 3
    end_date = datetime(2025, 3, 26).date()
    try:
        start_date, end_date = get_trading_days(end_date, trading_days)
    except Exception as e:
        st.error(f"無法計算交易日範圍：{str(e)}")
        return None

    for attempt in range(max_retries):
        try:
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False, timeout=15)
            if stock.empty:
                return None, f"數據為空"
            if 'Close' not in stock.columns or 'Volume' not in stock.columns:
                return None, f"缺少 'Close' 或 'Volume' 欄位"
            return stock, None
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"獲取 {ticker} 的數據失敗（第 {attempt + 1} 次嘗試）：{str(e)}，正在重試...")
                time.sleep(2)
            else:
                return None, f"重試 {max_retries} 次後仍失敗：{str(e)}"

def analyze_stock(args):
    ticker, prior_days, consol_days, min_rise_22, min_rise_67, max_range, min_adr = args
    stock, error = fetch_stock_data(ticker, trading_days=70)
    required_days = prior_days + consol_days + 30
    if stock is None:
        return None, ticker, error
    if len(stock) < required_days:
        error = f"數據不足，實際長度 {len(stock)} 天，需至少 {required_days} 天"
        return None, ticker, error
    
    close = stock['Close'].squeeze()
    volume = stock['Volume'].squeeze()
    dates = stock.index
    
    results = []
    messages = []
    has_match = False

    for i in range(-30, 0):
        if i - 22 < -len(close):
            rise_22 = 0
        else:
            price_22_days_ago = close.iloc[i - 22]
            current_price = close.iloc[i]
            rise_22 = ((current_price - price_22_days_ago) / price_22_days_ago) * 100 if price_22_days_ago != 0 else 0

        if i - 67 < -len(close):
            rise_67 = 0
        else:
            price_67_days_ago = close.iloc[i - 67]
            rise_67 = ((current_price - price_67_days_ago) / price_67_days_ago) * 100 if price_67_days_ago != 0 else 0

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
        
        if rise_22 >= min_rise_22 and rise_67 >= min_rise_67 and consolidation_range <= max_range and adr >= min_adr:
            has_match = True
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
            messages.append(f"股票 {ticker} 符合條件：22 日漲幅 = {rise_22:.2f}%, 67 日漲幅 = {rise_67:.2f}%, 盤整範圍 = {consolidation_range:.2f}%, ADR = {adr:.2f}%")
        else:
            messages.append(f"股票 {ticker} 不符合條件：22 日漲幅 = {rise_22:.2f}% (需 >= {min_rise_22}), 67 日漲幅 = {rise_67:.2f}% (需 >= {min_rise_67}), 盤整範圍 = {consolidation_range:.2f}% (需 <= {max_range}), ADR = {adr:.2f}% (需 >= {min_adr})")

    if messages:
        if has_match:
            for msg in reversed(messages):
                if "符合條件" in msg:
                    st.write(msg)
                    break
        else:
            st.write(messages[-1])

    return results, None, None

def screen_stocks(tickers, prior_days=20, consol_days=10, min_rise_22=10, min_rise_67=40, max_range=5, min_adr=5, progress_bar=None):
    total_tickers = len(tickers)
    results = []
    failed_stocks = {}

    for i, ticker in enumerate(tickers):
        try:
            result, failed_ticker, error = analyze_stock((ticker, prior_days, consol_days, min_rise_22, min_rise_67, max_range, min_adr))
            if result:
                results.extend(result)
            if failed_ticker and error:
                if error in failed_stocks:
                    failed_stocks[error].append(failed_ticker)
                else:
                    failed_stocks[error] = [failed_ticker]
        except Exception as e:
            error = f"處理時發生錯誤：{str(e)}"
            if error in failed_stocks:
                failed_stocks[error].append(ticker)
            else:
                failed_stocks[error] = [ticker]
        if progress_bar:
            progress_bar.progress(min((i + 1) / total_tickers, 1.0))
        time.sleep(0.05)

    if failed_stocks:
        for error, tickers in failed_stocks.items():
            st.warning(f"無法獲取以下股票的數據：{tickers}，原因：{error}")

    return pd.DataFrame(results)
