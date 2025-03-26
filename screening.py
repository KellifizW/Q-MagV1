import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from multiprocessing import Pool
import streamlit as st

def get_nasdaq_100():
    try:
        # 從 Wikipedia 獲取所有表格
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        st.write(f"從 Wikipedia 獲取到 {len(tables)} 個表格")
        
        # 遍歷所有表格，尋找包含股票代碼的表格
        for i, df in enumerate(tables):
            columns = df.columns.tolist()
            st.write(f"表格 {i} 的列名: {columns}")
            
            # 檢查是否有 'Ticker' 或類似列名（如 'Symbol'）
            ticker_col = None
            for col in columns:
                if str(col).lower() in ['ticker', 'symbol']:
                    ticker_col = col
                    break
            
            if ticker_col:
                tickers = df[ticker_col].tolist()
                # 過濾無效值並確保是字符串
                tickers = [str(ticker) for ticker in tickers if pd.notna(ticker) and str(ticker).strip()]
                st.write(f"在表格 {i} 中找到股票清單，包含 {len(tickers)} 隻股票")
                return tickers
        
        # 如果沒有找到合適的表格
        st.error("未找到包含 'Ticker' 或 'Symbol' 列的表格")
        raise ValueError("未找到股票清單表格")
    
    except Exception as e:
        st.error(f"無法從 Wikipedia 獲取 Nasdaq-100 清單: {str(e)}")
        # 返回更完整的後備清單
        backup = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'ADBE', 'PYPL', 'INTC',
                  'NFLX', 'CSCO', 'PEP', 'AVGO', 'COST', 'TMUS', 'AMD', 'TXN', 'QCOM', 'AMGN']
        st.write(f"使用後備清單，包含 {len(backup)} 隻股票")
        return backup

def get_sp500():
    try:
        return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    except Exception as e:
        st.error(f"無法從 Wikipedia 獲取 S&P 500 清單: {e}")
        return get_nasdaq_100()  # 後備使用 Nasdaq 100

def get_nasdaq_all():
    try:
        # 使用 Nasdaq 官方 CSV 檔案
        nasdaq = pd.read_csv('https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download')
        return nasdaq['Symbol'].tolist()
    except Exception as e:
        st.error(f"無法從 Nasdaq 獲取股票清單: {e}")
        return get_nasdaq_100()  # 後備使用 Nasdaq 100

@st.cache_data(ttl=3600)  # 快取 1 小時
def fetch_stock_data(ticker, days=90):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return stock if not stock.empty else None

def analyze_stock(args):
    ticker, prior_days, consol_days, min_rise, max_range, min_adr = args
    stock = fetch_stock_data(ticker)
    if stock is None or len(stock) < prior_days + consol_days + 30:
        return None
    
    close = stock['Close'].squeeze()
    volume = stock['Volume'].squeeze()
    dates = stock.index
    
    results = []
    for i in range(-30, 0):
        if i < -prior_days:
            prior_rise = (close.iloc[i] / close.iloc[i - prior_days] - 1) * 100
            recent_high = close.iloc[i - consol_days:i].max()
            recent_low = close.iloc[i - consol_days:i].min()
            consolidation_range = (recent_high / recent_low - 1) * 100
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
            
            if prior_rise > min_rise and consolidation_range < max_range and adr > min_adr:
                results.append({
                    'Ticker': ticker,
                    'Date': dates[i].strftime('%Y-%m-%d'),
                    'Price': close.iloc[i],
                    'Prior_Rise_%': prior_rise,
                    'Consolidation_Range_%': consolidation_range,
                    'ADR_%': adr,
                    'Breakout': breakout,
                    'Breakout_Volume': breakout_volume
                })
    return results

def screen_stocks(tickers, prior_days=20, consol_days=10, min_rise=30, max_range=5, min_adr=5, progress_bar=None):
    with Pool() as pool:
        total_tickers = len(tickers)
        results = []
        for i, result in enumerate(pool.imap_unordered(analyze_stock, 
                                                       [(ticker, prior_days, consol_days, min_rise, max_range, min_adr) 
                                                        for ticker in tickers])):
            if result:
                results.extend(result)
            if progress_bar:
                progress_bar.progress(min((i + 1) / total_tickers, 1.0))  # 更新進度條
    return pd.DataFrame(results)
