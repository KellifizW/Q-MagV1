import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from multiprocessing import Pool

# 動態獲取 NASDAQ 100 成分股
def get_nasdaq_100():
    try:
        return pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'].tolist()
    except Exception:
        return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'ADBE', 'PYPL', 'INTC']

nasdaq_100 = get_nasdaq_100()

def fetch_stock_data(ticker, days=90):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return stock if not stock.empty else None

def analyze_stock(ticker, prior_days=20, consol_days=10, min_rise=30, max_range=5, min_adr=5):
    stock = fetch_stock_data(ticker)
    if stock is None or len(stock) < prior_days + consol_days + 30:
        return None
    
    close = stock['Close']
    volume = stock['Volume']
    dates = stock.index
    
    results = []
    for i in range(-30, 0):
        if i < -prior_days:
            prior_rise = (close.iloc[i] / close.iloc[i - prior_days] - 1) * 100
            recent_high = close.iloc[i - consol_days:i].max()
            recent_low = close.iloc[i - consol_days:i].min()
            consolidation_range = (recent_high / recent_low - 1) * 100
            vol_decline = volume.iloc[i - consol_days:i].mean() < volume.iloc[i - prior_days:i - consol_days].mean()
            
            # 分步計算 ADR，確保純量
            high = stock['High'].iloc[i - prior_days:i]
            low = stock['Low'].iloc[i - prior_days:i]
            prev_close = stock['Close'].shift(1).iloc[i - prior_days:i]
            
            # 檢查數據完整性
            if high.empty or low.empty or prev_close.empty or prev_close.isna().all():
                adr = 0
            else:
                daily_range = (high - low) / prev_close
                adr = daily_range.mean() * 100 if not pd.isna(daily_range.mean()) else 0
            
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

def screen_stocks(tickers, prior_days=20, consol_days=10, min_rise=30, max_range=5, min_adr=5):
    with Pool() as pool:
        results = pool.starmap(analyze_stock, [(ticker, prior_days, consol_days, min_rise, max_range, min_adr) for ticker in tickers])
    all_results = [r for sublist in results if sublist for r in sublist]
    return pd.DataFrame(all_results)
