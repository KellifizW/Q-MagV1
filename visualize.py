# visualize.py
import streamlit as st
import pandas as pd
import altair as alt
from screening import fetch_stock_data

def plot_top_5_stocks(top_5_tickers):
    """繪製前 5 名股票的走勢圖（包含股價、成交量、10 日均線和 MACD）"""
    stock_data_batch = st.session_state.get('stock_data', None)
    
    for ticker in top_5_tickers:
        st.write(f"正在處理股票 {ticker}...")
        
        if stock_data_batch is not None and ticker in stock_data_batch.columns.get_level_values(1):
            try:
                stock_data = stock_data_batch.xs(ticker, level='Ticker', axis=1).reset_index()
                error = None
            except KeyError as e:
                stock_data = None
                error = f"無法從批量數據中提取 {ticker}：{str(e)}"
        else:
            stock_data, error = fetch_stock_data(ticker, trading_days=70)
            if stock_data is not None:
                stock_data = stock_data.reset_index()
        
        if stock_data is None or stock_data.empty:
            st.error(f"無法繪製 {ticker} 的圖表：{error if error else '數據為空'}")
            continue
        
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        st.write(f"{ticker} 數據長度：{len(stock_data)} 筆")
        st.write(f"{ticker} 數據日期範圍：{stock_data['Date'].min()} 到 {stock_data['Date'].max()}")
        
        required_columns = ['Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        if missing_columns:
            st.error(f"無法繪製 {ticker} 的圖表：缺少欄位 {missing_columns}")
            continue
        
        try:
            stock_data = stock_data.dropna()
            if len(stock_data) < 10:
                st.error(f"清理後 {ticker} 的數據長度 {len(stock_data)} 小於最小要求 10")
                continue
            
            stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
            stock_data['EMA12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
            stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
            stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
            stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
            stock_data['Histogram'] = stock_data['MACD'] - stock_data['Signal']
            
            # 價格和成交量圖表
            price_base = alt.Chart(stock_data).encode(
                x=alt.X('Date:T', title='日期'),
                tooltip=['Date:T', 'Close:Q', 'MA10:Q', 'Volume:Q']
            )
            
            price_line = price_base.mark_line(color='blue').encode(
                y=alt.Y('Close:Q', title='價格')
            )
            ma10_line = price_base.mark_line(color='orange').encode(
                y='MA10:Q'
            )
            volume_bar = price_base.mark_bar(opacity=0.3, color='pink').encode(
                y=alt.Y('Volume:Q', title='成交量', axis=alt.Axis(orient='right'))
            )
            
            price_chart = alt.layer(price_line, ma10_line, volume_bar).resolve_scale(
                y='independent'
            ).properties(
                height=500,
                title=f'{ticker} 近 3 個月走勢'
            )
            
            # MACD 圖表
            macd_base = alt.Chart(stock_data).encode(
                x=alt.X('Date:T', title='日期'),
                tooltip=['Date:T', 'MACD:Q', 'Signal:Q', 'Histogram:Q']
            )
            
            macd_line = macd_base.mark_line(color='blue').encode(
                y=alt.Y('MACD:Q', title='MACD')
            )
            signal_line = macd_base.mark_line(color='orange').encode(
                y='Signal:Q'
            )
            histogram_bar = macd_base.mark_bar(opacity=0.5, color='gray').encode(
                y='Histogram:Q'
            )
            
            macd_chart = alt.layer(macd_line, signal_line, histogram_bar).properties(
                height=200,
                title='MACD'
            )
            
            # 組合圖表
            final_chart = alt.vconcat(price_chart, macd_chart).resolve_scale(
                x='shared'
            ).configure_axisX(
                labelAngle=45,
                labelFontSize=10
            ).properties(
                title=f'{ticker} 近 3 個月走勢與 MACD'
            )
            
            st.altair_chart(final_chart, use_container_width=True)
            st.write(f"成功繪製 {ticker} 的圖表")
            
        except Exception as e:
            st.error(f"繪製 {ticker} 的圖表時發生錯誤：{str(e)}")

def plot_breakout_stocks(breakout_tickers, consol_days):
    """繪製突破股票的圖表（包含股價、阻力位、支撐位、買入信號、成交量和 MACD）"""
    stock_data_batch = st.session_state.get('stock_data', None)
    
    for ticker in breakout_tickers:
        st.write(f"正在處理突破股票 {ticker}...")
        
        if stock_data_batch is not None and ticker in stock_data_batch.columns.get_level_values(1):
            try:
                stock_data = stock_data_batch.xs(ticker, level='Ticker', axis=1).reset_index()
                error = None
            except KeyError as e:
                stock_data = None
                error = f"無法從批量數據中提取 {ticker}：{str(e)}"
        else:
            stock_data, error = fetch_stock_data(ticker, trading_days=70)
            if stock_data is not None:
                stock_data = stock_data.reset_index()
        
        if stock_data is None or stock_data.empty:
            st.error(f"無法繪製 {ticker} 的圖表：{error if error else '數據為空'}")
            continue
        
        min_required_length = max(consol_days + 1, 26)
        if len(stock_data) < min_required_length:
            st.error(f"無法繪製 {ticker} 的圖表：數據長度 {len(stock_data)} 小於最小要求 {min_required_length}")
            continue
        
        try:
            stock_data = stock_data.dropna()
            recent_high = stock_data['Close'][-consol_days-1:-1].max()
            recent_low = stock_data['Close'][-consol_days-1:-1].min()
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            stock_data['EMA12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
            stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
            stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']
            stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
            stock_data['Histogram'] = stock_data['MACD'] - stock_data['Signal']
            
            # 價格和成交量圖表
            price_base = alt.Chart(stock_data).encode(
                x=alt.X('Date:T', title='日期'),
                tooltip=['Date:T', 'Close:Q', 'Volume:Q']
            )
            
            price_line = price_base.mark_line(color='blue').encode(
                y=alt.Y('Close:Q', title='價格')
            )
            volume_bar = price_base.mark_bar(opacity=0.3, color='pink').encode(
                y=alt.Y('Volume:Q', title='成交量', axis=alt.Axis(orient='right'))
            )
            
            # 阻力位和支撐位
            resistance = alt.Chart(pd.DataFrame({'y': [recent_high]})).mark_rule(color='red', strokeDash=[5,5]).encode(
                y='y:Q'
            )
            support = alt.Chart(pd.DataFrame({'y': [recent_low]})).mark_rule(color='green', strokeDash=[5,5]).encode(
                y='y:Q'
            )
            buy_signal = price_base.mark_point(size=100, color='blue', shape='star').encode(
                y='Close:Q'
            ).transform_filter(
                alt.datum.Date == stock_data['Date'].iloc[-1]
            )
            
            price_chart = alt.layer(price_line, volume_bar, resistance, support, buy_signal).resolve_scale(
                y='independent'
            ).properties(
                height=500,
                title=f'{ticker} 突破圖表（買入信號）'
            )
            
            # MACD 圖表
            macd_base = alt.Chart(stock_data).encode(
                x=alt.X('Date:T', title='日期'),
                tooltip=['Date:T', 'MACD:Q', 'Signal:Q', 'Histogram:Q']
            )
            
            macd_line = macd_base.mark_line(color='blue').encode(
                y=alt.Y('MACD:Q', title='MACD')
            )
            signal_line = macd_base.mark_line(color='orange').encode(
                y='Signal:Q'
            )
            histogram_bar = macd_base.mark_bar(opacity=0.5, color='gray').encode(
                y='Histogram:Q'
            )
            
            macd_chart = alt.layer(macd_line, signal_line, histogram_bar).properties(
                height=200,
                title='MACD'
            )
            
            # 組合圖表
            final_chart = alt.vconcat(price_chart, macd_chart).resolve_scale(
                x='shared'
            ).configure_axisX(
                labelAngle=45,
                labelFontSize=10
            ).properties(
                title=f'{ticker} 突破圖表（買入信號）與 MACD'
            )
            
            st.altair_chart(final_chart, use_container_width=True)
            st.write(f"成功繪製 {ticker} 的圖表")
            
        except Exception as e:
            st.error(f"繪製 {ticker} 的圖表時發生錯誤：{str(e)}")
