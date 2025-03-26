# visualize.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from screening import fetch_stock_data

def plot_top_5_stocks(top_5_tickers):
    """繪製前 5 名股票的走勢圖（包含股價、成交量、10 日均線和 MACD）"""
    stock_data_batch = st.session_state.get('stock_data', None)
    
    for ticker in top_5_tickers:
        st.write(f"正在處理股票 {ticker}...")
        
        # 優先使用批量數據
        if stock_data_batch is not None and ticker in stock_data_batch.columns.get_level_values(1):
            stock_data = stock_data_batch[ticker]
            error = None
        else:
            stock_data, error = fetch_stock_data(ticker, trading_days=70)
        
        if stock_data is None or stock_data.empty:
            st.error(f"無法繪製 {ticker} 的圖表：{error if error else '數據為空'}")
            continue
        
        st.write(f"{ticker} 數據長度：{len(stock_data)} 筆")
        st.write(f"{ticker} 數據日期範圍：{stock_data.index[0]} 到 {stock_data.index[-1]}")
        
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
            
            dates = [date.strftime("%Y-%m-%d") for date in stock_data.index]
            prices = stock_data["Close"].astype(float).tolist()
            volumes = stock_data["Volume"].astype(int).tolist()
            
            stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
            ma10 = stock_data['MA10'].astype(float).tolist()
            
            ema12 = stock_data["Close"].ewm(span=12, adjust=False).mean()
            ema26 = stock_data["Close"].ewm(span=26, adjust=False).mean()
            macd_line = (ema12 - ema26).tolist()
            macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().tolist()
            macd_histogram = (pd.Series(macd_line) - pd.Series(macd_signal)).tolist()
            
            x_indices = list(range(len(dates)))
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                subplot_titles=(f"{ticker} 近 3 個月走勢", "MACD"), row_heights=[0.7, 0.3],
                                specs=[[{"secondary_y": True}], [{}]])
            
            customdata_price = [[dates[i], ma10[i]] for i in range(len(dates))]
            fig.add_trace(go.Scatter(x=x_indices, y=prices, mode="lines", name="股價", line=dict(color="blue"),
                                     hovertemplate="日期: %{customdata[0]}<br>股價: %{y:.2f}<br>10 日均線: %{customdata[1]:.2f}",
                                     customdata=customdata_price), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=x_indices, y=ma10, mode="lines", name="10 日均線", line=dict(color="orange"),
                                     hovertemplate="日期: %{customdata}<br>10 日均線: %{y:.2f}", customdata=dates),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Bar(x=x_indices, y=volumes, name="成交量", opacity=0.3, marker_color="pink",
                                 hovertemplate="日期: %{customdata}<br>成交量: %{y:,.0f}", customdata=dates),
                          row=1, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(x=x_indices, y=macd_line, mode="lines", name="MACD 線", line=dict(color="blue"),
                                     hovertemplate="日期: %{customdata}<br>MACD: %{y:.2f}", customdata=dates),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=x_indices, y=macd_signal, mode="lines", name="訊號線", line=dict(color="orange"),
                                     hovertemplate="日期: %{customdata}<br>訊號線: %{y:.2f}", customdata=dates),
                          row=2, col=1)
            fig.add_trace(go.Bar(x=x_indices, y=macd_histogram, name="MACD 柱狀圖", marker_color="gray", opacity=0.5,
                                 hovertemplate="日期: %{customdata}<br>MACD 柱狀圖: %{y:.2f}", customdata=dates),
                          row=2, col=1)
            
            fig.update_layout(title=f"{ticker} 近 3 個月走勢與 MACD", xaxis_title="日期", showlegend=True, height=800,
                              margin=dict(b=100))
            fig.update_yaxes(title_text="價格", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="成交量", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            
            tickvals = x_indices[::5]
            ticktext = [dates[i] for i in tickvals]
            fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, tickfont=dict(size=10), row=1, col=1)
            fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, tickfont=dict(size=10), row=2, col=1)
            
            st.plotly_chart(fig)
            st.write(f"成功繪製 {ticker} 的圖表")
        except Exception as e:
            st.error(f"繪製 {ticker} 的圖表時發生錯誤：{str(e)}")

def plot_breakout_stocks(breakout_tickers, consol_days):
    """繪製突破股票的圖表（包含股價、阻力位、支撐位、買入信號、成交量和 MACD）"""
    stock_data_batch = st.session_state.get('stock_data', None)
    
    for ticker in breakout_tickers:
        st.write(f"正在處理突破股票 {ticker}...")
        
        if stock_data_batch is not None and ticker in stock_data_batch.columns.get_level_values(1):
            stock_data = stock_data_batch[ticker]
            error = None
        else:
            stock_data, error = fetch_stock_data(ticker, trading_days=70)
        
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
            
            dates = [date.strftime("%Y-%m-%d") for date in stock_data.index]
            prices = stock_data["Close"].astype(float).tolist()
            volumes = stock_data["Volume"].astype(int).tolist()
            
            ema12 = stock_data["Close"].ewm(span=12, adjust=False).mean()
            ema26 = stock_data["Close"].ewm(span=26, adjust=False).mean()
            macd_line = (ema12 - ema26).tolist()
            macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().tolist()
            macd_histogram = (pd.Series(macd_line) - pd.Series(macd_signal)).tolist()
            
            x_indices = list(range(len(dates)))
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                subplot_titles=(f"{ticker} 突破圖表（買入信號）", "MACD"), row_heights=[0.7, 0.3],
                                specs=[[{"secondary_y": True}], [{}]])
            
            fig.add_trace(go.Scatter(x=x_indices, y=prices, mode="lines", name="價格", line=dict(color="blue"),
                                     hovertemplate="日期: %{customdata}<br>價格: %{y:.2f}", customdata=dates),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=[x_indices[0], x_indices[-1]], y=[recent_high, recent_high], mode='lines',
                                     line=dict(dash='dash', color='red'), name='阻力位', hovertemplate="阻力位: %{y:.2f}"),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=[x_indices[0], x_indices[-1]], y=[recent_low, recent_low], mode='lines',
                                     line=dict(dash='dash', color='green'), name='支撐位', hovertemplate="支撐位: %{y:.2f}"),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=[x_indices[-1]], y=[prices[-1]], mode='markers', marker=dict(size=12, color='blue', symbol='star'),
                                     name='買入信號', hovertemplate="日期: %{customdata}<br>價格: %{y:.2f}", customdata=[dates[-1]]),
                          row=1, col=1, secondary_y=False)
            fig.add_trace(go.Bar(x=x_indices, y=volumes, name="成交量", opacity=0.3, marker_color="pink",
                                 hovertemplate="日期: %{customdata}<br>成交量: %{y:,.0f}", customdata=dates),
                          row=1, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(x=x_indices, y=macd_line, mode="lines", name="MACD 線", line=dict(color="blue"),
                                     hovertemplate="日期: %{customdata}<br>MACD: %{y:.2f}", customdata=dates),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=x_indices, y=macd_signal, mode="lines", name="訊號線", line=dict(color="orange"),
                                     hovertemplate="日期: %{customdata}<br>訊號線: %{y:.2f}", customdata=dates),
                          row=2, col=1)
            fig.add_trace(go.Bar(x=x_indices, y=macd_histogram, name="MACD 柱狀圖", marker_color="gray", opacity=0.5,
                                 hovertemplate="日期: %{customdata}<br>MACD 柱狀圖: %{y:.2f}", customdata=dates),
                          row=2, col=1)
            
            fig.update_layout(title=f"{ticker} 突破圖表（買入信號）與 MACD", xaxis_title="日期", showlegend=True, height=800,
                              margin=dict(b=100))
            fig.update_yaxes(title_text="價格", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="成交量", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            
            tickvals = x_indices[::5]
            ticktext = [dates[i] for i in tickvals]
            fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, tickfont=dict(size=10), row=1, col=1)
            fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=45, tickfont=dict(size=10), row=2, col=1)
            
            st.plotly_chart(fig)
            st.write(f"成功繪製 {ticker} 的圖表")
        except Exception as e:
            st.error(f"繪製 {ticker} 的圖表時發生錯誤：{str(e)}")
