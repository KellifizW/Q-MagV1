# visualize.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from screening import fetch_stock_data

def plot_top_5_stocks(top_5_tickers):
    """繪製前 5 名股票的走勢圖（包含股價、成交量、10 日均線和 MACD）。"""
    for ticker in top_5_tickers:
        st.write(f"正在處理股票 {ticker}...")
        stock_data = fetch_stock_data(ticker, trading_days=90)
        
        # 檢查數據是否成功獲取
        if stock_data is None:
            st.error(f"無法繪製 {ticker} 的圖表：fetch_stock_data 返回 None")
            continue
        if stock_data.empty:
            st.error(f"無法繪製 {ticker} 的圖表：數據為空")
            continue
        
        # 檢查數據長度和日期範圍
        st.write(f"{ticker} 數據長度：{len(stock_data)} 筆")
        if not stock_data.empty:
            st.write(f"{ticker} 數據日期範圍：{stock_data.index[0]} 到 {stock_data.index[-1]}")
        
        # 檢查必要欄位
        required_columns = ['Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        if missing_columns:
            st.error(f"無法繪製 {ticker} 的圖表：缺少欄位 {missing_columns}")
            continue
        
        # 處理多層級索引（如果存在）
        if isinstance(stock_data.columns, pd.MultiIndex):
            st.write(f"檢測到多層級索引，嘗試提取 {ticker} 的數據")
            try:
                stock_data = stock_data.xs(ticker, axis=1, level=1)
                st.write(f"提取後的 stock_data 欄位：{stock_data.columns}")
            except KeyError as e:
                st.error(f"無法提取 {ticker} 的數據：{str(e)}")
                continue
        
        # 檢查數據類型
        if not isinstance(stock_data['Close'], pd.Series):
            st.error(f"無法繪製 {ticker} 的圖表：'Close' 欄位不是 Pandas Series，類型為 {type(stock_data['Close'])}")
            continue
        if not isinstance(stock_data['Volume'], pd.Series):
            st.error(f"無法繪製 {ticker} 的圖表：'Volume' 欄位不是 Pandas Series，類型為 {type(stock_data['Volume'])}")
            continue
        
        # 檢查數據長度是否滿足要求
        if len(stock_data) < 10:
            st.error(f"無法繪製 {ticker} 的圖表：數據長度 {len(stock_data)} 小於最小要求 10")
            continue
        
        try:
            # 檢查數據中是否有 NaN 值
            if stock_data['Close'].isna().any() or stock_data['Volume'].isna().any():
                st.warning(f"{ticker} 的數據中包含 NaN 值，將進行清理")
                stock_data = stock_data.dropna()
                st.write(f"清理後 {ticker} 數據長度：{len(stock_data)} 筆")
                if len(stock_data) < 10:
                    st.error(f"清理後 {ticker} 的數據長度 {len(stock_data)} 小於最小要求 10")
                    continue
            
            # 提取日期、股價和成交量，確保日期格式乾淨
            dates = [date.strftime("%Y-%m-%d") for date in stock_data.index]
            prices = stock_data["Close"].astype(float).tolist()
            volumes = stock_data["Volume"].astype(int).tolist()

            # 計算 10 日均線
            stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
            ma10 = stock_data['MA10'].astype(float).tolist()
            if stock_data['MA10'].isna().all():
                st.error(f"無法繪製 {ticker} 的圖表：10 日均線計算結果全為 NaN")
                continue
            st.write(f"10 日均線計算完成，樣本數據：\n{stock_data['MA10'].tail(5)}")

            # 計算 MACD 指標
            ema12 = stock_data["Close"].ewm(span=12, adjust=False).mean()
            ema26 = stock_data["Close"].ewm(span=26, adjust=False).mean()
            macd_line = (ema12 - ema26).tolist()
            macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().tolist()
            macd_histogram = (pd.Series(macd_line) - pd.Series(macd_signal)).tolist()

            # 檢查 MACD 計算結果
            if pd.Series(macd_line).isna().all():
                st.error(f"無法繪製 {ticker} 的圖表：MACD 線計算結果全為 NaN")
                continue
            if pd.Series(macd_signal).isna().all():
                st.error(f"無法繪製 {ticker} 的圖表：MACD 訊號線計算結果全為 NaN")
                continue
            if pd.Series(macd_histogram).isna().all():
                st.error(f"無法繪製 {ticker} 的圖表：MACD 柱狀圖計算結果全為 NaN")
                continue
            st.write(f"MACD 計算完成，樣本數據：")
            st.write(f"MACD 線：\n{pd.Series(macd_line).tail(5)}")
            st.write(f"訊號線：\n{pd.Series(macd_signal).tail(5)}")
            st.write(f"柱狀圖：\n{pd.Series(macd_histogram).tail(5)}")

            # 創建 X 軸的索引（0, 1, 2, ...）
            x_indices = list(range(len(dates)))

            # 創建圖表，包含兩個子圖：上圖顯示股價和均線，下圖顯示 MACD
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f"{ticker} 近 3 個月走勢", "MACD"),
                row_heights=[0.7, 0.3],
                specs=[[{"secondary_y": True}], [{}]]
            )

            # 為股價線準備 customdata，包含日期和 10 日均線
            customdata_price = [[dates[i], ma10[i]] for i in range(len(dates))]

            # 添加股價線，並設置滑鼠懸停時顯示日期、股價和 10 日均線（小數點後兩位）
            fig.add_trace(
                go.Scatter(
                    x=x_indices,
                    y=prices,
                    mode="lines",
                    name="股價",
                    line=dict(color="blue"),
                    hovertemplate="日期: %{customdata[0]}<br>股價: %{y:.2f}<br>10 日均線: %{customdata[1]:.2f}",
                    customdata=customdata_price
                ),
                row=1, col=1, secondary_y=False
            )

            # 添加 10 日均線，並設置滑鼠懸停時顯示日期和 10 日均線（小數點後兩位）
            fig.add_trace(
                go.Scatter(
                    x=x_indices,
                    y=ma10,
                    mode="lines",
                    name="10 日均線",
                    line=dict(color="orange"),
                    hovertemplate="日期: %{customdata}<br>10 日均線: %{y:.2f}",
                    customdata=dates
                ),
                row=1, col=1, secondary_y=False
            )

            # 添加成交量柱狀圖，並設置滑鼠懸停時顯示日期和成交量
            fig.add_trace(
                go.Bar(
                    x=x_indices,
                    y=volumes,
                    name="成交量",
                    opacity=0.3,
                    marker_color="pink",
                    hovertemplate="日期: %{customdata}<br>成交量: %{y:,.0f}",
                    customdata=dates
                ),
                row=1, col=1, secondary_y=True
            )

            # 添加 MACD 線，並設置滑鼠懸停時顯示日期和 MACD 值（小數點後兩位）
            fig.add_trace(
                go.Scatter(
                    x=x_indices,
                    y=macd_line,
                    mode="lines",
                    name="MACD 線",
                    line=dict(color="blue"),
                    hovertemplate="日期: %{customdata}<br>MACD: %{y:.2f}",
                    customdata=dates
                ),
                row=2, col=1
            )

            # 添加訊號線，並設置滑鼠懸停時顯示日期和訊號線值（小數點後兩位）
            fig.add_trace(
                go.Scatter(
                    x=x_indices,
                    y=macd_signal,
                    mode="lines",
                    name="訊號線",
                    line=dict(color="orange"),
                    hovertemplate="日期: %{customdata}<br>訊號線: %{y:.2f}",
                    customdata=dates
                ),
                row=2, col=1
            )

            # 添加 MACD 柱狀圖，並設置滑鼠懸停時顯示日期和柱狀圖值（小數點後兩位）
            fig.add_trace(
                go.Bar(
                    x=x_indices,
                    y=macd_histogram,
                    name="MACD 柱狀圖",
                    marker_color="gray",
                    opacity=0.5,
                    hovertemplate="日期: %{customdata}<br>MACD 柱狀圖: %{y:.2f}",
                    customdata=dates
                ),
                row=2, col=1
            )

            # 設置圖表佈局
            fig.update_layout(
                title=f"{ticker} 近 3 個月走勢與 MACD",
                xaxis_title="日期",
                showlegend=True,
                height=800,
                margin=dict(b=100)
            )

            # 設置 Y 軸標籤
            fig.update_yaxes(title_text="價格", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="成交量", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="MACD", row=2, col=1)

            # 設置 X 軸刻度標籤，適當省略（每 5 個數據點顯示一個日期）
            tickvals = x_indices[::5]
            ticktext = [dates[i] for i in tickvals]

            fig.update_xaxes(
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=45,
                tickfont=dict(size=10),
                row=1, col=1
            )
            fig.update_xaxes(
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=45,
                tickfont=dict(size=10),
                row=2, col=1
            )

            st.plotly_chart(fig)
            st.write(f"成功繪製 {ticker} 的圖表")
        except Exception as e:
            st.error(f"繪製 {ticker} 的圖表時發生錯誤：{str(e)}")

def plot_breakout_stocks(breakout_tickers, consol_days):
    """繪製突破股票的圖表（包含股價、阻力位、支撐位、買入信號、成交量和 MACD）。"""
    for ticker in breakout_tickers:
        st.write(f"正在處理突破股票 {ticker}...")
        stock_data = fetch_stock_data(ticker)
        
        # 檢查數據是否成功獲取
        if stock_data is None:
            st.error(f"無法繪製 {ticker} 的圖表：fetch_stock_data 返回 None")
            continue
        if stock_data.empty:
            st.error(f"無法繪製 {ticker} 的圖表：數據為空")
            continue
        
        # 檢查數據長度和日期範圍
        st.write(f"{ticker} 數據長度：{len(stock_data)} 筆")
        if not stock_data.empty:
            st.write(f"{ticker} 數據日期範圍：{stock_data.index[0]} 到 {stock_data.index[-1]}")
        
        # 檢查必要欄位
        required_columns = ['Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        if missing_columns:
            st.error(f"無法繪製 {ticker} 的圖表：缺少欄位 {missing_columns}")
            continue
        
        # 處理多層級索引（如果存在）
        if isinstance(stock_data.columns, pd.MultiIndex):
            st.write(f"檢測到多層級索引，嘗試提取 {ticker} 的數據")
            try:
                stock_data = stock_data.xs(ticker, axis=1, level=1)
                st.write(f"提取後的 stock_data 欄位：{stock_data.columns}")
            except KeyError as e:
                st.error(f"無法提取 {ticker} 的數據：{str(e)}")
                continue
        
        # 檢查數據類型
        if not isinstance(stock_data['Close'], pd.Series):
            st.error(f"無法繪製 {ticker} 的圖表：'Close' 欄位不是 Pandas Series，類型為 {type(stock_data['Close'])}")
            continue
        if not isinstance(stock_data['Volume'], pd.Series):
            st.error(f"無法繪製 {ticker} 的圖表：'Volume' 欄位不是 Pandas Series，類型為 {type(stock_data['Volume'])}")
            continue
        
        try:
            # 計算最近盤整區間的阻力位和支撐位
            recent_high = stock_data['Close'][-consol_days-1:-1].max()
            recent_low = stock_data['Close'][-consol_days-1:-1].min()
            
            # 檢查數據中是否有 NaN 值
            if stock_data['Close'].isna().any() or stock_data['Volume'].isna().any():
                st.warning(f"{ticker} 的數據中包含 NaN 值，將進行清理")
                stock_data = stock_data.dropna()
                st.write(f"清理後 {ticker} 數據長度：{len(stock_data)} 筆")
            
            # 提取日期、股價和成交量，確保日期格式乾淨
            dates = [date.strftime("%Y-%m-%d") for date in stock_data.index]
            prices = stock_data["Close"].astype(float).tolist()
            volumes = stock_data["Volume"].astype(int).tolist()

            # 計算 MACD 指標
            ema12 = stock_data["Close"].ewm(span=12, adjust=False).mean()
            ema26 = stock_data["Close"].ewm(span=26, adjust=False).mean()
            macd_line = (ema12 - ema26).tolist()
            macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().tolist()
            macd_histogram = (pd.Series(macd_line) - pd.Series(macd_signal)).tolist()

            # 檢查 MACD 計算結果
            if pd.Series(macd_line).isna().all():
                st.error(f"無法繪製 {ticker} 的圖表：MACD 線計算結果全為 NaN")
                continue
            if pd.Series(macd_signal).isna().all():
                st.error(f"無法繪製 {ticker} 的圖表：MACD 訊號線計算結果全為 NaN")
                continue
            if pd.Series(macd_histogram).isna().all():
                st.error(f"無法繪製 {ticker} 的圖表：MACD 柱狀圖計算結果全為 NaN")
                continue

            # 創建 X 軸的索引（0, 1, 2, ...）
            x_indices = list(range(len(dates)))

            # 創建子圖
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f"{ticker} 突破圖表（買入信號）", "MACD"),
                row_heights=[0.7, 0.3],
                specs=[[{"secondary_y": True}], [{}]]
            )

            # 添加股價線，並設置滑鼠懸停時顯示日期和價格（小數點後兩位）
            fig.add_trace(
                go.Scatter(
                    x=x_indices,
                    y=prices,
                    mode="lines",
                    name="價格",
                    line=dict(color="blue"),
                    hovertemplate="日期: %{customdata}<br>價格: %{y:.2f}",
                    customdata=dates
                ),
                row=1, col=1, secondary_y=False
            )

            # 添加阻力位和支撐位
            fig.add_trace(
                go.Scatter(
                    x=[x_indices[0], x_indices[-1]],
                    y=[recent_high, recent_high],
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    name='阻力位',
                    hovertemplate="阻力位: %{y:.2f}"
                ),
                row=1, col=1, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_indices[0], x_indices[-1]],
                    y=[recent_low, recent_low],
                    mode='lines',
                    line=dict(dash='dash', color='green'),
                    name='支撐位',
                    hovertemplate="支撐位: %{y:.2f}"
                ),
                row=1, col=1, secondary_y=False
            )

            # 添加買入信號
            fig.add_trace(
                go.Scatter(
                    x=[x_indices[-1]],
                    y=[prices[-1]],
                    mode='markers',
                    marker=dict(size=12, color='blue', symbol='star'),
                    name='買入信號',
                    hovertemplate="日期: %{customdata}<br>價格: %{y:.2f}",
                    customdata=[dates[-1]]
                ),
                row=1, col=1, secondary_y=False
            )

            # 添加成交量柱狀圖，並設置滑鼠懸停時顯示日期和成交量
            fig.add_trace(
                go.Bar(
                    x=x_indices,
                    y=volumes,
                    name="成交量",
                    opacity=0.3,
                    marker_color="pink",
                    hovertemplate="日期: %{customdata}<br>成交量: %{y:,.0f}",
                    customdata=dates
                ),
                row=1, col=1, secondary_y=True
            )

            # 添加 MACD 線，並設置滑鼠懸停時顯示日期和 MACD 值（小數點後兩位）
            fig.add_trace(
                go.Scatter(
                    x=x_indices,
                    y=macd_line,
                    mode="lines",
                    name="MACD 線",
                    line=dict(color="blue"),
                    hovertemplate="日期: %{customdata}<br>MACD: %{y:.2f}",
                    customdata=dates
                ),
                row=2, col=1
            )

            # 添加訊號線，並設置滑鼠懸停時顯示日期和訊號線值（小數點後兩位）
            fig.add_trace(
                go.Scatter(
                    x=x_indices,
                    y=macd_signal,
                    mode="lines",
                    name="訊號線",
                    line=dict(color="orange"),
                    hovertemplate="日期: %{customdata}<br>訊號線: %{y:.2f}",
                    customdata=dates
                ),
                row=2, col=1
            )

            # 添加 MACD 柱狀圖，並設置滑鼠懸停時顯示日期和柱狀圖值（小數點後兩位）
            fig.add_trace(
                go.Bar(
                    x=x_indices,
                    y=macd_histogram,
                    name="MACD 柱狀圖",
                    marker_color="gray",
                    opacity=0.5,
                    hovertemplate="日期: %{customdata}<br>MACD 柱狀圖: %{y:.2f}",
                    customdata=dates
                ),
                row=2, col=1
            )

            # 設置圖表佈局
            fig.update_layout(
                title=f"{ticker} 突破圖表（買入信號）與 MACD",
                xaxis_title="日期",
                showlegend=True,
                height=800,
                margin=dict(b=100)
            )

            # 設置 Y 軸標籤
            fig.update_yaxes(title_text="價格", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="成交量", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="MACD", row=2, col=1)

            # 設置 X 軸刻度標籤，適當省略（每 5 個數據點顯示一個日期）
            tickvals = x_indices[::5]
            ticktext = [dates[i] for i in tickvals]

            fig.update_xaxes(
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=45,
                tickfont=dict(size=10),
                row=1, col=1
            )
            fig.update_xaxes(
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=45,
                tickfont=dict(size=10),
                row=2, col=1
            )

            st.plotly_chart(fig)
            st.write(f"成功繪製 {ticker} 的圖表")
        except Exception as e:
            st.error(f"繪製 {ticker} 的圖表時發生錯誤：{str(e)}")
