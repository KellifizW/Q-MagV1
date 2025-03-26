import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from screening import screen_stocks, fetch_stock_data, get_nasdaq_100, get_sp500, get_nasdaq_all
from datetime import datetime, timedelta

st.title("Qullamaggie Breakout Screener")

# 簡介與參考
st.markdown("""
### 簡介
本程式基於 Qullamaggie Breakout 策略篩選股票，參考方法來自 [Reddit: Trade Like a Professional - Breakout Swing Trading](https://www.reddit.com/r/wallstreetbetsOGs/comments/om7h73/trade_like_a_professional_breakout_swing_trading/)。

#### 程式最終目的：
1. **偵測 Qullamaggie Breakout 特徵**：檢查最近 30 日內是否有股票符合 Qullamaggie 描述的 Breakout 特徵：
   - 前段顯著漲幅（前段漲幅 > 最小漲幅）。
   - 隨後進入低波動盤整（盤整範圍 < 最大盤整範圍），成交量下降。
   - 價格突破盤整區間高點，且成交量放大（> 過去 10 天均量的 1.5 倍）。
2. **識別買入時機並標記信號**：如果股票已到達買入時機（突破當天），在圖表上標記買入信號。

#### 篩選結果說明：
- 篩選結果顯示最近一天的數據，包含股票的當前狀態（例如「已突破且可買入」、「盤整中」）。
- 如果有超過 5 隻非重複股票符合條件，將按前段漲幅 (%) 排序，顯示前 5 名的 3 個月走勢圖（包含股價、成交量、10 日均線及 MACD）。
""")

# 用戶輸入參數
st.sidebar.header("篩選參數")
index_option = st.sidebar.selectbox("選擇股票池", ["NASDAQ 100", "S&P 500", "NASDAQ All"])
prior_days = st.sidebar.slider("前段上升天數", 10, 30, 20)
consol_days = st.sidebar.slider("盤整天數", 5, 15, 10)
min_rise = st.sidebar.slider("最小漲幅 (%)", 0, 50, 0, help="設為 0 以獲得更多結果")
max_range = st.sidebar.slider("最大盤整範圍 (%)", 3, 15, 10, help="增加此值以放寬整理區間")
min_adr = st.sidebar.slider("最小 ADR (%)", 0, 10, 0, help="設為 0 以納入更多股票")
max_stocks = st.sidebar.slider("最大篩選股票數量", 10, 500, 50, help="限制股票數量以加快處理速度，僅適用於 NASDAQ All")

# 每次選擇股票池時更新 tickers
if index_option != st.session_state.get('last_index_option', None):
    if index_option == "NASDAQ 100":
        st.session_state['tickers'] = get_nasdaq_100()
    elif index_option == "S&P 500":
        st.session_state['tickers'] = get_sp500()
    else:
        st.session_state['tickers'] = get_nasdaq_all()[:max_stocks]
    st.session_state['last_index_option'] = index_option
    # 重置篩選結果
    if 'df' in st.session_state:
        del st.session_state['df']

tickers = st.session_state['tickers']

# 篩選股票
if st.button("運行篩選"):
    # 重置篩選結果
    if 'df' in st.session_state:
        del st.session_state['df']
    
    with st.spinner("篩選中..."):
        progress_bar = st.progress(0)
        df = screen_stocks(tickers, prior_days, consol_days, min_rise, max_range, min_adr, progress_bar)
        progress_bar.progress(100)
        if df.empty:
            st.warning("無符合條件的股票。請嘗試以下調整：")
            st.write("- **降低最小漲幅** (目前: {}%)：嘗試設為 0-10%".format(min_rise))
            st.write("- **增加最大盤整範圍** (目前: {}%)：嘗試設為 10-15%".format(max_range))
            st.write("- **降低最小 ADR** (目前: {}%)：嘗試設為 0-2%".format(min_adr))
            st.write("- **擴大股票池**：選擇 NASDAQ All 並增加最大篩選股票數量")
        else:
            st.session_state['df'] = df
            st.success(f"找到 {len(df)} 隻符合條件的股票（{len(df['Ticker'].unique())} 隻非重複股票）")

# 顯示結果
if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("篩選結果")
    # 確保 Date 欄位格式一致
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    # 只顯示最近一天的數據
    latest_date = df['Date'].max()
    latest_df = df[df['Date'] == latest_date].copy()
    # 添加狀態欄位
    latest_df.loc[:, 'Status'] = latest_df.apply(
        lambda row: "已突破且可買入" if row['Breakout'] and row['Breakout_Volume']
        else "已突破但成交量不足" if row['Breakout']
        else "盤整中" if row['Consolidation_Range_%'] < max_range
        else "前段上升", axis=1
    )
    # 重命名欄位以更直觀
    display_df = latest_df.rename(columns={
        'Ticker': '股票代碼',
        'Date': '日期',
        'Price': '價格',
        'Prior_Rise_%': '前段漲幅 (%)',
        'Consolidation_Range_%': '盤整範圍 (%)',
        'ADR_%': '平均日波幅 (%)',
        'Breakout': '是否突破',
        'Breakout_Volume': '突破成交量'
    })
    st.dataframe(display_df[['股票代碼', '日期', '價格', '前段漲幅 (%)', '盤整範圍 (%)', '平均日波幅 (%)', 'Status']])
    
    # 檢查非重複股票數量並顯示前 5 名走勢圖
    unique_tickers = latest_df['Ticker'].unique()
    if len(unique_tickers) > 5:
        st.subheader("前 5 名股票走勢（按前段漲幅排序）")
        # 按前段漲幅排序並取前 5 個非重複股票
        top_5_df = latest_df.groupby('Ticker').agg({'Prior_Rise_%': 'max'}).reset_index()
        top_5_df = top_5_df.sort_values(by='Prior_Rise_%', ascending=False).head(5)
        top_5_tickers = top_5_df['Ticker'].tolist()
        
        for ticker in top_5_tickers:
            st.write(f"正在處理股票 {ticker}...")
            stock_data = fetch_stock_data(ticker, days=90)
            
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
    
    # 顯示突破股票的圖表
    breakout_df = latest_df[latest_df['Breakout'] & latest_df['Breakout_Volume']]
    if not breakout_df.empty:
        st.subheader("當前突破股票（可買入）")
        for ticker in breakout_df['Ticker'].unique():
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
    else:
        st.info("當前無突破股票（無可買入股票）。可能原因：")
        if latest_df['Breakout'].sum() == 0:
            st.write("- 無股票價格突破盤整區間高點。嘗試增加最大盤整範圍或降低最小漲幅。")
        elif latest_df['Breakout_Volume'].sum() == 0:
            st.write("- 突破股票的成交量不足（需 > 過去 10 天均量的 1.5 倍）。嘗試調整成交量條件。")

st.write(f"篩選範圍：{index_option} ({len(tickers)} 隻股票)")
