import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from screening import screen_stocks, fetch_stock_data, get_nasdaq_100, get_sp500, get_nasdaq_all

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
- 如果有超過 5 隻非重複股票符合條件，將按前段漲幅 (%) 排序，顯示前 5 名的 3 個月走勢圖（包含股價、成交量、10 日均線）。
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

# 選擇股票池（避免自動觸發篩選）
if 'tickers' not in st.session_state:
    if index_option == "NASDAQ 100":
        st.session_state['tickers'] = get_nasdaq_100()
    elif index_option == "S&P 500":
        st.session_state['tickers'] = get_sp500()
    else:
        st.session_state['tickers'] = get_nasdaq_all()[:max_stocks]
tickers = st.session_state['tickers']

# 篩選股票
if st.button("運行篩選"):
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
    # 只顯示最近一天的數據
    latest_df = df[df['Date'] == df['Date'].max()].copy()
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
            stock_data = fetch_stock_data(ticker, days=90)
            if stock_data is not None and not stock_data.empty and len(stock_data) >= 10:
                # 計算 10 日均線
                stock_data['MA10'] = stock_data['Close'].rolling(window=10).mean()
                
                fig = go.Figure()
                # 股價走勢
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='股價'))
                # 10 日均線
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA10'], mode='lines', name='10 日均線', line=dict(color='orange')))
                # 成交量
                fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='成交量', yaxis='y2', opacity=0.3))
                
                fig.update_layout(
                    title=f"{ticker} 近 3 個月走勢",
                    xaxis_title="日期",
                    yaxis_title="價格",
                    yaxis2=dict(title="成交量", overlaying='y', side='right'),
                    showlegend=True
                )
                st.plotly_chart(fig)
            else:
                st.error(f"無法繪製 {ticker} 的圖表：數據不足或獲取失敗")
    
    # 顯示突破股票的圖表
    breakout_df = latest_df[latest_df['Breakout'] & latest_df['Breakout_Volume']]
    if not breakout_df.empty:
        st.subheader("當前突破股票（可買入）")
        for ticker in breakout_df['Ticker'].unique():
            stock_data = fetch_stock_data(ticker)
            if stock_data is not None and not stock_data.empty:
                recent_high = stock_data['Close'][-consol_days-1:-1].max()
                recent_low = stock_data['Close'][-consol_days-1:-1].min()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='價格'))
                fig.add_trace(go.Scatter(x=[stock_data.index[0], stock_data.index[-1]], y=[recent_high, recent_high], 
                                        mode='lines', line=dict(dash='dash', color='red'), name='阻力位'))
                fig.add_trace(go.Scatter(x=[stock_data.index[0], stock_data.index[-1]], y=[recent_low, recent_low], 
                                        mode='lines', line=dict(dash='dash', color='green'), name='支撐位'))
                fig.add_trace(go.Scatter(x=[stock_data.index[-1]], y=[stock_data['Close'][-1]], mode='markers', 
                                        marker=dict(size=12, color='blue', symbol='star'), name='買入信號'))
                fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='成交量', yaxis='y2', opacity=0.3))
                
                fig.update_layout(
                    title=f"{ticker} 突破圖表（買入信號）",
                    xaxis_title="日期",
                    yaxis_title="價格",
                    yaxis2=dict(title="成交量", overlaying='y', side='right'),
                    showlegend=True
                )
                st.plotly_chart(fig)
            else:
                st.error(f"無法繪製 {ticker} 的圖表：數據獲取失敗")
    else:
        st.info("當前無突破股票（無可買入股票）。可能原因：")
        if latest_df['Breakout'].sum() == 0:
            st.write("- 無股票價格突破盤整區間高點。嘗試增加最大盤整範圍或降低最小漲幅。")
        elif latest_df['Breakout_Volume'].sum() == 0:
            st.write("- 突破股票的成交量不足（需 > 過去 10 天均量的 1.5 倍）。嘗試調整成交量條件。")

st.write(f"篩選範圍：{index_option} ({len(tickers)} 隻股票)")
