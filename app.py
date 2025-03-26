import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from screening import screen_stocks, fetch_stock_data, get_nasdaq_100, get_sp500, get_nasdaq_all

st.title("Qullamaggie Breakout Screener")

# 策略說明
st.markdown("""
### 策略說明
本程式基於 Qullamaggie Breakout 策略篩選股票，核心邏輯如下：
1. **前段上升**：股票在過去一段時間（前段上升天數）有顯著漲幅（前段漲幅 > 最小漲幅）。
2. **盤整階段**：隨後進入低波動盤整（盤整範圍 < 最大盤整範圍），成交量通常下降。
3. **突破條件**：股票價格突破盤整區間高點，且突破當天成交量顯著放大（> 過去 10 天均量的 1.5 倍）。
4. **買入時機**：突破當天或突破後回踩確認（本程式以突破當天為買入信號）。
""")

# 用戶輸入參數
st.sidebar.header("篩選參數")
index_option = st.sidebar.selectbox("選擇股票池", ["NASDAQ 100", "S&P 500", "NASDAQ All"])
prior_days = st.sidebar.slider("前段上升天數", 10, 30, 20)
consol_days = st.sidebar.slider("盤整天數", 5, 15, 10)
min_rise = st.sidebar.slider("最小漲幅 (%)", 0, 50, 0, help="設為 0 以獲得更多結果")
max_range = st.sidebar.slider("最大盤整範圍 (%)", 3, 15, 10, help="增加此值以放寬整理區間")
min_adr = st.sidebar.slider("最小 ADR (%)", 0, 10, 0, help="設為 0 以納入更多股票")
max_stocks = st.sidebar.slider("最大篩選股票數量", 50, 1000, 100, help="限制股票數量以加快處理速度，僅適用於 NASDAQ All")

# 選擇股票池
if index_option == "NASDAQ 100":
    tickers = get_nasdaq_100()
elif index_option == "S&P 500":
    tickers = get_sp500()
else:
    tickers = get_nasdaq_all()[:max_stocks]

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
            st.success(f"找到 {len(df)} 隻符合條件的股票")

# 顯示結果
if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("篩選結果")
    # 只顯示最近一天的數據
    latest_df = df[df['Date'] == df['Date'].max()]
    # 添加狀態欄位
    latest_df['Status'] = latest_df.apply(
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
    
    # 顯示突破股票的圖表
    breakout_df = latest_df[latest_df['Breakout'] & latest_df['Breakout_Volume']]
    if not breakout_df.empty:
        st.subheader("當前突破股票（可買入）")
        for ticker in breakout_df['Ticker'].unique():
            stock_data = fetch_stock_data(ticker)
            if stock_data is not None:
                recent_high = stock_data['Close'][-consol_days-1:-1].max()
                recent_low = stock_data['Close'][-consol_days-1:-1].min()
                
                fig = go.Figure()
                # 價格線
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='價格'))
                # 阻力位
                fig.add_trace(go.Scatter(x=[stock_data.index[0], stock_data.index[-1]], y=[recent_high, recent_high], 
                                        mode='lines', line=dict(dash='dash', color='red'), name='阻力位'))
                # 支撐位
                fig.add_trace(go.Scatter(x=[stock_data.index[0], stock_data.index[-1]], y=[recent_low, recent_low], 
                                        mode='lines', line=dict(dash='dash', color='green'), name='支撐位'))
                # 突破點（買入信號）
                fig.add_trace(go.Scatter(x=[stock_data.index[-1]], y=[stock_data['Close'][-1]], mode='markers', 
                                        marker=dict(size=12, color='blue', symbol='star'), name='買入信號'))
                # 成交量
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
                st.error(f"無法獲取 {ticker} 的數據")
    else:
        st.info("當前無突破股票（無可買入股票）。可能原因：")
        if latest_df['Breakout'].sum() == 0:
            st.write("- 無股票價格突破盤整區間高點。嘗試增加最大盤整範圍或降低最小漲幅。")
        elif latest_df['Breakout_Volume'].sum() == 0:
            st.write("- 突破股票的成交量不足（需 > 過去 10 天均量的 1.5 倍）。嘗試調整成交量條件。")

st.write(f"篩選範圍：{index_option} ({len(tickers)} 隻股票)")
