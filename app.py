import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from screening import screen_stocks, fetch_stock_data, get_nasdaq_100, get_sp500

st.title("Qullamaggie Breakout Screener")

# 用戶輸入參數
st.sidebar.header("篩選參數")
index_option = st.sidebar.selectbox("選擇股票池", ["NASDAQ 100", "S&P 500"])
prior_days = st.sidebar.slider("前段上升天數", 10, 30, 20)
consol_days = st.sidebar.slider("盤整天數", 5, 15, 10)
min_rise = st.sidebar.slider("最小漲幅 (%)", 10, 50, 30, help="降低此值以獲得更多結果")
max_range = st.sidebar.slider("最大盤整範圍 (%)", 3, 15, 5, help="增加此值以放寬整理區間")
min_adr = st.sidebar.slider("最小 ADR (%)", 2, 10, 5, help="降低此值以納入更多股票")

# 選擇股票池
if index_option == "NASDAQ 100":
    tickers = get_nasdaq_100()
else:
    tickers = get_sp500()

# 篩選股票
if st.button("運行篩選"):
    with st.spinner("篩選中..."):
        df = screen_stocks(tickers, prior_days, consol_days, min_rise, max_range, min_adr)
        if df.empty:
            st.warning("無符合條件的股票。請嘗試以下調整：")
            st.write("- **降低最小漲幅** (目前: {}%)：嘗試設為 10-20%".format(min_rise))
            st.write("- **增加最大盤整範圍** (目前: {}%)：嘗試設為 10-15%".format(max_range))
            st.write("- **降低最小 ADR** (目前: {}%)：嘗試設為 2-3%".format(min_adr))
            st.write("- **擴大股票池**：選擇 S&P 500 (約 500 隻股票)")
        else:
            st.session_state['df'] = df
            st.success(f"找到 {len(df)} 隻符合條件的股票")

# 顯示結果
if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("篩選結果")
    st.dataframe(df[['Ticker', 'Date', 'Price', 'Prior_Rise_%', 'Consolidation_Range_%', 'ADR_%', 'Breakout']])
    
    # 顯示突破股票的圖表
    breakout_df = df[df['Breakout'] & df['Breakout_Volume']]
    if not breakout_df.empty:
        st.subheader("當前突破股票")
        for ticker in breakout_df['Ticker'].unique():
            stock_data = fetch_stock_data(ticker)
            if stock_data is not None:
                recent_high = stock_data['Close'][-consol_days-1:-1].max()
                recent_low = stock_data['Close'][-consol_days-1:-1].min()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Price'))
                fig.add_trace(go.Scatter(x=[stock_data.index[0], stock_data.index[-1]], y=[recent_high, recent_high], 
                                        mode='lines', line=dict(dash='dash', color='red'), name='Resistance'))
                fig.add_trace(go.Scatter(x=[stock_data.index[0], stock_data.index[-1]], y=[recent_low, recent_low], 
                                        mode='lines', line=dict(dash='dash', color='green'), name='Support'))
                fig.add_trace(go.Scatter(x=[stock_data.index[-1]], y=[stock_data['Close'][-1]], mode='markers', 
                                        marker=dict(size=10, color='blue'), name='Breakout'))
                fig.update_layout(title=f"{ticker} Breakout Chart", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)
            else:
                st.error(f"無法獲取 {ticker} 的數據")
    else:
        st.info("當前無突破股票")

st.write(f"篩選範圍：{index_option} ({len(tickers)} 隻股票)")
