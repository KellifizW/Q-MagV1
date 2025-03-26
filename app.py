import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from screening import screen_stocks, fetch_stock_data, nasdaq_100

st.title("Qullamaggie Breakout Screener")

# 用戶輸入參數
st.sidebar.header("篩選參數")
prior_days = st.sidebar.slider("前段上升天數", 10, 30, 20)
consol_days = st.sidebar.slider("盤整天數", 5, 15, 10)
min_rise = st.sidebar.slider("最小漲幅 (%)", 20, 50, 30)
max_range = st.sidebar.slider("最大盤整範圍 (%)", 3, 10, 5)
min_adr = st.sidebar.slider("最小 ADR (%)", 3, 10, 5)

# 篩選股票
if st.button("運行篩選"):
    with st.spinner("篩選中..."):
        df = screen_stocks(nasdaq_100, prior_days, consol_days, min_rise, max_range, min_adr)
        st.session_state['df'] = df

# 顯示結果
if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("篩選結果")
    st.dataframe(df[['Ticker', 'Date', 'Price', 'Prior_Rise_%', 'Consolidation_Range_%', 'ADR_%', 'Breakout']])
    
    # 選擇股票繪圖
    breakout_stocks = df[df['Breakout'] & df['Breakout_Volume']]['Ticker'].unique()
    selected_ticker = st.selectbox("選擇股票查看圖表", breakout_stocks)
    
    if selected_ticker:
        stock_data = fetch_stock_data(selected_ticker)
        recent_high = stock_data['Close'][-consol_days-1:-1].max()
        recent_low = stock_data['Close'][-consol_days-1:-1].min()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(stock_data.index, stock_data['Close'], label='Price')
        ax.axhline(recent_high, color='r', linestyle='--', label='Resistance')
        ax.axhline(recent_low, color='g', linestyle='--', label='Support')
        ax.axvline(stock_data.index[-1], color='b', linestyle='-', label='Breakout')
        ax.set_title(f"{selected_ticker} Breakout Chart")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

st.write("預設篩選範圍：NASDAQ 100")
