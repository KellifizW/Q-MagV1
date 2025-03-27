import streamlit as st
import pandas as pd
from screening import screen_stocks, get_nasdaq_100, get_sp500, get_nasdaq_all
from visualize import plot_top_5_stocks, plot_breakout_stocks
from database import init_repo, push_to_github, initialize_database, update_database
import os
from datetime import datetime

st.title("Qullamaggie Breakout Screener")

# 配置
REPO_DIR = "."  # Streamlit Cloud 中，檔案位於應用根目錄
DB_PATH = os.path.join(REPO_DIR, "stocks.db")  # 即 ./stocks.db

# 簡介與參考
st.markdown("""
### 簡介
本程式基於 Qullamaggie Breakout 策略篩選股票，參考方法來自 [Reddit: Trade Like a Professional - Breakout Swing Trading](https://www.reddit.com/r/wallstreetbetsOGs/comments/om7h73/trade_like_a_professional_breakout_swing_trading/)。

#### 程式最終目的：
1. **偵測 Qullamaggie Breakout 特徵**：檢查最近 30 日內是否有股票符合 Qullamaggie 描述的 Breakout 特徵：
   - 前段顯著漲幅（22 日內漲幅 > 22 日內最小漲幅，67 日內漲幅 > 67 日內最小漲幅）。
   - 隨後進入低波動盤整（盤整範圍 < 最大盤整範圍），成交量下降。
   - 價格突破盤整區間高點，且成交量放大（> 過去 10 天均量的 1.5 倍）。
2. **識別買入時機並標記信號**：如果股票已到達買入時機（突破當天），在圖表上標記買入信號。

#### 篩選結果說明：
- 篩選結果顯示最近一天的數據，包含股票的當前狀態（例如「已突破且可買入」、「盤整中」）。
- 顯示所有符合條件的股票（最多 5 隻），按 22 日內漲幅排序，繪製 3 個月走勢圖（包含股價、成交量、10 日均線及 MACD）。

#### 補充說明：
- **原作者 Qullamaggie 使用的參數**（根據 Reddit 文章）：
  - 前段上升天數：20 天
  - 盤整天數：10 天
  - 22 日內最小漲幅：10%
  - 67 日內最小漲幅：40%
  - 最大盤整範圍：10%
  - 最小 ADR：2%
""")

# 初始化現有的 Git 倉庫
repo = init_repo()
if repo is None:
    st.error("無法初始化 GitHub 倉庫，請檢查 TOKEN 配置或倉庫設置")
    st.stop()

try:
    tickers_df = pd.read_csv("tickers.csv")  # 直接讀取根目錄下的 tickers.csv
    csv_tickers = tickers_df['Ticker'].tolist()
except FileNotFoundError:
    st.error("找不到 tickers.csv，請確保該檔案已上傳至 GitHub 倉庫根目錄")
    st.stop()

if not os.path.exists(DB_PATH):
    st.write("初始化資料庫...")
    initialize_database(csv_tickers, repo=repo)
    push_to_github(repo, "Initial stocks.db creation")
else:
    st.write("更新資料庫...")
    if update_database(csv_tickers, repo=repo):
        push_to_github(repo, f"Daily update: {datetime.now().strftime('%Y-%m-%d')}")

# 用戶輸入參數
with st.sidebar.form(key="screening_form"):
    st.header("篩選參數")
    index_option = st.selectbox("選擇股票池", ["NASDAQ 100", "S&P 500", "NASDAQ All"])
    prior_days = st.slider("前段上升天數", 10, 30, 20)
    consol_days = st.slider(
        "盤整天數", 5, 15, 10,
        help="盤整天數是指股票在突破前低波動盤整的天數。計算方式：從最近一天向前回溯指定天數，檢查這段期間的價格波動範圍是否小於最大盤整範圍。"
    )
    min_rise_22 = st.slider("22 日內最小漲幅 (%)", 0, 50, 10, help="股票在過去 22 日內的最小漲幅要求")
    min_rise_67 = st.slider("67 日內最小漲幅 (%)", 0, 100, 40, help="股票在過去 67 日內的最小漲幅要求")
    max_range = st.slider("最大盤整範圍 (%)", 3, 15, 10, help="增加此值以放寬整理區間")
    min_adr = st.slider("最小 ADR (%)", 0, 10, 2, help="設為 0 以納入更多股票")
    max_stocks = st.slider("最大篩選股票數量", 10, 500, 50, help="限制股票數量以加快處理速度，僅適用於 NASDAQ All")
    submit_button = st.form_submit_button("運行篩選")

# 重置按鈕
if st.sidebar.button("重置篩選"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# 篩選邏輯
if submit_button:
    if 'df' in st.session_state:
        del st.session_state['df']
    
    if index_option == "NASDAQ 100":
        tickers = get_nasdaq_100(csv_tickers)
    elif index_option == "S&P 500":
        tickers = get_sp500()
    else:
        tickers = get_nasdaq_all(csv_tickers)[:max_stocks]
    st.session_state['tickers'] = tickers
    
    with st.spinner("篩選中..."):
        progress_bar = st.progress(0)
        df = screen_stocks(tickers, index_option, prior_days, consol_days, min_rise_22, min_rise_67, max_range, min_adr, progress_bar)
        progress_bar.progress(100)
        if 'stock_data' in st.session_state:
            st.write(f"批量數據已載入，涵蓋 {len(st.session_state['stock_data'].columns.get_level_values(1))} 檔股票")
        if df.empty:
            st.warning("無符合條件的股票。請嘗試以下調整：")
            st.write("- **降低 22 日內最小漲幅** (目前: {}%)：嘗試設為 0-10%".format(min_rise_22))
            st.write("- **降低 67 日內最小漲幅** (目前: {}%)：嘗試設為 20-40%".format(min_rise_67))
            st.write("- **增加最大盤整範圍** (目前: {}%)：嘗試設為 10-15%".format(max_range))
            st.write("- **降低最小 ADR** (目前: {}%)：嘗試設為 0-2%".format(min_adr))
            st.write("- **擴大股票池**：選擇 NASDAQ All 並增加最大篩選股票數量")
        else:
            st.session_state['df'] = df
            st.success(f"找到 {len(df)} 隻符合條件的股票（{len(df['Ticker'].unique())} 隻非重複股票）")

# 顯示篩選結果
if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("篩選結果")
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    latest_date = df['Date'].max()
    latest_df = df[df['Date'] == latest_date].copy()
    latest_df.loc[:, 'Status'] = latest_df.apply(
        lambda row: "已突破且可買入" if row['Breakout'] and row['Breakout_Volume']
        else "已突破但成交量不足" if row['Breakout']
        else "盤整中" if row['Consolidation_Range_%'] < max_range
        else "前段上升", axis=1
    )
    st.write("篩選結果的欄位：", latest_df.columns.tolist())
    
    display_df = latest_df.rename(columns={
        'Ticker': '股票代碼',
        'Date': '日期',
        'Price': '價格',
        'Prior_Rise_22_%': '22 日內漲幅 (%)',
        'Prior_Rise_67_%': '67 日內漲幅 (%)',
        'Consolidation_Range_%': '盤整範圍 (%)',
        'ADR_%': '平均日波幅 (%)',
        'Breakout': '是否突破',
        'Breakout_Volume': '突破成交量'
    })
    
    desired_columns = ['股票代碼', '日期', '價格', '22 日內漲幅 (%)', '67 日內漲幅 (%)', '盤整範圍 (%)', '平均日波幅 (%)', 'Status']
    available_columns = [col for col in desired_columns if col in display_df.columns]
    missing_columns = [col for col in desired_columns if col not in display_df.columns]
    
    if missing_columns:
        st.warning(f"以下欄位在篩選結果中缺失：{missing_columns}")
        st.write("可能原因：")
        st.write("- 篩選條件過嚴（例如 22 日內最小漲幅或 67 日內最小漲幅過高），導致無股票符合條件。")
        st.write("- 數據下載失敗，部分股票數據缺失。")
        st.write("建議：")
        st.write("- 降低 22 日內最小漲幅或 67 日內最小漲幅。")
        st.write("- 檢查網絡連線，確保數據下載正常。")
    
    if available_columns:
        st.dataframe(display_df[available_columns])
    else:
        st.error("無可顯示的欄位，請檢查篩選條件或數據來源。")
    
    unique_tickers = latest_df['Ticker'].unique()
    if len(unique_tickers) > 0:
        st.subheader("符合條件的股票走勢（按 22 日內漲幅排序）")
        if 'Prior_Rise_22_%' in latest_df.columns:
            top_df = latest_df.groupby('Ticker').agg({'Prior_Rise_22_%': 'max'}).reset_index()
            top_df = top_df.sort_values(by='Prior_Rise_22_%', ascending=False)
            num_to_display = min(len(unique_tickers), 5)
            top_tickers = top_df['Ticker'].head(num_to_display).tolist()
            plot_top_5_stocks(top_tickers)
        else:
            st.warning("無法繪製圖表：缺少 'Prior_Rise_22_%' 欄位，無法排序股票。")
    
    breakout_df = latest_df[latest_df['Breakout'] & latest_df['Breakout_Volume']]
    if not breakout_df.empty:
        st.subheader("當前突破股票（可買入）")
        breakout_tickers = breakout_df['Ticker'].unique()
        plot_breakout_stocks(breakout_tickers, consol_days)
    else:
        st.info("當前無突破股票（無可買入股票）。可能原因：")
        if latest_df['Breakout'].sum() == 0:
            st.write("- 無股票價格突破盤整區間高點。嘗試增加最大盤整範圍或降低 22 日/67 日內最小漲幅。")
        elif latest_df['Breakout_Volume'].sum() == 0:
            st.write("- 突破股票的成交量不足（需 > 過去 10 天均量的 1.5 倍）。嘗試調整成交量條件。")

# 顯示篩選範圍
tickers = st.session_state.get('tickers', [])
st.write(f"篩選範圍：{index_option} ({len(tickers)} 隻股票)")
