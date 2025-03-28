import streamlit as st
import pandas as pd
from screening import screen_stocks, fetch_stock_data, get_nasdaq_100, get_sp500, get_nasdaq_all
from visualize import plot_top_5_stocks, plot_breakout_stocks
from database import init_repo, init_database, update_database  # 匯入所有必要函數
from datetime import datetime, timedelta

st.title("Qullamaggie Breakout Screener")

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

# 初始化 Git 倉庫和資料庫（僅首次執行）
if 'repo_initialized' not in st.session_state:
    repo = init_repo()
    st.session_state['repo_initialized'] = repo is not None
    st.session_state['repo'] = repo

init_database()

# 提供檢查比例選擇
check_percentage = st.slider("檢查和更新比例 (%)", 0, 100, 10, help="選擇要檢查和更新的股票比例（從末尾開始）") / 100.0

# 更新和初始化按鈕
if st.button("初始化並更新資料庫", key="init_and_update"):
    repo = init_repo()
    if repo:
        st.session_state['repo_initialized'] = True
        st.session_state['repo'] = repo
        update_database(repo=repo, check_percentage=check_percentage)
    else:
        st.error("Git 倉庫初始化失敗，無法更新資料庫")

if st.button("更新資料庫", key="update_db"):
    if st.session_state.get('repo_initialized', False):
        update_database(repo=st.session_state['repo'], check_percentage=check_percentage)
    else:
        st.error("Git 倉庫未初始化，無法更新資料庫")

# 用戶輸入參數（使用 st.form）
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
if st.sidebar.button("重置篩選", key="reset_screening"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# 處理股票池選擇和篩選
if submit_button:
    # 重置篩選結果
    if 'df' in st.session_state:
        del st.session_state['df']

    # 讀取 Tickers.csv 作為基礎清單
    try:
        tickers_df = pd.read_csv("Tickers.csv")
        csv_tickers = tickers_df['Ticker'].tolist()
    except Exception as e:
        st.error(f"無法讀取 Tickers.csv: {str(e)}")
        csv_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA']  # 備用清單

    # 更新 tickers
    if index_option == "NASDAQ 100":
        tickers = get_nasdaq_100(csv_tickers)
    elif index_option == "S&P 500":
        tickers = get_sp500(csv_tickers)
    else:
        tickers = get_nasdaq_all(csv_tickers="Tickers.csv")[:max_stocks]
    st.session_state['tickers'] = tickers

    # 檢查資料庫是否存在
    if not os.path.exists("stocks.db"):
        st.error("資料庫 stocks.db 不存在，請先點擊「初始化並更新資料庫」或「更新資料庫」")
    else:
        # 篩選邏輯
        with st.spinner("篩選中..."):
            progress_bar = st.progress(0)
            df = screen_stocks(tickers, prior_days, consol_days, min_rise_22, min_rise_67, max_range, min_adr, progress_bar)
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
    # 檢查 latest_df 的欄位
    st.write("篩選結果的欄位：", latest_df.columns.tolist())
    
    # 重命名欄位以更直觀
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
    
    # 定義要顯示的欄位
    desired_columns = ['股票代碼', '日期', '價格', '22 日內漲幅 (%)', '67 日內漲幅 (%)', '盤整範圍 (%)', '平均日波幅 (%)', 'Status']
    # 檢查哪些欄位存在
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
    
    # 繪製符合條件的股票走勢圖
    unique_tickers = latest_df['Ticker'].unique()
    if len(unique_tickers) > 0:  # 只要有符合條件的股票就繪製圖表
        st.subheader("符合條件的股票走勢（按 22 日內漲幅排序）")
        # 按 22 日內漲幅排序
        if 'Prior_Rise_22_%' in latest_df.columns:
            top_df = latest_df.groupby('Ticker').agg({'Prior_Rise_22_%': 'max'}).reset_index()
            top_df = top_df.sort_values(by='Prior_Rise_22_%', ascending=False)
            # 如果股票數量大於 5，則只取前 5 隻；否則取所有股票
            num_to_display = min(len(unique_tickers), 5)
            top_tickers = top_df['Ticker'].head(num_to_display).tolist()
            plot_top_5_stocks(top_tickers)
        else:
            st.warning("無法繪製圖表：缺少 'Prior_Rise_22_%' 欄位，無法排序股票。")
    
    # 繪製突破股票圖表
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
