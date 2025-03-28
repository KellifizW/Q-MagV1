import streamlit as st
import pandas as pd
import os
from screening import screen_stocks, get_nasdaq_100, get_sp500, get_nasdaq_all
from visualize import plot_top_5_stocks, plot_breakout_stocks
from database import update_database  # 假設 database.py 已更新
from git_utils import GitRepoManager  # 更新後的模組
from file_utils import diagnose_db_file  # 從之前模組導入
from datetime import datetime, timedelta
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定義
DB_PATH = "stocks.db"
TICKERS_CSV = "Tickers.csv"
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"

st.title("Qullamaggie Breakout Screener")

st.markdown("""
### 簡介
本程式基於 Qullamaggie Breakout 策略篩選股票，參考方法來自 [Reddit: Trade Like a Professional - Breakout Swing Trading](https://www.reddit.com/r/wallstreetbetsOGs/comments/om7h73/trade_like_a_professional_breakout_swing_trading/)。

#### 程式最終目的：
1. **偵測 Qullamaggie Breakout 特徵**：檢查最近 30 日內是否有股票符合 Qullamaggie 描述的 Breakout 特徵：
   - 前段顯著漲幅（22 日內漲幅 > 22 日內最小漲幅，67 日內漲幅 > 67 日內最小漲幅, 126 日內漲幅 > 126 日內最小漲幅）。
   - 隨後進入低波動盤整（盤整範圍 < 最大盤整範圍），成交量下降。
   - 價格突破盤整區間高點，且成交量放大（> 過去 10 天均量的 1.5 倍）。
2. **識別買入時機並標記信號**：如果股票已到達買入時機（突破當天），在圖表上標記買入信號。

#### 篩選結果說明：
- 篩選結果顯示最近一天的數據，包含股票的當前狀態（例如「已突破且可買入」、「盤整中」）。
- 顯示所有符合條件的股票（最多 5 隻），按 22 日內漲幅排序，繪製 3 個月走勢圖（包含股價、成交量、10 日均線及 MACD）。
""")

# 初始化 Git 倉庫（僅首次執行）
if 'repo_initialized' not in st.session_state:
    try:
        repo_manager = GitRepoManager(".", REPO_URL, st.secrets.get("TOKEN", ""))
        st.session_state['repo_initialized'] = True
        st.session_state['repo_manager'] = repo_manager
    except Exception as e:
        st.error(f"Git 倉庫初始化失敗：{str(e)}")
        st.session_state['repo_initialized'] = False

# 初始化資料庫（檢查檔案是否存在並診斷）
def init_database():
    if not os.path.exists(DB_PATH):
        st.error("資料庫 stocks.db 不存在，請點擊「初始化並更新資料庫」")
    diagnostics = diagnose_db_file(DB_PATH)
    if any("錯誤" in diag for diag in diagnostics):
        st.warning("資料庫存在問題，請檢查以下診斷資訊並考慮重建：")
        for line in diagnostics:
            st.write(line)
        if st.button("重建資料庫"):
            os.remove(DB_PATH)
            with open(DB_PATH, "wb") as f:  # 創建空檔案，後續由 update_database 填入結構
                pass
            repo_manager = st.session_state.get('repo_manager')
            if repo_manager:
                repo_manager.track_lfs(DB_PATH)
            st.success("資料庫已重建，請更新資料庫以填充數據")

init_database()

# 提供檢查比例選擇
check_percentage = st.slider("檢查和更新比例 (%)", 0, 100, 10, help="選擇要檢查和更新的股票比例（從末尾開始）") / 100.0

# 更新和初始化按鈕
if st.button("初始化並更新資料庫", key="init_and_update"):
    repo_manager = GitRepoManager(".", REPO_URL, st.secrets.get("TOKEN", ""))
    st.session_state['repo_manager'] = repo_manager
    st.session_state['repo_initialized'] = True
    diagnostics = diagnose_db_file(DB_PATH)
    st.write("資料庫診斷資訊：")
    for line in diagnostics:
        st.write(line)
    update_success = update_database(
        tickers_file=TICKERS_CSV,
        db_path=DB_PATH,
        repo_manager=repo_manager,
        check_percentage=check_percentage
    )
    if not update_success:
        st.error("資料庫初始化與更新失敗，請檢查日誌或網絡連線")

if st.button("更新資料庫", key="update_db"):
    if st.session_state.get('repo_initialized', False):
        diagnostics = diagnose_db_file(DB_PATH)
        st.write("資料庫診斷資訊：")
        for line in diagnostics:
            st.write(line)
        update_success = update_database(
            tickers_file=TICKERS_CSV,
            db_path=DB_PATH,
            repo_manager=st.session_state['repo_manager'],
            check_percentage=check_percentage
        )
        if not update_success:
            st.error("資料庫更新失敗，請檢查日誌或網絡連線")
    else:
        st.error("Git 倉庫未初始化，請先點擊「初始化並更新資料庫」")

# 用戶輸入參數（使用 st.form）
with st.sidebar.form(key="screening_form"):
    st.header("篩選參數")
    index_option = st.selectbox("選擇股票池", ["NASDAQ 100", "S&P 500", "NASDAQ All"])
    prior_days = st.slider("前段上升天數", 10, 30, 20)
    consol_days = st.slider("盤整天數", 5, 15, 10, help="盤整天數是指股票在突破前低波動盤整的天數")
    min_rise_22 = st.slider("22 日內最小漲幅 (%)", 0, 50, 10)
    min_rise_67 = st.slider("67 日內最小漲幅 (%)", 0, 100, 40)
    min_rise_126 = st.slider("126 日內最小漲幅 (%)", 0, 300, 80)
    max_range = st.slider("最大盤整範圍 (%)", 3, 15, 10)
    min_adr = st.slider("最小 ADR (%)", 0, 10, 2)
    max_stocks = st.slider("最大篩選股票數量(測試用)", 10, 1000, 50)
    submit_button = st.form_submit_button("運行篩選")

# 重置按鈕
if st.sidebar.button("重置篩選", key="reset_screening"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# 處理股票池選擇和篩選
if submit_button:
    if 'df' in st.session_state:
        del st.session_state['df']

    try:
        tickers_df = pd.read_csv(TICKERS_CSV)
        csv_tickers = tickers_df['Ticker'].tolist()
    except Exception as e:
        st.error(f"無法讀取 {TICKERS_CSV}: {str(e)}")
        csv_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA']

    if index_option == "NASDAQ 100":
        tickers = get_nasdaq_100(csv_tickers)
    elif index_option == "S&P 500":
        tickers = get_sp500()
    else:
        tickers = get_nasdaq_all(csv_tickers)[:max_stocks]
    st.session_state['tickers'] = tickers

    if not os.path.exists(DB_PATH):
        st.error("資料庫 stocks.db 不存在，請先初始化或更新資料庫")
    else:
        with st.spinner("篩選中..."):
            progress_bar = st.progress(0)
            df = screen_stocks(
                tickers=tickers,
                stock_pool=index_option,
                prior_days=prior_days,
                consol_days=consol_days,
                min_rise_22=min_rise_22,
                min_rise_67=min_rise_67,
                min_rise_126=min_rise_126,
                max_range=max_range,
                min_adr=min_adr,
                progress_bar=progress_bar
            )
            progress_bar.progress(1.0)
            if df.empty:
                st.warning("無符合條件的股票，請調整以下參數：")
                st.write(f"- 降低 22 日內最小漲幅（當前: {min_rise_22}%）")
                st.write(f"- 降低 67 日內最小漲幅（當前: {min_rise_67}%）")
                st.write(f"- 降低 126 日內最小漲幅（當前: {min_rise_126}%）")
                st.write(f"- 增加最大盤整範圍（當前: {max_range}%）")
                st.write(f"- 降低最小 ADR（當前: {min_adr}%）")
            else:
                st.session_state['df'] = df
                st.success(f"找到 {len(df)} 筆符合條件的記錄（{len(df['Ticker'].unique())} 隻股票）")

# 顯示篩選結果
if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("篩選結果")
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    latest_date = df['Date'].max()
    latest_df = df[df['Date'] == latest_date].copy()
    latest_df['Status'] = latest_df.apply(
        lambda row: "已突破且可買入" if row['Breakout'] and row['Breakout_Volume']
        else "已突破但成交量不足" if row['Breakout']
        else "盤整中" if row['Consolidation_Range_%'] < max_range
        else "前段上升", axis=1
    )

    display_df = latest_df.rename(columns={
        'Ticker': '股票代碼', 'Date': '日期', 'Price': '價格',
        'Prior_Rise_22_%': '22 日內漲幅 (%)', 'Prior_Rise_67_%': '67 日內漲幅 (%)',
        'Prior_Rise_126_%': '126 日內漲幅 (%)', 'Consolidation_Range_%': '盤整範圍 (%)',
        'ADR_%': '平均日波幅 (%)', 'Breakout': '是否突破', 'Breakout_Volume': '突破成交量'
    })

    desired_columns = ['股票代碼', '日期', '價格', '22 日內漲幅 (%)', '67 日內漲幅 (%)', '126 日內漲幅 (%)', '盤整範圍 (%)', '平均日波幅 (%)', 'Status']
    available_columns = [col for col in desired_columns if col in display_df.columns]
    if available_columns:
        st.dataframe(display_df[available_columns])
    else:
        st.error("無可顯示的欄位，請檢查篩選條件或數據來源")

    unique_tickers = latest_df['Ticker'].unique()
    if len(unique_tickers) > 0:
        st.subheader("符合條件的股票走勢（按 22 日內漲幅排序）")
        if 'Prior_Rise_22_%' in latest_df.columns:
            top_df = latest_df.groupby('Ticker').agg({'Prior_Rise_22_%': 'max'}).reset_index()
            top_df = top_df.sort_values(by='Prior_Rise_22_%', ascending=False)
            top_tickers = top_df['Ticker'].head(min(len(unique_tickers), 5)).tolist()
            plot_top_5_stocks(top_tickers)
        else:
            st.warning("無法繪製圖表：缺少 'Prior_Rise_22_%' 欄位")

    breakout_df = latest_df[latest_df['Breakout'] & latest_df['Breakout_Volume']]
    if not breakout_df.empty:
        st.subheader("當前突破股票（可買入）")
        plot_breakout_stocks(breakout_df['Ticker'].unique(), consol_days)
    else:
        st.info("當前無突破股票")

# 顯示篩選範圍
tickers = st.session_state.get('tickers', [])
st.write(f"篩選範圍：{index_option} ({len(tickers)} 隻股票)")