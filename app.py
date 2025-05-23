import streamlit as st
import pandas as pd
import os
from screening import screen_stocks, get_nasdaq_100, get_sp500, get_nasdaq_all, screen_single_stock
from visualize import plot_top_5_stocks, plot_breakout_stocks
from database import update_database
from git_utils import GitRepoManager
from file_utils import diagnose_db_file
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "stocks.db"
TICKERS_CSV = "Tickers.csv"
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"

st.title("Qullamaggie Breakout Screener")

st.markdown("""
    <style>
    .main .block-container {
        max-width: 130%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
### 簡介
本程式基於 Qullamaggie Breakout 策略篩選股票，參考方法來自 [Reddit: Trade Like a Professional - Breakout Swing Trading](https://www.reddit.com/r/wallstreetbetsOGs/comments/om7h73/trade_like_a_professional_breakout_swing_trading/)。

#### 程式最終目的：
1. **偵測 Qullamaggie Breakout 特徵**：檢查最近 30 日內是否有股票符合 Qullamaggie 描述的 Breakout 特徵：
   - **顯著的前段漲幅**：股票在過去某段時間內展現強勁上漲。
   - **低波動盤整期**：隨後進入低波動盤整階段，價格波動範圍縮小，成交量下降。
   - **突破關鍵阻力**：價格突破盤整區間高點，伴隨成交量顯著放大。
2. **識別買入時機並標記信號**：如果股票已到達買入時機（突破當天），在圖表上標記買入信號。
""")

if 'repo_initialized' not in st.session_state:
    try:
        repo_manager = GitRepoManager(".", REPO_URL, st.secrets.get("TOKEN", ""))
        st.session_state['repo_initialized'] = True
        st.session_state['repo_manager'] = repo_manager
    except Exception as e:
        st.error(f"Git 倉庫初始化失敗：{str(e)}")
        st.session_state['repo_initialized'] = False

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
            with open(DB_PATH, "wb") as f:
                pass
            repo_manager = st.session_state.get('repo_manager')
            if repo_manager:
                repo_manager.track_lfs(DB_PATH)
            st.success("資料庫已重建，請更新資料庫以填充數據")

init_database()

check_percentage = st.slider("檢查和更新比例 (%)", 0, 100, 10, help="選擇要檢查和更新的股票比例") / 100.0

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

# 初始化篩選參數的狀態
if 'screening_params' not in st.session_state:
    st.session_state['screening_params'] = {
        'index_option': "NASDAQ 100",
        'use_prior_days': True,
        'prior_days': 20,
        'use_consol_days': True,
        'consol_days': 10,
        'use_min_rise_22': True,
        'min_rise_22': 10,
        'use_min_rise_67': True,
        'min_rise_67': 40,
        'use_min_rise_126': True,
        'min_rise_126': 80,
        'use_max_range': True,
        'max_range': 10,
        'use_min_adr': True,
        'min_adr': 2,
        'use_candle_strength': True,
        'use_volume_filter': False,
        'min_volume': 1000,
        'use_intraday_rise': False,
        'min_intraday_rise': 10,
        'max_stock_percentage': 50.0
    }

# 篩選參數（直接使用 st.sidebar）
st.sidebar.header("篩選參數")
index_option = st.sidebar.selectbox("選擇股票池", ["NASDAQ 100", "S&P 500", "NASDAQ All"], 
                                    index=["NASDAQ 100", "S&P 500", "NASDAQ All"].index(st.session_state['screening_params']['index_option']))

# 前段上升天數
use_prior_days = st.sidebar.checkbox("啟用前段上升天數篩選", value=st.session_state['screening_params']['use_prior_days'], 
                                     key="use_prior_days_checkbox")
prior_days = st.sidebar.slider("前段上升天數", 10, 30, st.session_state['screening_params']['prior_days'], 
                               disabled=not use_prior_days)

# 盤整天數
use_consol_days = st.sidebar.checkbox("啟用盤整天數篩選", value=st.session_state['screening_params']['use_consol_days'], 
                                      key="use_consol_days_checkbox")
consol_days = st.sidebar.slider("盤整天數", 5, 15, st.session_state['screening_params']['consol_days'], 
                                disabled=not use_consol_days)

# 22 日內最小漲幅
use_min_rise_22 = st.sidebar.checkbox("啟用 22 日內最小漲幅篩選", value=st.session_state['screening_params']['use_min_rise_22'], 
                                      key="use_min_rise_22_checkbox")
min_rise_22 = st.sidebar.slider("22 日內最小漲幅 (%)", 0, 50, st.session_state['screening_params']['min_rise_22'], 
                                disabled=not use_min_rise_22)

# 67 日內最小漲幅
use_min_rise_67 = st.sidebar.checkbox("啟用 67 日內最小漲幅篩選", value=st.session_state['screening_params']['use_min_rise_67'], 
                                      key="use_min_rise_67_checkbox")
min_rise_67 = st.sidebar.slider("67 日內最小漲幅 (%)", 0, 100, st.session_state['screening_params']['min_rise_67'], 
                                disabled=not use_min_rise_67)

# 126 日內最小漲幅
use_min_rise_126 = st.sidebar.checkbox("啟用 126 日內最小漲幅篩選", value=st.session_state['screening_params']['use_min_rise_126'], 
                                       key="use_min_rise_126_checkbox")
min_rise_126 = st.sidebar.slider("126 日內最小漲幅 (%)", 0, 300, st.session_state['screening_params']['min_rise_126'], 
                                 disabled=not use_min_rise_126)

# 最大盤整範圍
use_max_range = st.sidebar.checkbox("啟用最大盤整範圍篩選", value=st.session_state['screening_params']['use_max_range'], 
                                    key="use_max_range_checkbox")
max_range = st.sidebar.slider("最大盤整範圍 (%)", 3, 15, st.session_state['screening_params']['max_range'], 
                              disabled=not use_max_range)

# 最小 ADR
use_min_adr = st.sidebar.checkbox("啟用最小 ADR 篩選", value=st.session_state['screening_params']['use_min_adr'], 
                                  key="use_min_adr_checkbox")
min_adr = st.sidebar.slider("最小 ADR (%)", 0, 10, st.session_state['screening_params']['min_adr'], 
                            disabled=not use_min_adr)

# K線強度篩選
use_candle_strength = st.sidebar.checkbox("啟用K線強度篩選", value=st.session_state['screening_params']['use_candle_strength'], 
                                          help="勾選以要求突破K線收盤價靠近高點 (>70%)")

# 成交量篩選
use_volume_filter = st.sidebar.checkbox("啟用成交量篩選", value=st.session_state['screening_params']['use_volume_filter'], 
                                        key="use_volume_filter_checkbox")
min_volume = st.sidebar.slider("最近10日最小成交量（萬股）", 1000, 9000, st.session_state['screening_params']['min_volume'], 
                               disabled=not use_volume_filter)

# 日內價格上漲篩選
use_intraday_rise = st.sidebar.checkbox("啟用最近5日內日內上漲篩選", value=st.session_state['screening_params']['use_intraday_rise'], 
                                        key="use_intraday_rise_checkbox")
min_intraday_rise = st.sidebar.slider("日內最小上漲幅度 (%)", 10, 100, st.session_state['screening_params']['min_intraday_rise'], 
                                      disabled=not use_intraday_rise)

# 篩選股票比例（始終啟用）
max_stock_percentage = st.sidebar.slider("篩選股票比例 (%)", 10, 100, int(st.session_state['screening_params']['max_stock_percentage'])) / 100.0

# 更新 st.session_state 中的篩選參數
st.session_state['screening_params'].update({
    'index_option': index_option,
    'use_prior_days': use_prior_days,
    'prior_days': prior_days,
    'use_consol_days': use_consol_days,
    'consol_days': consol_days,
    'use_min_rise_22': use_min_rise_22,
    'min_rise_22': min_rise_22,
    'use_min_rise_67': use_min_rise_67,
    'min_rise_67': min_rise_67,
    'use_min_rise_126': use_min_rise_126,
    'min_rise_126': min_rise_126,
    'use_max_range': use_max_range,
    'max_range': max_range,
    'use_min_adr': use_min_adr,
    'min_adr': min_adr,
    'use_candle_strength': use_candle_strength,
    'use_volume_filter': use_volume_filter,
    'min_volume': min_volume,
    'use_intraday_rise': use_intraday_rise,
    'min_intraday_rise': min_intraday_rise,
    'max_stock_percentage': max_stock_percentage * 100.0
})

# 調試資訊：顯示勾選框和滑桿的狀態
st.sidebar.write("### 調試資訊")
st.sidebar.write(f"前段上升天數篩選啟用: {use_prior_days}")
st.sidebar.write(f"盤整天數篩選啟用: {use_consol_days}")
st.sidebar.write(f"22 日漲幅篩選啟用: {use_min_rise_22}")
st.sidebar.write(f"67 日漲幅篩選啟用: {use_min_rise_67}")
st.sidebar.write(f"126 日漲幅篩選啟用: {use_min_rise_126}")
st.sidebar.write(f"盤整範圍篩選啟用: {use_max_range}")
st.sidebar.write(f"ADR 篩選啟用: {use_min_adr}")
st.sidebar.write(f"成交量篩選啟用: {use_volume_filter}")
st.sidebar.write(f"日內上漲篩選啟用: {use_intraday_rise}")

# 運行篩選按鈕
submit_button = st.sidebar.button("運行篩選", key="submit_screening")

# 重置篩選按鈕
if st.sidebar.button("重置篩選", key="reset_screening"):
    st.session_state['screening_params'] = {
        'index_option': "NASDAQ 100",
        'use_prior_days': True,
        'prior_days': 20,
        'use_consol_days': True,
        'consol_days': 10,
        'use_min_rise_22': True,
        'min_rise_22': 10,
        'use_min_rise_67': True,
        'min_rise_67': 40,
        'use_min_rise_126': True,
        'min_rise_126': 80,
        'use_max_range': True,
        'max_range': 10,
        'use_min_adr': True,
        'min_adr': 2,
        'use_candle_strength': True,
        'use_volume_filter': False,
        'min_volume': 1000,
        'use_intraday_rise': False,
        'min_intraday_rise': 10,
        'max_stock_percentage': 50.0
    }
    if 'df' in st.session_state:
        del st.session_state['df']
    if 'tickers' in st.session_state:
        del st.session_state['tickers']
    st.rerun()

st.sidebar.header("即時股票查詢")
query_ticker = st.sidebar.text_input("輸入股票代碼（例如 AAPL）")
if st.sidebar.button("查詢股票"):
    if query_ticker:
        with st.spinner(f"正在查詢 {query_ticker}..."):
            result = screen_single_stock(
                query_ticker,
                prior_days=st.session_state['screening_params']['prior_days'],
                consol_days=st.session_state['screening_params']['consol_days'],
                min_rise_22=st.session_state['screening_params']['min_rise_22'],
                min_rise_67=st.session_state['screening_params']['min_rise_67'],
                min_rise_126=st.session_state['screening_params']['min_rise_126'],
                max_range=st.session_state['screening_params']['max_range'],
                min_adr=st.session_state['screening_params']['min_adr'],
                use_candle_strength=st.session_state['screening_params']['use_candle_strength'],
                use_volume_filter=st.session_state['screening_params']['use_volume_filter'],
                min_volume=st.session_state['screening_params']['min_volume'],
                use_intraday_rise=st.session_state['screening_params']['use_intraday_rise'],
                min_intraday_rise=st.session_state['screening_params']['min_intraday_rise']
            )
            st.session_state['query_result'] = result
            st.subheader(f"{query_ticker} 篩選結果")
            if result.empty:
                st.error(f"無法從 yfinance 獲取 {query_ticker} 的數據，請檢查股票代碼或網絡連線")
            else:
                result['Date'] = pd.to_datetime(result['Date']).dt.strftime('%Y-%m-%d')
                latest_date = result['Date'].max()
                latest_result = result[result['Date'] == latest_date].copy()
                latest_result['Status'] = latest_result.apply(
                    lambda row: "已突破且可買入" if row['Breakout'] and row['Breakout_Volume'] and (not st.session_state['screening_params']['use_candle_strength'] or row['Candle_Strength'])
                    else "已突破但條件不足" if row['Breakout']
                    else "盤整中" if row['Consolidation_Range_%'] < st.session_state['screening_params']['max_range']
                    else "前段上升", axis=1
                )
                display_df = latest_result.rename(columns={
                    'Ticker': '股票代碼', 'Date': '日期', 'Price': '價格',
                    'Prior_Rise_22_%': '22 日內漲幅 (%)', 'Prior_Rise_67_%': '67 日內漲幅 (%)',
                    'Prior_Rise_126_%': '126 日內漲幅 (%)', 'Consolidation_Range_%': '盤整範圍 (%)',
                    'ADR_%': '平均日波幅 (%)', 'Breakout': '是否突破', 'Breakout_Volume': '突破成交量',
                    'Candle_Strength': 'K線強度', 'Stop_Loss': '止損點',
                    'Target_20%': '目標20%', 'Target_50%': '目標50%', 'Target_100%': '目標100%'
                })
                desired_columns = ['股票代碼', '日期', '價格', '22 日內漲幅 (%)', '67 日內漲幅 (%)', '126 日內漲幅 (%)', 
                                   '盤整範圍 (%)', '平均日波幅 (%)', 'Status', 'K線強度', '止損點', '目標20%', '目標50%', '目標100%']
                available_columns = [col for col in desired_columns if col in display_df.columns]
                st.dataframe(display_df[available_columns])
                if latest_result['Status'].iloc[0] == "已突破且可買入":
                    st.success(f"{query_ticker} 符合條件：已突破且可買入")
                else:
                    st.warning(f"{query_ticker} 不符合條件，當前狀態：{latest_result['Status'].iloc[0]}")

if submit_button:
    if 'df' in st.session_state:
        del st.session_state['df']

    try:
        tickers_df = pd.read_csv(TICKERS_CSV)
        csv_tickers = tickers_df['Ticker'].tolist()
    except Exception as e:
        st.error(f"無法讀取 {TICKERS_CSV}: {str(e)}")
        csv_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA']

    if st.session_state['screening_params']['index_option'] == "NASDAQ 100":
        tickers = get_nasdaq_100(csv_tickers)
    elif st.session_state['screening_params']['index_option'] == "S&P 500":
        tickers = get_sp500()
    else:
        tickers = get_nasdaq_all(csv_tickers)
    total_tickers = len(tickers)
    max_stocks = int(total_tickers * (st.session_state['screening_params']['max_stock_percentage'] / 100.0))
    tickers = tickers[:max_stocks]
    st.session_state['tickers'] = tickers

    if not os.path.exists(DB_PATH):
        st.error("資料庫 stocks.db 不存在，請先初始化或更新資料庫")
    else:
        with st.spinner("篩選中..."):
            progress_bar = st.progress(0)
            df = screen_stocks(
                tickers=tickers,
                stock_pool=st.session_state['screening_params']['index_option'],
                prior_days=st.session_state['screening_params']['prior_days'],
                consol_days=st.session_state['screening_params']['consol_days'],
                min_rise_22=st.session_state['screening_params']['min_rise_22'],
                min_rise_67=st.session_state['screening_params']['min_rise_67'],
                min_rise_126=st.session_state['screening_params']['min_rise_126'],
                max_range=st.session_state['screening_params']['max_range'],
                min_adr=st.session_state['screening_params']['min_adr'],
                use_candle_strength=st.session_state['screening_params']['use_candle_strength'],
                progress_bar=progress_bar,
                use_volume_filter=st.session_state['screening_params']['use_volume_filter'],
                min_volume=st.session_state['screening_params']['min_volume'],
                use_intraday_rise=st.session_state['screening_params']['use_intraday_rise'],
                min_intraday_rise=st.session_state['screening_params']['min_intraday_rise']
            )
            progress_bar.progress(1.0)
            if df.empty:
                st.warning("無符合條件的股票，請調整參數")
            else:
                st.session_state['df'] = df

if 'df' in st.session_state:
    df = st.session_state['df']
    st.subheader("篩選結果")
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    latest_date = df['Date'].max()
    latest_df = df[df['Date'] == latest_date].copy()
    latest_df['Status'] = latest_df.apply(
        lambda row: "已突破且可買入" if row['Breakout'] and row['Breakout_Volume'] and (not st.session_state['screening_params']['use_candle_strength'] or row['Candle_Strength'])
        else "已突破但條件不足" if row['Breakout']
        else "盤整中" if (not st.session_state['screening_params']['use_max_range'] or row['Consolidation_Range_%'] < st.session_state['screening_params']['max_range'])
        else "前段上升", axis=1
    )

    # 計算最近5日平均成交量
    stock_data = st.session_state.get('stock_data')
    if stock_data is not None:
        avg_volumes = {}
        for ticker in latest_df['Ticker'].unique():
            try:
                ticker_data = stock_data.xs(ticker, level='Ticker', axis=1)
                recent_volume = ticker_data['Volume'].tail(5)
                avg_volumes[ticker] = recent_volume.mean()
            except KeyError:
                avg_volumes[ticker] = None
        latest_df['Avg_Volume_5D'] = latest_df['Ticker'].map(avg_volumes)

    display_df = latest_df.rename(columns={
        'Ticker': '股票代碼', 'Date': '日期', 'Price': '價格',
        'Prior_Rise_22_%': '22 日內漲幅 (%)', 'Prior_Rise_67_%': '67 日內漲幅 (%)',
        'Prior_Rise_126_%': '126 日內漲幅 (%)', 'Consolidation_Range_%': '盤整範圍 (%)',
        'ADR_%': '平均日波幅 (%)', 'Breakout': '是否突破', 'Breakout_Volume': '突破成交量',
        'Candle_Strength': 'K線強度', 'Stop_Loss': '止損點',
        'Target_20%': '目標20%', 'Target_50%': '目標50%', 'Target_100%': '目標100%',
        'Avg_Volume_5D': '最近5日平均成交量'
    })

    desired_columns = ['股票代碼', '日期', '價格', '22 日內漲幅 (%)', '67 日內漲幅 (%)', '126 日內漲幅 (%)', 
                       '盤整範圍 (%)', '平均日波幅 (%)', 'Status', 'K線強度', '止損點', '目標20%', '目標50%', '目標100%', '最近5日平均成交量']
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

    breakout_df = latest_df[latest_df['Breakout'] & latest_df['Breakout_Volume'] & (latest_df['Candle_Strength'] if st.session_state['screening_params']['use_candle_strength'] else True)]
    if not breakout_df.empty:
        st.subheader("當前突破股票（可買入）")
        plot_breakout_stocks(breakout_df['Ticker'].unique(), st.session_state['screening_params']['consol_days'])
    else:
        st.info("當前無突破股票")

tickers = st.session_state.get('tickers', [])
st.write(f"篩選範圍：{st.session_state['screening_params']['index_option']} ({len(tickers)} 隻股票)")
