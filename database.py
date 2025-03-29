import time
import os
import subprocess
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import requests
from pytz import timezone
import logging
from typing import Optional, List, Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constant configuration
REPO_DIR = "."
DB_PATH = "stocks.db"
TICKERS_CSV = "Tickers.csv"
REPO_URL = "https://github.com/KellifizW/Q-MagV1.git"
US_EASTERN = timezone('US/Eastern')
YF_BATCH_SIZE = 20  # yfinance batch size
MS_BATCH_SIZE = 100  # Marketstack batch size
MONTHLY_REQUEST_LIMIT = 100  # Marketstack free tier monthly limit
MAX_HISTORY_DAYS = 90  # Maximum historical data days

# Streamlit log display
def log_to_page(message: str, level: str = "INFO") -> None:
    """Display log messages on the Streamlit page"""
    if level == "INFO":
        st.info(message)
    elif level == "WARNING":
        st.warning(message)
    elif level == "ERROR":
        st.error(message)
    elif level == "DEBUG":
        st.write(f"DEBUG: {message}")

# Data validation functions
def safe_float(value: Any, column_name: str, ticker: str, date: str) -> Optional[float]:
    """Safely convert to float"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (ValueError, TypeError) as e:
        logger.error(f"Conversion failed for {column_name}, Ticker: {ticker}, Date: {date}, Value: {repr(value)}")
        return None

def safe_int(value: Any, column_name: str, ticker: str, date: str) -> int:
    """Safely convert to integer"""
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (ValueError, TypeError) as e:
        logger.error(f"Conversion failed for {column_name}, Ticker: {ticker}, Date: {date}, Value: {repr(value)}")
        return 0

# Data cache check
def check_data_exists(db_conn: sqlite3.Connection, ticker: str, date: str) -> bool:
    """Check if data already exists"""
    cursor = None
    try:
        cursor = db_conn.cursor()
        cursor.execute(
            "SELECT 1 FROM stocks WHERE Ticker = ? AND Date = ?",
            (ticker, date))
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"Failed to check data existence: {str(e)}")
        return False
    finally:
        if cursor:
            cursor.close()

# Marketstack data processing
def process_marketstack_data(api_data: List[Dict]) -> Optional[pd.DataFrame]:
    """Process raw data returned from Marketstack API"""
    if not api_data:
        return None

    records = []
    for item in api_data:
        # Verify required fields
        required_fields = {'date', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        if not all(field in item for field in required_fields):
            continue

        try:
            records.append({
                "Date": item["date"].split("T")[0],
                "Ticker": item["symbol"],
                "Open": item["open"],
                "High": item["high"],
                "Low": item["low"],
                "Close": item["close"],
                "Volume": item["volume"],
                "Adj_Close": item.get("adj_close", item["close"])
            })
        except Exception as e:
            logger.error(f"Failed to process data item: {str(e)}")
            continue

    return pd.DataFrame(records) if records else None

# Optimized Marketstack download function
def download_with_marketstack(
    tickers: List[str],
    start: datetime,
    end: datetime,
    api_key: str,
    request_count: List[int],
    db_conn: sqlite3.Connection
) -> Optional[pd.DataFrame]:
    """Optimized Marketstack data download"""
    if request_count[0] >= MONTHLY_REQUEST_LIMIT:
        logger.error(f"Monthly request limit of {MONTHLY_REQUEST_LIMIT} reached")
        return None

    try:
        # Filter out existing data (only check the last day)
        needed_tickers = [
            t for t in tickers 
            if not check_data_exists(db_conn, t, end.strftime('%Y-%m-%d'))
        ]
        
        if not needed_tickers:
            logger.info(f"All {len(tickers)} stock data already exists, skipping download")
            return pd.DataFrame()

        symbols = ','.join(needed_tickers)
        url = f"http://api.marketstack.com/v1/eod?access_key={api_key}&symbols={symbols}&date_from={start.strftime('%Y-%m-%d')}&date_to={end.strftime('%Y-%m-%d')}"
        
        logger.info(f"Requesting Marketstack API: {url.replace(api_key, '***')}")
        response = requests.get(url, timeout=10)
        
        # Increment counter only on successful response
        if response.status_code == 200:
            request_count[0] += 1
            logger.info(f"Remaining API requests: {MONTHLY_REQUEST_LIMIT - request_count[0]}")
        else:
            logger.error(f"Request failed, status code: {response.status_code}")
            return None

        data = response.json()
        
        if "error" in data:
            logger.error(f"API returned error: {data['error']}")
            return None
            
        return process_marketstack_data(data.get("data", []))
        
    except Exception as e:
        logger.error(f"Marketstack request exception: {str(e)}")
        return None

# Data download function with retry
def download_with_retry(
    tickers: List[str],
    start: datetime,
    end: datetime,
    api_key: str,
    request_count: List[int],
    db_conn: sqlite3.Connection,
    retries: int = 2,
    delay: int = 5
) -> Optional[pd.DataFrame]:
    """Data download with retry"""
    for attempt in range(retries):
        try:
            # Force yfinance failure (for testing)
            raise Exception("Force yfinance failure")
            data = yf.download(tickers, start=start, end=end, group_by='ticker', progress=False)
            if data.empty:
                logger.warning(f"yfinance data empty for: {tickers}")
                return None
            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[1] if col[0] == '' else '_'.join(col) for col in data.columns]
            return data
        except Exception as e:
            logger.warning(f"yfinance download failed for {tickers}, attempt {attempt+1}/{retries}: {str(e)}")
            time.sleep(delay)

    # Fall back to Marketstack
    return download_with_marketstack(tickers, start, end, api_key, request_count, db_conn)

# Batch insert data into database
def bulk_insert_data(df: pd.DataFrame, db_conn: sqlite3.Connection) -> bool:
    """Efficiently batch insert data into database"""
    if df.empty:
        return True

    try:
        # Ensure DataFrame columns match the table schema
        expected_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        if not all(col in expected_columns for col in df.columns):
            df = df.reindex(columns=expected_columns)
            logger.warning(f"Adjusted DataFrame columns to match schema: {df.columns.tolist()}")

        df.to_sql('stocks', db_conn, if_exists='append', index=False, 
                 dtype={
                     'Date': 'TEXT',
                     'Ticker': 'TEXT',
                     'Open': 'REAL',
                     'High': 'REAL',
                     'Low': 'REAL',
                     'Close': 'REAL',
                     'Adj_Close': 'REAL',
                     'Volume': 'INTEGER'
                 })
        db_conn.commit()
        return True
    except Exception as e:
        logger.error(f"Batch insert failed: {str(e)}")
        db_conn.rollback()
        return False

# Initialize Git repository
def init_repo() -> bool:
    """Initialize Git repository"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists('.git'):
            subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
            log_to_page("Initialized Git repository", "INFO")

        if "TOKEN" not in st.secrets:
            st.error("GitHub TOKEN not found, please configure it in Secrets")
            return False

        token = st.secrets["TOKEN"]
        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        
        subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True, text=True)
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'config', 'user.name', 'KellifizW'], check=True)
        subprocess.run(['git', 'config', 'user.email', 'your.email@example.com'], check=True)
        
        log_to_page("Git repository initialization completed", "INFO")
        return True
    except Exception as e:
        st.error(f"Failed to initialize Git repository: {str(e)}")
        return False

# Push to GitHub
def push_to_github(message: str = "Update stocks.db") -> bool:
    """Push changes to GitHub"""
    try:
        os.chdir(REPO_DIR)
        if not os.path.exists(DB_PATH):
            st.error(f"{DB_PATH} does not exist")
            return False

        # Check for changes
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if not status.stdout.strip():
            st.write("No changes to push")
            return True

        subprocess.run(['git', 'add', DB_PATH], check=True, capture_output=True, text=True)
        subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True, text=True)
        
        token = st.secrets["TOKEN"]
        remote_url = f"https://{token}@github.com/KellifizW/Q-MagV1.git"
        branch = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip() or 'main'
        
        subprocess.run(['git', 'push', remote_url, branch], check=True, capture_output=True, text=True)
        return True
    except Exception as e:
        st.error(f"Push failed: {str(e)}")
        return False

# Initialize database
def init_database() -> bool:
    """Initialize database"""
    if 'db_initialized' not in st.session_state:
        try:
            if "TOKEN" not in st.secrets:
                st.error("GitHub TOKEN not found")
                return False

            # Attempt to download existing database from GitHub
            url = "https://raw.githubusercontent.com/KellifizW/Q-MagV1/main/stocks.db"
            headers = {"Authorization": f"token {st.secrets['TOKEN']}"}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                with open(DB_PATH, "wb") as f:
                    f.write(response.content)
                st.write("Successfully downloaded stocks.db from GitHub")
            else:
                # Create new database
                conn = sqlite3.connect(DB_PATH)
                conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
                    Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, 
                    Close REAL, Adj_Close REAL, Volume INTEGER,
                    PRIMARY KEY (Date, Ticker))''')
                conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
                conn.commit()
                conn.close()
                st.write("Created new stocks.db database")
            
            st.session_state['db_initialized'] = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize database: {str(e)}")
            st.session_state['db_initialized'] = False
            return False
    return True

# Main update function
def update_database(
    tickers_file: str = TICKERS_CSV,
    db_path: str = DB_PATH,
    repo: Optional[bool] = None,
    check_percentage: float = 0.1
) -> bool:
    """Main database update function"""
    if repo is None:
        st.error("Git repository object not provided")
        return False

    try:
        # Read stock tickers
        tickers_df = pd.read_csv(tickers_file)
        tickers = tickers_df['Ticker'].tolist()
        log_to_page(f"Read {len(tickers)} stocks from {tickers_file}", "INFO")

        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        
        # Ensure table structure exists
        conn.execute('''CREATE TABLE IF NOT EXISTS stocks (
            Date TEXT, Ticker TEXT, Open REAL, High REAL, Low REAL, 
            Close REAL, Adj_Close REAL, Volume INTEGER,
            PRIMARY KEY (Date, Ticker))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS metadata (last_updated TEXT)''')
        conn.commit()

        # Get current date
        current_date = datetime.now(US_EASTERN).date()
        end_date = current_date - timedelta(days=1)
        
        # Get last update date for each ticker
        ticker_dates = pd.read_sql_query(
            "SELECT Ticker, MAX(Date) as last_date FROM stocks GROUP BY Ticker", 
            conn)
        existing_tickers = dict(zip(ticker_dates['Ticker'], pd.to_datetime(ticker_dates['last_date']).dt.date))

        # Determine stocks to update
        num_to_check = max(1, int(len(tickers) * check_percentage))
        tickers_to_check = tickers[-num_to_check:]
        tickers_to_update = []
        default_start_date = end_date - timedelta(days=MAX_HISTORY_DAYS)

        for ticker in tickers_to_check:
            last_date = existing_tickers.get(ticker)
            if not last_date or (end_date - last_date).days > 0:
                tickers_to_update.append(ticker)

        if not tickers_to_update:
            if len(existing_tickers) >= len(tickers):
                st.write("Database is up-to-date and complete, no update needed")
                conn.close()
                return True
            else:
                st.write(f"Database missing some stock data (existing {len(existing_tickers)} / total {len(tickers)}), updating missing parts")
                tickers_to_update = [t for t in tickers if t not in existing_tickers]

        # Get API key
        try:
            api_key = st.secrets["MARKETSTACK_API_KEY"]
            if not api_key:
                st.error("Marketstack API Key is empty")
                return False
        except KeyError:
            st.error("MARKETSTACK_API_KEY not found")
            return False

        # Set progress display
        st.write("Download progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        request_count = [0]
        success_count = [0]
        fail_count = [0]
        
        # Display API usage
        api_usage = st.sidebar.empty()
        
        # Process in batches
        total_batches = (len(tickers_to_update) + MS_BATCH_SIZE - 1) // MS_BATCH_SIZE
        
        for i in range(0, len(tickers_to_update), MS_BATCH_SIZE):
            batch_tickers = tickers_to_update[i:i + MS_BATCH_SIZE]
            batch_num = i // MS_BATCH_SIZE + 1
            
            # Update progress
            progress = min((i + MS_BATCH_SIZE) / len(tickers_to_update), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing batch {batch_num}/{total_batches}: {', '.join(batch_tickers[:3])}...")
            
            # Check which stocks need updating
            need_update = []
            for ticker in batch_tickers:
                if not check_data_exists(conn, ticker, end_date.strftime('%Y-%m-%d')):
                    need_update.append(ticker)
            
            if not need_update:
                logger.info(f"Batch {batch_num} data is up-to-date, skipping")
                continue
                
            # Calculate batch start date
            batch_start_dates = [
                existing_tickers.get(t, default_start_date) - timedelta(days=1)
                for t in need_update
            ]
            start_date = min(batch_start_dates)
            
            # Download data
            data = download_with_retry(
                need_update, 
                start=start_date,
                end=end_date,
                api_key=api_key,
                request_count=request_count,
                db_conn=conn
            )
            
            if data is not None and not data.empty:
                if bulk_insert_data(data, conn):
                    success_count[0] += 1
                else:
                    fail_count[0] += 1
            
            # Update API usage display
            api_usage.markdown(f"""
                **API Usage**  
                Success: {success_count[0]}  
                Failure: {fail_count[0]}  
                Remaining: {MONTHLY_REQUEST_LIMIT - request_count[0]}
            """)

        # Update metadata
        conn.execute("INSERT OR REPLACE INTO metadata (last_updated) VALUES (?)", 
                    (end_date.strftime('%Y-%m-%d'),))
        conn.commit()
        conn.close()

        # Push to GitHub
        if push_to_github(f"Updated data for {len(tickers_to_update)} stocks"):
            st.success("Database updated and pushed to GitHub")
        else:
            st.warning("Database updated but failed to push to GitHub")
        
        # Add download button
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                st.download_button(
                    label="Download stocks.db",
                    data=f,
                    file_name="stocks.db",
                    mime="application/octet-stream"
                )
        
        return True

    except Exception as e:
        st.error(f"Database update failed: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return False

def fetch_stock_data(
    tickers: List[str], 
    db_path: str = DB_PATH, 
    trading_days: int = 140
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Fetch stock data from database
    
    Args:
        tickers: List of stock tickers
        db_path: Database path (default: DB_PATH)
        trading_days: Number of trading days to fetch (default: 140)
        
    Returns:
        Tuple[Optional DataFrame, List of original tickers]:
            - DataFrame: Contains stock data (None if failed)
            - List[str]: Original input ticker list
    """
    try:
        # Verify database exists
        if not os.path.exists(db_path):
            st.error(f"Database file {db_path} does not exist")
            return None, tickers
            
        # Calculate start date
        start_date = (datetime.now(US_EASTERN).date() - timedelta(days=trading_days * 1.5)).strftime('%Y-%m-%d')
        
        # Use parameterized query to prevent SQL injection
        placeholders = ','.join(['?'] * len(tickers))
        query = f"""
        SELECT * FROM stocks 
        WHERE Ticker IN ({placeholders}) 
        AND Date >= ?
        ORDER BY Date
        """
        
        # Execute query
        conn = sqlite3.connect(db_path)
        data = pd.read_sql_query(
            query, 
            conn, 
            params=tickers + [start_date],
            parse_dates=['Date']
        )
        conn.close()
        
        # Check for empty data
        if data.empty:
            st.error(f"No data found for: {tickers}")
            return None, tickers
            
        # Convert to pivot format
        pivoted_data = data.pivot(index='Date', columns='Ticker')
        
        # Log statistics
        st.write(
            f"Fetched data - Stocks: {len(tickers)}, "
            f"Entries: {len(data)}, "
            f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}"
        )
        
        return pivoted_data, tickers
        
    except sqlite3.Error as e:
        st.error(f"Database query failed: {str(e)}")
        return None, tickers
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return None, tickers

# Main program
if __name__ == "__main__":
    st.title("Stock Database Update Tool")
    
    if init_repo() and init_database():
        if st.button("Initialize and Update Database"):
            update_database()
        
        if st.button("Update Database Only"):
            update_database()
