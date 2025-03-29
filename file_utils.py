import os
import subprocess
import sqlite3
import hashlib
import magic
import streamlit as st
import logging
import requests

logger = logging.getLogger(__name__)

def check_and_fetch_lfs_file(file_path, repo_url, token):
    """檢查並從 Git LFS 下載檔案"""
    try:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size < 1024:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.startswith('version https://git-lfs.github.com'):
                        logger.info(f"檢測到 {file_path} 是 LFS 指標檔案，開始下載")
                        subprocess.run(['git', 'lfs', 'pull', '--include', file_path], check=True, capture_output=True, text=True)
                        logger.info(f"已從 LFS 下載 {file_path}")
                        return True
            else:
                logger.info(f"{file_path} 檔案大小 {file_size} 位元組，假設已完整")
                return True
        else:
            logger.info(f"{file_path} 不存在，嘗試從 GitHub 下載")
            raw_url = f"https://raw.githubusercontent.com/KellifizW/Q-MagV1/main/{os.path.basename(file_path)}"
            response = requests.get(raw_url, headers={"Authorization": f"token {token}"})
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"已下載 {file_path}")
                return True
            else:
                logger.warning(f"無法從 {raw_url} 下載檔案，狀態碼：{response.status_code}")
                return False
    except Exception as e:
        logger.error(f"處理 LFS 檔案 {file_path} 時出錯：{str(e)}")
        return False

def diagnose_db_file(db_path):
    """診斷資料庫檔案"""
    diagnostics = [f"檢查檔案：{os.path.abspath(db_path)}"]
    if not os.path.exists(db_path):
        diagnostics.append("錯誤：檔案不存在")
        return diagnostics
    
    file_size = os.path.getsize(db_path)
    diagnostics.append(f"檔案大小：{file_size} 位元組")
    if file_size == 0:
        diagnostics.append("警告：檔案為空")

    try:
        file_type = magic.from_file(db_path, mime=True)
        diagnostics.append(f"檔案類型：{file_type}")
        if file_type != "application/x-sqlite3":
            diagnostics.append("警告：檔案不是 SQLite 資料庫格式")
    except Exception as e:
        diagnostics.append(f"檢查檔案類型失敗：{str(e)}")

    with open(db_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    diagnostics.append(f"檔案 MD5 哈希值：{file_hash}")

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'")
            if not cursor.fetchone():
                diagnostics.append("錯誤：缺少 'stocks' 表")
            else:
                cursor.execute("PRAGMA table_info(stocks)")
                columns = [col[1] for col in cursor.fetchall()]
                expected = {'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'}
                if not expected.issubset(columns):
                    diagnostics.append(f"警告：'stocks' 表缺少欄位：{expected - set(columns)}")
    except sqlite3.DatabaseError as e:
        diagnostics.append(f"資料庫結構檢查失敗：{str(e)}")

    return diagnostics
