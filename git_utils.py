import os
import subprocess
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class GitRepoManager:
    def __init__(self, repo_dir, repo_url, token):
        self.repo_dir = repo_dir
        self.repo_url = repo_url
        self.token = token
        self.remote_url = f"https://{token}@{repo_url.replace('https://', '')}"
        os.chdir(repo_dir)
        self._init_repo()

    def _init_repo(self):
        """初始化 Git 倉庫"""
        try:
            if not os.path.exists('.git'):
                subprocess.run(['git', 'init'], check=True, capture_output=True, text=True)
                subprocess.run(['git', 'lfs', 'install'], check=True, capture_output=True, text=True)
                logger.info("初始化 Git 倉庫並啟用 LFS")
            subprocess.run(['git', 'remote', 'remove', 'origin'], capture_output=True, text=True)
            subprocess.run(['git', 'remote', 'add', 'origin', self.remote_url], check=True, capture_output=True, text=True)
            subprocess.run(['git', 'config', 'user.name', 'KellifizW'], check=True)
            subprocess.run(['git', 'config', 'user.email', 'your.email@example.com'], check=True)
            logger.info("Git 倉庫初始化完成")
        except Exception as e:
            st.error(f"初始化 Git 倉庫失敗：{str(e)}")
            raise

    def track_lfs(self, file_path):
        """配置 LFS 追蹤"""
        try:
            subprocess.run(['git', 'lfs', 'track', file_path], check=True, capture_output=True, text=True)
            logger.info(f"已配置 {file_path} 為 LFS 檔案")
        except subprocess.CalledProcessError as e:
            logger.error(f"配置 LFS 追蹤失敗：{e.stderr}")
            raise

    def push(self, message="Update stocks.db"):
        """推送至 GitHub"""
        try:
            if not os.path.exists(DB_PATH):
                st.error(f"stocks.db 不存在於路徑：{DB_PATH}")
                return False

            subprocess.run(['git', 'add', DB_PATH], check=True, capture_output=True, text=True)
            status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
            if not status.stdout:
                st.write("無變更需要推送")
                return True

            subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True, text=True)
            branch = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip() or 'main'
            pull_result = subprocess.run(['git', 'pull', self.remote_url, branch], capture_output=True, text=True)
            if pull_result.returncode != 0:
                st.error(f"拉取遠端變更失敗：{pull_result.stderr}")
                return False

            subprocess.run(['git', 'push', self.remote_url, branch], check=True, capture_output=True, text=True)
            subprocess.run(['git', 'lfs', 'push', '--all', self.remote_url, branch], check=True, capture_output=True, text=True)
            st.write("成功推送至 GitHub（包含 LFS 檔案）")
            return True
        except subprocess.CalledProcessError as e:
            st.error(f"推送至 GitHub 失敗：{e.stderr}")
            return False
        except Exception as e:
            st.error(f"推送至 GitHub 失敗：{str(e)}")
            return False
