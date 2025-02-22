@echo off
cd /d D:\PROJECTS\research\mouse-data-servers
call venv\Scripts\activate
streamlit run scripts\app.py --server.port 8502
