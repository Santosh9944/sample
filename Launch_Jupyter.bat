@echo off
echo ============================================
echo   VerifAI - Launching Jupyter Notebook
echo ============================================
cd /d "c:\Users\santo\OneDrive\Desktop\DL PROJECT"
echo.
echo Starting Jupyter Notebook in your project folder...
echo The browser will open automatically at http://localhost:8888
echo.
echo Press Ctrl+C to stop the server.
echo ============================================
python -m notebook --notebook-dir="c:\Users\santo\OneDrive\Desktop\DL PROJECT"
pause
