@echo off
echo ============================================
echo   VerifAI - Launching Flask Web App
echo ============================================
cd /d "%~dp0"
echo.
echo Starting VerifAI web server...
echo Open your browser at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server.
echo ============================================
python app.py
pause
