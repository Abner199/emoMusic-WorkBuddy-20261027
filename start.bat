@echo off
chcp 65001 >nul
title 心音 EmoSong - 本地预览服务器
echo ============================================
echo   心音 EmoSong  ^| 基于人脸情绪识别的歌曲推荐
echo   请勿直接双击 index.html，必须通过 http 访问
echo ============================================
echo.

set PORT=8787

where python >nul 2>nul
if %errorlevel%==0 (
    echo [INFO] 使用 Python 启动: http://127.0.0.1:%PORT%/
    start "" "http://127.0.0.1:%PORT%/"
    python -m http.server %PORT% --bind 127.0.0.1
    goto :eof
)

where py >nul 2>nul
if %errorlevel%==0 (
    echo [INFO] 使用 py 启动: http://127.0.0.1:%PORT%/
    start "" "http://127.0.0.1:%PORT%/"
    py -m http.server %PORT% --bind 127.0.0.1
    goto :eof
)

where node >nul 2>nul
if %errorlevel%==0 (
    echo [INFO] 未找到 Python，尝试用 Node 启动...
    npx --yes serve -l %PORT% .
    goto :eof
)

echo [ERROR] 未找到 Python 或 Node，请安装其中之一，或用任意静态服务器打开本目录。
pause
