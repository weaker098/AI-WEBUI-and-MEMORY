@echo off
SETLOCAL ENABLEEXTENSIONS
cd /d "%~dp0"

:: === Color Setup ===
color 0A

cls
echo.
echo  ██████╗ ██╗██╗   ██╗███████╗████████╗
echo  ██╔══██╗██║██║   ██║██╔════╝╚══██╔══╝
echo  ██████╔╝██║██║   ██║█████╗     ██║   
echo  ██╔══██╗██║╚██╗ ██╔╝██╔══╝     ██║   
echo  ██║  ██║██║ ╚████╔╝ ███████╗   ██║   
echo  ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝   ╚═╝   
echo.
echo  [ LAUNCHER v2.0 ]  ~  Your sassy local AI, ready to roll.
echo  ================================================================
echo.

:: === Log timestamp ===
echo [%date% %time%] Launch attempt >> install_log.txt

:: ================================================================
:: STEP 1 — Python Check
:: ================================================================
echo  [1/5] Checking Python...
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ❌  Python not found!
    echo      Download it at: https://www.python.org/downloads/
    echo      Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b
)

for /f "tokens=*" %%V in ('python --version 2^>^&1') do set PYVER=%%V
echo  ✅  %PYVER% detected.
echo.

:: ================================================================
:: STEP 2 — Pip Check
:: ================================================================
echo  [2/5] Checking pip...
python -m pip --version >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  ⚠️   pip missing — attempting install...
    python -m ensurepip --default-pip
) ELSE (
    echo  ✅  pip is available.
)
echo.

:: ================================================================
:: STEP 3 — Package Check
:: ================================================================
echo  [3/5] Checking required packages...
echo  ----------------------------------------------------------------

SET PACKAGES=flask flask-cors fuzzywuzzy rapidfuzz requests python-dateutil flake8 ddgs tiktoken

FOR %%P IN (%PACKAGES%) DO (
    python -c "import %%P" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo  📦  Installing missing package: %%P ...
        python -m pip install %%P --quiet
        python -c "import %%P" >nul 2>nul
        IF %ERRORLEVEL% NEQ 0 (
            echo  ❌  Failed to install %%P — check your internet connection.
        ) ELSE (
            echo  ✅  %%P installed successfully.
        )
    ) ELSE (
        echo  ✅  %%P already installed.
    )
)

:: Deprecated package warning
python -c "import importlib.util; exit(0) if importlib.util.find_spec('duckduckgo_search') is None else exit(1)" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ⚠️   WARNING: Deprecated package "duckduckgo_search" detected.
    echo      Please replace it with "ddgs" in your environment.
)
echo.

:: ================================================================
:: STEP 4 — Flake8 Lint
:: ================================================================
echo  [4/5] Running flake8 lint check on app.py...
echo  ----------------------------------------------------------------

python -m flake8 app.py --max-line-length=350 > flake8_log.txt 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo  ⚠️   Flake8 found some issues ^(check flake8_log.txt^):
    type flake8_log.txt
) ELSE (
    echo  ✅  app.py passed flake8 — no issues found.
)
echo.

:: ================================================================
:: STEP 5 — Launch
:: ================================================================
echo  [5/5] Starting app...
echo  ----------------------------------------------------------------
echo.
echo  🦊  Server starting at http://127.0.0.1:5000
echo  🦊  Press CTRL+C to stop the server.
echo.
echo  ================================================================
echo.

:: Log successful launch
echo [%date% %time%] Server launched successfully >> install_log.txt

python app.py

:: ================================================================
:: On Exit
:: ================================================================
echo.
echo  ================================================================
echo  🦊  app has stopped. Press any key to close.
echo  ================================================================
echo.
echo [%date% %time%] Server stopped >> install_log.txt

pause
exit /b
