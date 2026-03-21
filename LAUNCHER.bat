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
echo  [ LAUNCHER v3.0 ]  ~  Your sassy local AI, ready to roll.
echo  ================================================================
echo.

:: === Log timestamp ===
echo [%date% %time%] Launch attempt >> install_log.txt

:: Track install counts
SET MISSING_COUNT=0
SET INSTALLED_COUNT=0
SET FAILED_COUNT=0
SET FAILED_LIST=

:: ================================================================
:: STEP 1 — Python Check
:: ================================================================
echo  [1/6] Checking Python...
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
echo  [2/6] Checking pip...
python -m pip --version >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  ⚠️   pip missing — attempting install...
    python -m ensurepip --default-pip
) ELSE (
    echo  ✅  pip is available.
)
echo.

:: ================================================================
:: STEP 3 — Core Packages  (import name == pip name, fast installs)
:: ================================================================
echo  [3/6] Checking core packages...
echo  ----------------------------------------------------------------

SET CORE=flask requests tiktoken ddgs fuzzywuzzy rapidfuzz flake8

FOR %%P IN (%CORE%) DO (
    python -c "import %%P" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo  📦  Installing: %%P ...
        SET /A MISSING_COUNT+=1
        python -m pip install %%P --quiet
        python -c "import %%P" >nul 2>nul
        IF %ERRORLEVEL% NEQ 0 (
            echo  ❌  FAILED: %%P
            SET /A FAILED_COUNT+=1
            SET FAILED_LIST=%FAILED_LIST% %%P
        ) ELSE (
            echo  ✅  Installed: %%P
            SET /A INSTALLED_COUNT+=1
        )
    ) ELSE (
        echo  ✅  %%P
    )
)
echo.

:: ================================================================
:: STEP 4 — AI/ML Packages  (pip name != import name, handled manually)
:: ================================================================
echo  [4/6] Checking AI/ML packages...
echo  ----------------------------------------------------------------
echo  ℹ️   Heavy packages — first install may take a few minutes.
echo.

:: --- numpy ---
python -c "import numpy" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  📦  Installing: numpy ...
    SET /A MISSING_COUNT+=1
    python -m pip install numpy --quiet
    python -c "import numpy" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 ( echo  ❌  FAILED: numpy & SET /A FAILED_COUNT+=1 & SET FAILED_LIST=%FAILED_LIST% numpy ) ELSE ( echo  ✅  Installed: numpy & SET /A INSTALLED_COUNT+=1 )
) ELSE ( echo  ✅  numpy )

:: --- beautifulsoup4 (imports as bs4) ---
python -c "import bs4" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  📦  Installing: beautifulsoup4 ...
    SET /A MISSING_COUNT+=1
    python -m pip install beautifulsoup4 --quiet
    python -c "import bs4" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 ( echo  ❌  FAILED: beautifulsoup4 & SET /A FAILED_COUNT+=1 & SET FAILED_LIST=%FAILED_LIST% beautifulsoup4 ) ELSE ( echo  ✅  Installed: beautifulsoup4 & SET /A INSTALLED_COUNT+=1 )
) ELSE ( echo  ✅  beautifulsoup4 ^(bs4^) )

:: --- faiss-cpu (imports as faiss) ---
python -c "import faiss" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  📦  Installing: faiss-cpu ...
    SET /A MISSING_COUNT+=1
    python -m pip install faiss-cpu --quiet
    python -c "import faiss" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 ( echo  ❌  FAILED: faiss-cpu & SET /A FAILED_COUNT+=1 & SET FAILED_LIST=%FAILED_LIST% faiss-cpu ) ELSE ( echo  ✅  Installed: faiss-cpu & SET /A INSTALLED_COUNT+=1 )
) ELSE ( echo  ✅  faiss-cpu )

:: --- sentence-transformers (imports as sentence_transformers) ---
python -c "import sentence_transformers" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  📦  Installing: sentence-transformers ^(this one is chunky, hang tight...^)
    SET /A MISSING_COUNT+=1
    python -m pip install sentence-transformers --quiet
    python -c "import sentence_transformers" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 ( echo  ❌  FAILED: sentence-transformers & SET /A FAILED_COUNT+=1 & SET FAILED_LIST=%FAILED_LIST% sentence-transformers ) ELSE ( echo  ✅  Installed: sentence-transformers & SET /A INSTALLED_COUNT+=1 )
) ELSE ( echo  ✅  sentence-transformers )

:: --- flask-cors (imports as flask_cors) ---
python -c "import flask_cors" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  📦  Installing: flask-cors ...
    SET /A MISSING_COUNT+=1
    python -m pip install flask-cors --quiet
    python -c "import flask_cors" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 ( echo  ❌  FAILED: flask-cors & SET /A FAILED_COUNT+=1 & SET FAILED_LIST=%FAILED_LIST% flask-cors ) ELSE ( echo  ✅  Installed: flask-cors & SET /A INSTALLED_COUNT+=1 )
) ELSE ( echo  ✅  flask-cors )

:: --- python-dateutil (imports as dateutil) ---
python -c "import dateutil" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  📦  Installing: python-dateutil ...
    SET /A MISSING_COUNT+=1
    python -m pip install python-dateutil --quiet
    python -c "import dateutil" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 ( echo  ❌  FAILED: python-dateutil & SET /A FAILED_COUNT+=1 & SET FAILED_LIST=%FAILED_LIST% python-dateutil ) ELSE ( echo  ✅  Installed: python-dateutil & SET /A INSTALLED_COUNT+=1 )
) ELSE ( echo  ✅  python-dateutil )

:: --- Deprecated package warning ---
python -c "import importlib.util; exit(0) if importlib.util.find_spec('duckduckgo_search') is None else exit(1)" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ⚠️   WARNING: Deprecated package "duckduckgo_search" detected.
    echo      Please uninstall it: pip uninstall duckduckgo_search
    echo      The new package "ddgs" is already installed above.
)
echo.

:: ================================================================
:: STEP 5 — Flake8 Lint
:: ================================================================
echo  [5/6] Running flake8 lint check on app.py...
echo  ----------------------------------------------------------------

python -m flake8 app.py --max-line-length=350 > flake8_log.txt 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo  ⚠️   Flake8 found some issues ^(saved to flake8_log.txt^)
    echo      Not blocking launch — check the log when you get a chance.
) ELSE (
    echo  ✅  app.py passed flake8 — clean code!
)
echo.

:: ================================================================
:: INSTALL SUMMARY
:: ================================================================
echo  ================================================================
echo  📋  PACKAGE SUMMARY
echo  ================================================================

IF %MISSING_COUNT%==0 (
    echo  ✅  All packages already installed. Nothing to do!
) ELSE (
    echo  📦  Found missing: %MISSING_COUNT% package^(s^)
    echo  ✅  Installed:     %INSTALLED_COUNT% package^(s^)
    IF %FAILED_COUNT% GTR 0 (
        echo  ❌  Failed:        %FAILED_COUNT% package^(s^) —%FAILED_LIST%
        echo.
        echo  ⚠️   Some packages failed to install.
        echo      Try running this launcher as Administrator,
        echo      or install manually: pip install%FAILED_LIST%
    )
)
echo.

:: ================================================================
:: WINDOWS DEFENDER TIP
:: ================================================================
echo  💡  TIP: If you see "_safe_replace fallback" warnings in the logs,
echo      add this folder to Windows Defender exclusions:
echo      Windows Security → Virus Protection → Exclusions → Add folder
echo      Path: %~dp0
echo.

:: ================================================================
:: STEP 6 — Launch
:: ================================================================
echo  [6/6] Starting app...
echo  ================================================================
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
