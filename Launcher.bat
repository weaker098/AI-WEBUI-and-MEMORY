@echo off
SETLOCAL ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
cd /d "%~dp0"

:: === Color Setup ===
color 0A

:: === ANSI Escape Codes (Windows 10+) ===
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

cls
echo.
echo  ██████╗ ██╗██╗   ██╗███████╗████████╗
echo  ██╔══██╗██║██║   ██║██╔════╝╚══██╔══╝
echo  ██████╔╝██║██║   ██║█████╗     ██║   
echo  ██╔══██╗██║╚██╗ ██╔╝██╔══╝     ██║   
echo  ██║  ██║██║ ╚████╔╝ ███████╗   ██║   
echo  ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝   ╚═╝   
echo.
echo  [ LAUNCHER v3.1 ]  ~  Your sassy local AI, ready to roll.
echo  ================================================================
echo.

:: === Log timestamp ===
echo [%date% %time%] Launch attempt >> install_log.txt

SET MISSING_COUNT=0
SET INSTALLED_COUNT=0
SET FAILED_COUNT=0
SET FAILED_LIST=

:: ================================================================
:: STEP 1 — Python Check
:: ================================================================
echo  [1/6] Checking Python...
SET PYTHON_CMD=
where python >nul 2>nul
IF %ERRORLEVEL% EQU 0 SET PYTHON_CMD=python
IF NOT DEFINED PYTHON_CMD (
    where py >nul 2>nul
    IF %ERRORLEVEL% EQU 0 SET PYTHON_CMD=py
)

IF NOT DEFINED PYTHON_CMD (
    echo X Python not found!
    pause
    exit /b
)
echo  OK detected.
echo.

:: ================================================================
:: STEP 2 — Pip Check
:: ================================================================
echo  [2/6] Checking pip...
%PYTHON_CMD% -m pip --version >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    %PYTHON_CMD% -m ensurepip --default-pip
) ELSE (
    echo  OK  pip is available.
)
echo.

:: ================================================================
:: STEP 3 — Core Packages
:: ================================================================
echo  [3/6] Checking core packages...
echo  ----------------------------------------------------------------

SET CORE=flask requests tiktoken ddgs fuzzywuzzy rapidfuzz

FOR %%P IN (%CORE%) DO (
    %PYTHON_CMD% -c "import %%P" >nul 2>nul
    IF !ERRORLEVEL! NEQ 0 (
        echo  ..  Installing: %%P ...
        SET /A MISSING_COUNT+=1
        %PYTHON_CMD% -m pip install %%P --quiet
        %PYTHON_CMD% -c "import %%P" >nul 2>nul
        IF !ERRORLEVEL! NEQ 0 (
            echo  X   FAILED: %%P
            SET /A FAILED_COUNT+=1
            SET FAILED_LIST=!FAILED_LIST! %%P
        ) ELSE (
            echo  OK  Installed: %%P
            SET /A INSTALLED_COUNT+=1
        )
    ) ELSE (
        echo  OK  %%P
    )
)
echo.

:: ================================================================
:: STEP 4 — AI/ML Packages
:: ================================================================
echo  [4/6] Checking AI/ML packages...
echo  ----------------------------------------------------------------

:: List of manual check packages
SET ML_LIST=numpy beautifulsoup4 faiss-cpu sentence-transformers keybert python-dateutil
:: Logic for ML imports
%PYTHON_CMD% -c "import numpy" >nul 2>nul || (echo Installing numpy... && %PYTHON_CMD% -m pip install numpy --quiet)
%PYTHON_CMD% -c "import bs4" >nul 2>nul || (echo Installing bs4... && %PYTHON_CMD% -m pip install beautifulsoup4 --quiet)
%PYTHON_CMD% -c "import faiss" >nul 2>nul || (echo Installing faiss... && %PYTHON_CMD% -m pip install faiss-cpu --quiet)
%PYTHON_CMD% -c "import sentence_transformers" >nul 2>nul || (echo Installing sentence-transformers... && %PYTHON_CMD% -m pip install sentence-transformers --quiet)
%PYTHON_CMD% -c "import keybert" >nul 2>nul || (echo Installing keybert... && %PYTHON_CMD% -m pip install keybert --quiet)
%PYTHON_CMD% -c "import dateutil" >nul 2>nul || (echo Installing dateutil... && %PYTHON_CMD% -m pip install python-dateutil --quiet)

echo  OK  AI/ML packages verified.
echo.

:: ================================================================
:: STEP 6 — Pre-flight & Launch
:: ================================================================
echo  [6/7] Pre-flight checks...
echo  ----------------------------------------------------------------

IF NOT EXIST "app.py" (
    echo  X   app.py not found!
    pause
    exit /b
)

netstat -ano 2>nul | findstr /r ":5000 .*LISTENING" >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo  !!  WARNING: Port 5000 is already in use!
    timeout /t 5 >nul
)

echo  [7/7] Starting app...
echo  ================================================================
echo.
echo  %ESC%[97m  Server starting at %ESC%[91mhttp://127.0.0.1:5000%ESC%[0m
echo.

:: Running the app directly so you can see the Flask output
%PYTHON_CMD% app.py

SET APP_EXIT_CODE=%ERRORLEVEL%
IF %APP_EXIT_CODE% NEQ 0 (
    echo.
    echo  !! App exited with error code: %APP_EXIT_CODE%
)

:end
echo.
pause
exit /b