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
echo  [ LAUNCHER v3.2 ]  ~  Your sassy local AI, ready to roll.
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
echo  [1/7] Checking Python...
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
echo  [2/7] Checking pip...
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
echo  [3/7] Checking core packages...
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
echo  [4/7] Checking AI/ML packages...
echo  ----------------------------------------------------------------

%PYTHON_CMD% -c "import numpy" >nul 2>nul || (echo  ..  Installing numpy... && %PYTHON_CMD% -m pip install numpy --quiet && echo  OK  Installed numpy.)
%PYTHON_CMD% -c "import bs4" >nul 2>nul || (echo  ..  Installing beautifulsoup4... && %PYTHON_CMD% -m pip install beautifulsoup4 --quiet && echo  OK  Installed beautifulsoup4.)
%PYTHON_CMD% -c "import faiss" >nul 2>nul || (echo  ..  Installing faiss-cpu... && %PYTHON_CMD% -m pip install faiss-cpu --quiet && echo  OK  Installed faiss-cpu.)
%PYTHON_CMD% -c "import sentence_transformers" >nul 2>nul || (echo  ..  Installing sentence-transformers... && %PYTHON_CMD% -m pip install sentence-transformers --quiet && echo  OK  Installed sentence-transformers.)

echo  OK  AI/ML packages verified.
echo.

:: ================================================================
:: STEP 5 — Video Packages
:: ================================================================
echo  [5/7] Checking video packages...
echo  ----------------------------------------------------------------

%PYTHON_CMD% -c "import cv2" >nul 2>nul || (echo  ..  Installing opencv-python-headless... && %PYTHON_CMD% -m pip install opencv-python-headless --quiet && echo  OK  Installed opencv-python-headless.)

echo  OK  Video packages verified.
echo.

:: ================================================================
:: STEP 6 — Pre-flight Checks
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

echo  OK  Pre-flight clear.
echo.

:: ================================================================
:: STEP 7 — Launch
:: ================================================================
echo  [7/7] Starting app...
echo  ================================================================
echo.
echo  %ESC%[97m  Server starting at %ESC%[91mhttp://127.0.0.1:5000%ESC%[0m
echo.

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
