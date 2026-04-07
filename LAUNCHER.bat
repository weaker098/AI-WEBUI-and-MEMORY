@echo off
SETLOCAL ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
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
echo  [ LAUNCHER v3.1 ]  ~  Your sassy local AI, ready to roll.
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

SET PYTHON_CMD=
where python >nul 2>nul
IF %ERRORLEVEL% EQU 0 SET PYTHON_CMD=python

IF NOT DEFINED PYTHON_CMD (
    where py >nul 2>nul
    IF %ERRORLEVEL% EQU 0 SET PYTHON_CMD=py
)

IF NOT DEFINED PYTHON_CMD (
    echo.
    echo  X  Python not found!
    echo     Download it at: https://www.python.org/downloads/
    echo     Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b
)

for /f "tokens=*" %%V in ('%PYTHON_CMD% --version 2^>^&1') do set PYVER=%%V
echo  OK  %PYVER% detected  ^(using command: %PYTHON_CMD%^)
echo.

:: ================================================================
:: STEP 2 — Pip Check
:: ================================================================
echo  [2/6] Checking pip...
%PYTHON_CMD% -m pip --version >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  !!  pip missing -- attempting install...
    %PYTHON_CMD% -m ensurepip --default-pip
) ELSE (
    echo  OK  pip is available.
)
echo.

:: ================================================================
:: STEP 3 — Core Packages  (import name == pip name, fast installs)
:: ================================================================
echo  [3/6] Checking core packages...
echo  ----------------------------------------------------------------

:: NOTE: flake8 removed from here — it's a dev tool, not a runtime dep.
::       It's checked separately in the lint step below.
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
:: STEP 4 — AI/ML Packages  (pip name != import name, handled manually)
:: ================================================================
echo  [4/6] Checking AI/ML packages...
echo  ----------------------------------------------------------------
echo  i   Heavy packages -- first install may take a few minutes.
echo.

:: --- numpy ---
%PYTHON_CMD% -c "import numpy" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  ..  Installing: numpy ...
    SET /A MISSING_COUNT+=1
    %PYTHON_CMD% -m pip install numpy --quiet
    %PYTHON_CMD% -c "import numpy" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo  X   FAILED: numpy
        SET /A FAILED_COUNT+=1
        SET FAILED_LIST=!FAILED_LIST! numpy
    ) ELSE (
        echo  OK  Installed: numpy
        SET /A INSTALLED_COUNT+=1
    )
) ELSE ( echo  OK  numpy )

:: --- beautifulsoup4 (imports as bs4) ---
%PYTHON_CMD% -c "import bs4" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  ..  Installing: beautifulsoup4 ...
    SET /A MISSING_COUNT+=1
    %PYTHON_CMD% -m pip install beautifulsoup4 --quiet
    %PYTHON_CMD% -c "import bs4" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo  X   FAILED: beautifulsoup4
        SET /A FAILED_COUNT+=1
        SET FAILED_LIST=!FAILED_LIST! beautifulsoup4
    ) ELSE (
        echo  OK  Installed: beautifulsoup4
        SET /A INSTALLED_COUNT+=1
    )
) ELSE ( echo  OK  beautifulsoup4 ^(bs4^) )

:: --- faiss-cpu (imports as faiss) ---
%PYTHON_CMD% -c "import faiss" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  ..  Installing: faiss-cpu ...
    SET /A MISSING_COUNT+=1
    %PYTHON_CMD% -m pip install faiss-cpu --quiet
    %PYTHON_CMD% -c "import faiss" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo  X   FAILED: faiss-cpu
        SET /A FAILED_COUNT+=1
        SET FAILED_LIST=!FAILED_LIST! faiss-cpu
    ) ELSE (
        echo  OK  Installed: faiss-cpu
        SET /A INSTALLED_COUNT+=1
    )
) ELSE ( echo  OK  faiss-cpu )

:: --- sentence-transformers (imports as sentence_transformers) ---
%PYTHON_CMD% -c "import sentence_transformers" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  ..  Installing: sentence-transformers ^(chunky -- hang tight...^)
    SET /A MISSING_COUNT+=1
    %PYTHON_CMD% -m pip install sentence-transformers --quiet
    %PYTHON_CMD% -c "import sentence_transformers" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo  X   FAILED: sentence-transformers
        SET /A FAILED_COUNT+=1
        SET FAILED_LIST=!FAILED_LIST! sentence-transformers
    ) ELSE (
        echo  OK  Installed: sentence-transformers
        SET /A INSTALLED_COUNT+=1
    )
) ELSE ( echo  OK  sentence-transformers )

:: --- keybert (session auto-naming, lazy-loaded on first message) ---
%PYTHON_CMD% -c "import keybert" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  ..  Installing: keybert ^(session auto-naming^) ...
    SET /A MISSING_COUNT+=1
    %PYTHON_CMD% -m pip install keybert --quiet
    %PYTHON_CMD% -c "import keybert" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo  X   FAILED: keybert
        SET /A FAILED_COUNT+=1
        SET FAILED_LIST=!FAILED_LIST! keybert
    ) ELSE (
        echo  OK  Installed: keybert
        SET /A INSTALLED_COUNT+=1
    )
) ELSE ( echo  OK  keybert )

:: --- python-dateutil (imports as dateutil) ---
%PYTHON_CMD% -c "import dateutil" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  ..  Installing: python-dateutil ...
    SET /A MISSING_COUNT+=1
    %PYTHON_CMD% -m pip install python-dateutil --quiet
    %PYTHON_CMD% -c "import dateutil" >nul 2>nul
    IF %ERRORLEVEL% NEQ 0 (
        echo  X   FAILED: python-dateutil
        SET /A FAILED_COUNT+=1
        SET FAILED_LIST=!FAILED_LIST! python-dateutil
    ) ELSE (
        echo  OK  Installed: python-dateutil
        SET /A INSTALLED_COUNT+=1
    )
) ELSE ( echo  OK  python-dateutil )

:: --- Deprecated package warning ---
%PYTHON_CMD% -c "import importlib.util; exit(0) if importlib.util.find_spec('duckduckgo_search') is None else exit(1)" >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo  !!  WARNING: Deprecated package "duckduckgo_search" detected.
    echo      Please uninstall it: pip uninstall duckduckgo_search
    echo      The new package "ddgs" is already installed above.
)
echo.

:: ================================================================
:: STEP 5 — Flake8 Lint  (optional dev check — skipped if not installed)
:: ================================================================
echo  [5/6] Running flake8 lint check on app.py...
echo  ----------------------------------------------------------------

%PYTHON_CMD% -m flake8 --version >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo  --  flake8 not installed -- skipping lint. ^(Install with: pip install flake8^)
) ELSE (
    %PYTHON_CMD% -m flake8 app.py --max-line-length=350 > flake8_log.txt 2>&1
    IF %ERRORLEVEL% NEQ 0 (
        echo  !!  Flake8 found some issues ^(saved to flake8_log.txt^)
        echo      Not blocking launch -- check the log when you get a chance.
    ) ELSE (
        echo  OK  app.py passed flake8 -- clean code!
    )
)
echo.

:: ================================================================
:: INSTALL SUMMARY
:: ================================================================
echo  ================================================================
echo  PACKAGE SUMMARY
echo  ================================================================

IF !MISSING_COUNT!==0 (
    echo  OK  All packages already installed. Nothing to do!
) ELSE (
    echo  ..  Found missing: !MISSING_COUNT! package^(s^)
    echo  OK  Installed:     !INSTALLED_COUNT! package^(s^)
    IF !FAILED_COUNT! GTR 0 (
        echo  X   Failed:        !FAILED_COUNT! package^(s^) --!FAILED_LIST!
        echo.
        echo  !!  Some packages failed to install.
        echo      Try running this launcher as Administrator,
        echo      or install manually: pip install!FAILED_LIST!
    )
)
echo.

:: ================================================================
:: WINDOWS DEFENDER TIP
:: ================================================================
echo  TIP: If you see "_safe_replace fallback" warnings in the logs,
echo      add this folder to Windows Defender exclusions:
echo      Windows Security ^> Virus Protection ^> Exclusions ^> Add folder
echo      Path: %~dp0
echo.

:: ================================================================
:: STEP 6 — Pre-flight Checks
:: ================================================================
echo  [6/7] Pre-flight checks...
echo  ----------------------------------------------------------------

:: --- app.py exists? ---
IF NOT EXIST "app.py" (
    echo  X   app.py not found in this folder!
    echo      Make sure LAUNCHER.bat is in the same folder as app.py.
    echo      Current folder: %~dp0
    echo [%date% %time%] PREFLIGHT FAIL: app.py not found >> install_log.txt
    pause
    exit /b
)
echo  OK  app.py found.

:: --- Port 5000 already in use? ---
netstat -ano 2>nul | findstr /r ":5000 .*LISTENING" >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo.
    echo  !!  WARNING: Port 5000 is already in use!
    echo      Something else is running on that port -- the app may fail to start.
    echo      Common culprits: another copy of this app, AirPlay ^(Mac^), or another Flask app.
    echo      To find what's using it: run  netstat -ano ^| findstr :5000
    echo      Then kill it via Task Manager or: taskkill /PID ^<pid^> /F
    echo.
    echo      Launching anyway in 5 seconds... ^(Ctrl+C to abort^)
    timeout /t 5 >nul
) ELSE (
    echo  OK  Port 5000 is free.
)

:: --- RAM check (models run CPU-only -- RAM matters, not VRAM) ---
:: NOTE: Your app forces device='cpu' on all models, so GPU/VRAM is not relevant.
::       What matters is free RAM. Nomic model alone needs ~1.5GB on load.
SET RAM_OK=1
for /f "tokens=2 delims==" %%M in ('wmic OS get FreePhysicalMemory /Value 2^>nul ^| findstr "="') do set FREE_RAM_KB=%%M

IF DEFINED FREE_RAM_KB (
    :: Convert KB to MB for readability
    SET /A FREE_RAM_MB=!FREE_RAM_KB! / 1024
    IF !FREE_RAM_MB! LSS 1500 (
        echo.
        echo  !!  WARNING: Low RAM detected -- only !FREE_RAM_MB! MB free.
        echo      Your app loads CPU-based AI models ^(no GPU needed^).
        echo      Recommended free RAM: 2000 MB+ for smooth model loading.
        echo      Close other apps and browsers to free up memory before continuing.
        echo      Launching anyway in 5 seconds... ^(Ctrl+C to abort^)
        SET RAM_OK=0
        timeout /t 5 >nul
    ) ELSE (
        echo  OK  RAM: !FREE_RAM_MB! MB free.
    )
) ELSE (
    echo  --  RAM check skipped ^(wmic unavailable^).
)

:: --- GPU note (informational only) ---
nvidia-smi >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo  OK  NVIDIA GPU detected -- but your app runs fully on CPU. GPU is unused.
) ELSE (
    echo  OK  No NVIDIA GPU -- your app runs fully on CPU, so this is fine.
)

:: --- Block launch if critical packages failed ---
IF !FAILED_COUNT! GTR 0 (
    echo.
    echo  ================================================================
    echo  !!  LAUNCH BLOCKED -- fix the failed packages above first.
    echo  ================================================================
    echo.
    echo [%date% %time%] Launch BLOCKED -- failed packages:!FAILED_LIST! >> install_log.txt
    pause
    exit /b
)

echo.

:: ================================================================
:: STEP 7 — Launch
:: ================================================================
echo  [7/7] Starting app...
echo  ================================================================
echo.
echo  Server starting at http://127.0.0.1:5000
echo  Press CTRL+C to stop the server.
echo.
echo  ================================================================
echo.

:: Log successful launch
echo [%date% %time%] Server launched successfully >> install_log.txt

:: Run and capture exit code.
:: stderr goes to crash_log.txt so tracebacks are preserved.
:: stdout still prints live to the console so you see Flask logs normally.
%PYTHON_CMD% app.py 2> crash_log.txt
SET APP_EXIT_CODE=%ERRORLEVEL%

:: ================================================================
:: Post-Exit — Crash Diagnosis
:: ================================================================
echo.

IF %APP_EXIT_CODE% EQU 0 (
    echo  ================================================================
    echo  App stopped cleanly ^(exit code 0^). Press any key to close.
    echo  ================================================================
    echo [%date% %time%] Server stopped cleanly >> install_log.txt
    goto :end
)

:: Non-zero exit = something went wrong. Diagnose.
echo  ================================================================
echo  !!  App crashed or exited with error ^(code: %APP_EXIT_CODE%^)
echo  ================================================================
echo.
echo  Diagnosing...
echo  ----------------------------------------------------------------

:: --- Check crash_log.txt for known patterns ---
SET DIAGNOSED=0

findstr /i "address already in use\|OSError.*5000\|WinError 10048" crash_log.txt >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo  CAUSE: Port 5000 is already in use.
    echo  FIX:   Another process is occupying port 5000.
    echo         Run: netstat -ano ^| findstr :5000
    echo         Then: taskkill /PID ^<pid^> /F
    SET DIAGNOSED=1
)

findstr /i "MemoryError\|Cannot allocate\|out of memory\|std::bad_alloc" crash_log.txt >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo  CAUSE: Out of RAM -- not enough memory to load AI models.
    echo  FIX:   Close other applications to free up RAM.
    IF DEFINED FREE_RAM_MB echo         Free RAM at launch was: !FREE_RAM_MB! MB
    echo         The Nomic model needs ~1.5 GB free. Try switching to all-mini in settings.
    SET DIAGNOSED=1
)

findstr /i "ModuleNotFoundError\|ImportError\|No module named" crash_log.txt >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo  CAUSE: A required Python package is missing or broken.
    echo  FIX:   Re-run this launcher -- it will attempt to reinstall missing packages.
    echo         If it keeps failing, try: pip install --force-reinstall ^<package^>
    SET DIAGNOSED=1
)

findstr /i "FileNotFoundError\|No such file or directory" crash_log.txt >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo  CAUSE: A required file or folder is missing.
    echo  FIX:   Make sure all app files are present and LAUNCHER.bat is in the app folder.
    SET DIAGNOSED=1
)

findstr /i "PermissionError\|Access is denied\|WinError 5" crash_log.txt >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo  CAUSE: File permission error ^(Windows blocked access to a file^).
    echo  FIX 1: Right-click LAUNCHER.bat and "Run as Administrator".
    echo  FIX 2: Add this folder to Windows Defender exclusions.
    SET DIAGNOSED=1
)

findstr /i "ConnectionRefusedError\|Failed to establish\|HTTPSConnectionPool" crash_log.txt >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo  CAUSE: Network error -- could not reach HuggingFace to download a model.
    echo  FIX:   Check your internet connection.
    echo         If you're offline, pre-download models or set HF_HUB_OFFLINE=1.
    SET DIAGNOSED=1
)

findstr /i "RuntimeError\|CUDA\|cuDNN\|cudart" crash_log.txt >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo  CAUSE: CUDA/GPU runtime error ^(unexpected -- app should be CPU-only^).
    echo  FIX:   This is unusual. Check crash_log.txt for the full traceback.
    SET DIAGNOSED=1
)

findstr /i "SyntaxError" crash_log.txt >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo  CAUSE: Python syntax error in app.py.
    echo  FIX:   Check flake8_log.txt for the exact line.
    SET DIAGNOSED=1
)

IF !DIAGNOSED! EQU 0 (
    echo  Could not auto-diagnose the crash.
    echo  Check crash_log.txt in this folder for the full Python traceback.
)

echo.
echo  ----------------------------------------------------------------
echo  Full error log saved to: crash_log.txt
echo  Tip: scroll up in this window to see the live output before the crash.
echo  ----------------------------------------------------------------
echo.
echo [%date% %time%] Server CRASHED ^(code %APP_EXIT_CODE%^) >> install_log.txt

:end
echo.
pause
exit /b
