@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ---------------------------
REM  ML Lab - one-click runner
REM  - Creates/updates conda env from environment.yml
REM  - Activates env
REM  - Runs Streamlit
REM  - Opens browser automatically
REM ---------------------------

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "ENV_FILE=%PROJECT_DIR%environment.yml"
set "ENV_NAME=streamlit_app_duth"
set "APP_FILE=app.py"
set "PORT=8501"

echo ---------------------------------------
echo   ML Lab - install env and run app
echo   Project: %PROJECT_DIR%
echo   Env:     %ENV_NAME%
echo   EnvFile: %ENV_FILE%
echo ---------------------------------------

IF NOT EXIST "%ENV_FILE%" (
  echo ERROR: environment.yml not found:
  echo   %ENV_FILE%
  pause
  exit /b 1
)

set "CONDA_BAT="
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if "%CONDA_BAT%"=="" if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if "%CONDA_BAT%"=="" if exist "C:\miniconda3\condabin\conda.bat" set "CONDA_BAT=C:\miniconda3\condabin\conda.bat"
if "%CONDA_BAT%"=="" if exist "C:\anaconda3\condabin\conda.bat"  set "CONDA_BAT=C:\anaconda3\condabin\conda.bat"

if "%CONDA_BAT%"=="" (
  for /f "delims=" %%i in ('where conda.bat 2^>nul') do (
    set "CONDA_BAT=%%i"
    goto :conda_found
  )
)
:conda_found

if "%CONDA_BAT%"=="" (
  echo.
  echo ERROR: Conda was not found automatically.
  set /p CONDA_ROOT=Anaconda/Miniconda install folder: 
  set "CONDA_ROOT=%CONDA_ROOT:"=%"
  if exist "%CONDA_ROOT%\condabin\conda.bat" (
    set "CONDA_BAT=%CONDA_ROOT%\condabin\conda.bat"
  ) else (
    echo ERROR: Could not find conda.bat at %CONDA_ROOT%\condabin\conda.bat
    pause
    exit /b 1
  )
)

echo.
echo Using conda:
echo   %CONDA_BAT%

set "ENV_EXISTS=0"
for /f "delims=" %%i in ('call "%CONDA_BAT%" env list 2^>nul ^| findstr /i /r "^[ ]*%ENV_NAME%[ ]"') do (
  set "ENV_EXISTS=1"
)

if "%ENV_EXISTS%"=="0" (
  echo Creating conda env "%ENV_NAME%"...
  call "%CONDA_BAT%" env create -n "%ENV_NAME%" -f "%ENV_FILE%"
  if errorlevel 1 (
    echo ERROR: Failed to create environment.
    pause
    exit /b 1
  )
) else (
  echo Updating conda env "%ENV_NAME%"...
  call "%CONDA_BAT%" env update -n "%ENV_NAME%" -f "%ENV_FILE%" --prune
  if errorlevel 1 (
    echo ERROR: Failed to update environment.
    pause
    exit /b 1
  )
)

echo Activating env "%ENV_NAME%" ...
call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
  echo ERROR: Failed to activate environment.
  pause
  exit /b 1
)

python --version
python -c "import streamlit; print('streamlit', streamlit.__version__)" 1>nul 2>nul
if errorlevel 1 (
  echo ERROR: Streamlit is not installed in this environment.
  pause
  exit /b 1
)

echo Starting Streamlit on port %PORT%...
start "ML Lab" /B cmd /c streamlit run "%PROJECT_DIR%%APP_FILE%" --server.port=%PORT% --server.headless=false
timeout /t 2 >nul
start "" "http://localhost:%PORT%"

echo If browser did not open, visit: http://localhost:%PORT%
pause
endlocal
