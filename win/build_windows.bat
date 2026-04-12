@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1

echo.
echo  ============================
echo   Shaper Windows Build
echo  ============================
echo.

where py >nul 2>&1
if not errorlevel 1 (
    set "BOOTSTRAP=py -3"
) else (
    set "BOOTSTRAP=python"
)

%BOOTSTRAP% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.9+ was not found.
    exit /b 1
)

cd /d "%~dp0\.."
set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [1/4] Creating virtual environment...
    %BOOTSTRAP% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        exit /b 1
    )
) else (
    echo [1/4] Reusing existing virtual environment...
)

echo [2/4] Installing dependencies...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)

"%VENV_PY%" -m pip install -r win\requirements.txt pyinstaller
if errorlevel 1 (
    echo [ERROR] Failed to install build dependencies.
    exit /b 1
)

echo [3/4] Building Windows app...
if not exist "tools\primitive.exe" (
    echo [ERROR] Missing bundled primitive binary: tools\primitive.exe
    echo         Build or restore that file before packaging the Windows app.
    exit /b 1
)

"%VENV_PY%" -m PyInstaller win\shaper.spec --clean --noconfirm
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed.
    exit /b 1
)

echo.
echo [4/4] Build complete!
echo.
echo   Output folder: dist\
echo   Executable:    dist\Shaper.exe
echo.
echo   Ship the whole dist folder to publish the desktop app.
echo.

endlocal
