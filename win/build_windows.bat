@echo off
REM ============================================
REM   Shaper Windows 构建脚本
REM   在 Windows 上运行此脚本来打包 .exe
REM ============================================
echo.
echo  ============================
echo   Shaper Windows Build
echo  ============================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python, 请先安装 Python 3.9+
    pause
    exit /b 1
)

REM 切到项目根目录
cd /d "%~dp0\.."

REM 创建虚拟环境 (如果不存在)
if not exist ".venv" (
    echo [1/4] 创建虚拟环境...
    python -m venv .venv
)

REM 激活虚拟环境并安装依赖
echo [2/4] 安装依赖...
call .venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r win\requirements.txt
pip install pyinstaller pywebview[cef]

REM 打包
echo [3/4] 打包为 Windows 应用...
pyinstaller win\shaper.spec --clean --noconfirm

echo.
echo [4/4] 构建完成!
echo.
echo   输出目录: dist\Shaper\
echo   可执行文件: dist\Shaper\Shaper.exe
echo.
echo   可以将 dist\Shaper 文件夹整体复制到任意位置运行
echo.
pause
