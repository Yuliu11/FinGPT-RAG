@echo off
REM Financial RAG Agent - 环境初始化脚本 (批处理版本)
REM 一键安装所有 Python 依赖

echo ========================================
echo Financial RAG Agent - 环境初始化
echo ========================================
echo.

REM 检查 Python 是否安装
echo [1/4] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ✗ 错误: 未找到 Python，请先安装 Python 3.10 或更高版本
    pause
    exit /b 1
)
python --version
echo ✓ Python 环境正常
echo.

REM 检查 pip 是否可用
echo [2/4] 检查 pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ✗ 错误: 未找到 pip，请先安装 pip
    pause
    exit /b 1
)
pip --version
echo ✓ pip 正常
echo.

REM 升级 pip
echo [3/4] 升级 pip 到最新版本...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo ⚠ 警告: pip 升级失败，继续安装依赖...
) else (
    echo ✓ pip 升级成功
)
echo.

REM 安装依赖
echo [4/4] 安装项目依赖...
echo 这可能需要几分钟时间，请耐心等待...
echo.

if exist requirements.txt (
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ========================================
        echo ✗ 依赖安装失败，请检查错误信息
        echo ========================================
        echo.
        pause
        exit /b 1
    ) else (
        echo.
        echo ========================================
        echo ✓ 所有依赖安装成功！
        echo ========================================
        echo.
        echo 下一步：
        echo 1. 复制 .env.example 到 .env
        echo 2. 编辑 .env 文件，填入你的 OPENAI_API_KEY
        echo 3. 运行数据导入: python scripts/ingest.py
        echo.
    )
) else (
    echo ✗ 错误: 未找到 requirements.txt 文件
    pause
    exit /b 1
)

pause

