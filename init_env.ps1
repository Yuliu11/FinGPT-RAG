# Financial RAG Agent - 环境初始化脚本
# 一键安装所有 Python 依赖

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Financial RAG Agent - 环境初始化" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python 是否安装
Write-Host "[1/4] 检查 Python 环境..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ 找到 Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ 错误: 未找到 Python，请先安装 Python 3.10 或更高版本" -ForegroundColor Red
    exit 1
}

# 检查 pip 是否可用
Write-Host "[2/4] 检查 pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✓ 找到 pip: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ 错误: 未找到 pip，请先安装 pip" -ForegroundColor Red
    exit 1
}

# 升级 pip
Write-Host "[3/4] 升级 pip 到最新版本..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ pip 升级成功" -ForegroundColor Green
} else {
    Write-Host "⚠ 警告: pip 升级失败，继续安装依赖..." -ForegroundColor Yellow
}

# 安装依赖
Write-Host "[4/4] 安装项目依赖..." -ForegroundColor Yellow
Write-Host "这可能需要几分钟时间，请耐心等待..." -ForegroundColor Gray
Write-Host ""

if (Test-Path "requirements.txt") {
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✓ 所有依赖安装成功！" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "下一步：" -ForegroundColor Cyan
        Write-Host "1. 复制 .env.example 到 .env" -ForegroundColor White
        Write-Host "2. 编辑 .env 文件，填入你的 OPENAI_API_KEY" -ForegroundColor White
        Write-Host "3. 运行数据导入: python scripts/ingest.py" -ForegroundColor White
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Red
        Write-Host "✗ 依赖安装失败，请检查错误信息" -ForegroundColor Red
        Write-Host "========================================" -ForegroundColor Red
        Write-Host ""
        exit 1
    }
} else {
    Write-Host "✗ 错误: 未找到 requirements.txt 文件" -ForegroundColor Red
    exit 1
}

