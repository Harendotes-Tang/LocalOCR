#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PythonExecutable,
    [Parameter(Mandatory)][string]$ModelsDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [string]$BuildDirectory = (Join-Path (Split-Path -Parent $PSScriptRoot) 'artifacts/dependencies/local-ai-ocr-build'),
    [switch]$InstallBuildDependencies
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
foreach ($path in @($PythonExecutable, $ModelsDirectory, $OutputDirectory, $BuildDirectory)) {
    if (-not [IO.Path]::IsPathFullyQualified($path)) {
        throw '构建 OCR helper 时必须显式指定绝对路径；应用运行时不查找系统 Python。'
    }
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw '指定的构建 Python 不存在。' }
if (-not (Test-Path -LiteralPath $ModelsDirectory -PathType Container)) { throw '本地模型目录不存在。' }
if (Test-Path -LiteralPath $OutputDirectory) { throw '输出目录已经存在；请使用新的输出目录以保留先前产物。' }
if ($InstallBuildDependencies) {
    & $PythonExecutable -m pip install --requirement (Join-Path $repoRoot 'tools/LocalAiOcr/requirements-build.txt')
    if ($LASTEXITCODE -ne 0) { throw '安装固定构建依赖失败。' }
}
& $PythonExecutable (Join-Path $repoRoot 'tools/LocalAiOcr/build_worker.py') `
    --models $ModelsDirectory --output $OutputDirectory --build-dir $BuildDirectory
if ($LASTEXITCODE -ne 0) { throw '本地 AI OCR 自包含 helper 构建或验证失败。' }
