param(
    [string]$BuildPython,
    [string]$Output,
    [string]$BuildDirectory,
    [string]$Notices
)
$ErrorActionPreference = 'Stop'
$repository = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if (-not $BuildPython) { $BuildPython = Join-Path $repository 'artifacts/runtime-build-env/Scripts/python.exe' }
if (-not [System.IO.Path]::IsPathRooted($BuildPython)) { $BuildPython = Join-Path $repository $BuildPython }
if (-not (Test-Path -LiteralPath $BuildPython -PathType Leaf)) { throw 'Prepare the isolated pinned build environment first; this script never installs/downloads packages.' }
$versionProbe = 'import sys,json; sys.path.insert(0,sys.argv[1]); from localocr.version import artifact_names; print(json.dumps(artifact_names()))'
$namesJson = & $BuildPython -I -c $versionProbe $repository
if ($LASTEXITCODE -ne 0) { throw 'Cannot load the central software artifact names.' }
$names = $namesJson | ConvertFrom-Json
if (-not $Output) { $Output = Join-Path (Join-Path $repository 'artifacts') $names.runtime_dir }
if (-not $BuildDirectory) { $BuildDirectory = Join-Path (Join-Path $repository 'artifacts') $names.build_dir }
if (-not [System.IO.Path]::IsPathRooted($Output)) { $Output = Join-Path $repository $Output }
if (-not [System.IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory = Join-Path $repository $BuildDirectory }
if (-not $Notices -and (Test-Path -LiteralPath (Join-Path $repository 'build-notices') -PathType Container)) { $Notices = Join-Path $repository 'build-notices' }
if ($Notices -and -not [System.IO.Path]::IsPathRooted($Notices)) { $Notices = Join-Path $repository $Notices }
$buildArguments = @('--output', $Output, '--build-dir', $BuildDirectory)
if ($Notices) { $buildArguments += @('--notices', $Notices) }
$buildEntry = 'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); sys.argv=["build_runtime",*sys.argv[1:]]; runpy.run_module("tools.LocalAiOcr.build_runtime",run_name="__main__")'
Push-Location $repository
try {
    & $BuildPython -I -c $buildEntry $repository @buildArguments
    if ($LASTEXITCODE -ne 0) { throw "Runtime build failed: $LASTEXITCODE" }
} finally { Pop-Location }
