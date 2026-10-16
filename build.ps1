[CmdletBinding()]
param(
    [string]$PythonExe,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot

function Invoke-Python {
    param([string[]]$PythonArguments)
    & $script:PythonExe -X utf8 @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE): $($PythonArguments -join ' ')"
    }
}

if (-not $PythonExe) {
    $PythonExe = Join-Path $projectRoot '.venv-build\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        if ($SkipInstall) {
            throw 'No build environment exists. Run without -SkipInstall or pass -PythonExe.'
        }
        # Isolate build tools from the user's global Python installation.
        & py -3.12 -X utf8 -m venv (Join-Path $projectRoot '.venv-build')
        if ($LASTEXITCODE -ne 0) { throw 'Creating the Python 3.12 build environment failed.' }
    }
}

# Resolve an explicit relative executable before entering the project directory.
if (Test-Path -LiteralPath $PythonExe) {
    $PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
}

Push-Location -LiteralPath $projectRoot
try {
    Invoke-Python -PythonArguments @('-c', "import sys; assert sys.platform == 'win32', 'Build on Windows'; assert sys.maxsize > 2**32, '64-bit Python is required'")
    if (-not $SkipInstall) {
        Invoke-Python -PythonArguments @('-m', 'pip', 'install', '-r', 'requirements-build.txt')
    }
    Invoke-Python -PythonArguments @('-m', 'PyInstaller', '--noconfirm', '--clean', 'meeting_record.spec')

    $bundle = Join-Path $projectRoot 'dist\MeetingRecord'
    $executable = Join-Path $bundle 'MeetingRecord.exe'
    if (-not (Test-Path -LiteralPath $executable)) {
        throw "Build did not produce $executable"
    }
    Invoke-Python -PythonArguments @('-m', 'unittest', 'test_packaging', '-v')
    Copy-Item -LiteralPath (Join-Path $projectRoot 'README.md') -Destination $bundle
    & $PythonExe -X utf8 -m pip freeze | Set-Content -LiteralPath (Join-Path $bundle 'BUILD-DEPENDENCIES.txt') -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw 'Recording build dependencies failed.' }
    $archive = Join-Path $projectRoot 'dist\MeetingRecord-windows-x64.zip'
    Compress-Archive -LiteralPath $bundle -DestinationPath $archive -Force
    Write-Host "Executable: $executable"
    Write-Host "Distribution: $archive"
}
finally {
    Pop-Location
}
