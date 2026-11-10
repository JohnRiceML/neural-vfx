# Configure + build the native runtime with MSVC + Ninja. OWNER: lead.
#   powershell -File native/build.ps1 [-Config Release|RelWithDebInfo|Debug] [-Clean]
#   -BuildDir <name>: build into native/<name> instead of native/build (lets two engineers build at once;
#                     only names starting with "build" are git-ignored)
param([string]$Config = "Release", [switch]$Clean, [string]$BuildDir = "build")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$build = Join-Path $root $BuildDir
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "MSVC C++ build tools not found" }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"

$cmd = "`"$vcvars`" >nul && cmake -S `"$root`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=$Config -DCMAKE_CXX_COMPILER=cl && cmake --build `"$build`""
cmd /c $cmd
exit $LASTEXITCODE
