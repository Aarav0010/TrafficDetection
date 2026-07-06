@echo off
echo ============================================
echo  Traffic Detection C++ Pipeline - Builder
echo ============================================

:: Find Visual Studio 2022
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "tokens=*" %%i in ('"%VSWHERE%" -latest -property installationPath 2^>nul') do set "VS_PATH=%%i"

if not defined VS_PATH (
    echo [ERROR] Visual Studio not found!
    pause
    exit /b 1
)

echo [INFO] Found Visual Studio at: %VS_PATH%

:: Set up MSVC environment
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

:: Create build directory
if not exist build mkdir build
cd build

:: Configure with CMake (MSVC generator)
echo.
echo [INFO] Configuring with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64

:: Build Release
echo.
echo [INFO] Building Release...
cmake --build . --config Release

echo.
echo ============================================
echo  Build complete!
echo  Run: build\Release\traffic_detection.exe
echo ============================================

:: Add Python venv CUDA toolkit DLLs to PATH so it can run on GPU
set PATH=F:\TrafficDetection\venv\Lib\site-packages\torch\lib;%PATH%

pause
