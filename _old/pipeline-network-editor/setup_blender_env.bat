@echo off
echo ==========================================
echo Setting up Blender Internal Python...
echo ==========================================

:: Define the path to Blender's Python (adjust if installing on a different version)
set BLENDER_PYTHON="C:\Program Files\Blender Foundation\Blender 5.1\5.1\python\bin\python.exe"

:: Check if the executable exists
if not exist %BLENDER_PYTHON% (
    echo [ERROR] Blender Python not found at %BLENDER_PYTHON%
    pause
    exit /b
)

:: Install the requirements directly into Blender (-s ignores user AppData)
%BLENDER_PYTHON% -s -m pip install --upgrade pip
%BLENDER_PYTHON% -s -m pip install -r requirements-blender.txt

echo.
echo Blender setup complete!
pause