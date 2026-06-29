@echo off
echo ==========================================
echo Setting up Local Virtual Environment...
echo ==========================================

:: 1. Create the virtual environment folder named 'venv'
python -m venv venv

:: 2. Activate it and install the requirements
call .\venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Local setup complete! To activate this environment in the future, run:
echo .\venv\Scripts\activate
pause