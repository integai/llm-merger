@echo off

REM Check if Python is installed
where /q python
if %errorlevel% neq 0 (
    echo Python could not be found
    exit /b
)

REM Check if pip is installed
where /q pip
if %errorlevel% neq 0 (
    echo pip could not be found
    exit /b
)

REM Install requirements
pip install argparse torch copy safetensors

REM Run merger.py with arguments
python merger.py --model1 %1 --model2 %2 --output %3 --range %4 %5
