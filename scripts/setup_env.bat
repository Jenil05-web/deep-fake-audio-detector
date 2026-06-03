@echo off
REM Windows setup script for deepfake-audio-detector

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo Running pre-commit setup...
pre-commit install

echo.
echo ===================================
echo Setup completed successfully!
echo ===================================
echo.
echo To activate the environment in future sessions, run:
echo   venv\Scripts\activate.bat
echo.
echo To run tests:
echo   pytest
echo.
echo To start training:
echo   python -m src.training.train
echo.

pause
