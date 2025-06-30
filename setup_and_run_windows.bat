@echo off
setlocal enabledelayedexpansion

:: ChatGPT OCR Application - Windows Setup and Launch Script
:: This script will set up and run the ChatGPT OCR application on Windows

title ChatGPT OCR - Setup and Launch

echo ========================================
echo   ChatGPT OCR Application Setup
echo ========================================
echo.

:: Check if Python is installed
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

python --version
echo Python is installed successfully!
echo.

:: Check if pip is available
echo [2/6] Checking pip installation...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed or not in PATH
    echo.
    echo Please reinstall Python with pip included
    echo.
    pause
    exit /b 1
)

echo pip is available!
echo.

:: Install Python dependencies
echo [3/6] Installing Python dependencies...
echo This may take a few minutes...
echo.

pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    echo.
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully!
echo.

:: Check if .env file exists, if not create it
echo [4/6] Setting up OpenAI API configuration...

if not exist .env (
    echo Creating environment configuration file...
    echo.
    echo You need an OpenAI API key to use this application.
    echo You can get one from: https://platform.openai.com/api-keys
    echo.
    
    set /p OPENAI_KEY="Please enter your OpenAI API key: "
    
    if "!OPENAI_KEY!"=="" (
        echo ERROR: OpenAI API key is required
        echo.
        pause
        exit /b 1
    )
    
    echo OPENAI_API_KEY=!OPENAI_KEY! > .env
    echo.
    echo Environment file created successfully!
) else (
    echo Environment file already exists!
    
    :: Check if the API key is set
    findstr /C:"OPENAI_API_KEY=" .env >nul
    if errorlevel 1 (
        echo.
        echo WARNING: No OpenAI API key found in .env file
        set /p UPDATE_KEY="Would you like to add/update your API key? (y/n): "
        
        if /i "!UPDATE_KEY!"=="y" (
            set /p OPENAI_KEY="Please enter your OpenAI API key: "
            if not "!OPENAI_KEY!"=="" (
                echo OPENAI_API_KEY=!OPENAI_KEY! >> .env
                echo API key added to .env file!
            )
        )
    ) else (
        echo OpenAI API key is configured!
    )
)
echo.

:: Create necessary directories
echo [5/6] Creating necessary directories...
if not exist uploads mkdir uploads
if not exist images mkdir images
echo Directories ready!
echo.

:: Start the application
echo [6/6] Starting ChatGPT OCR Application...
echo.
echo ========================================
echo   Application is starting...
echo ========================================
echo.
echo The application will be available at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the application
echo.

:: Wait a moment and then open browser
timeout /t 3 /nobreak >nul
start http://localhost:5000

:: Start the Flask application
python app.py

:: If we get here, the app has stopped
echo.
echo ========================================
echo   Application has stopped
echo ========================================
echo.
pause 