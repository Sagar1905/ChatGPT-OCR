# ChatGPT OCR Application - Windows PowerShell Setup and Launch Script
# This script will set up and run the ChatGPT OCR application on Windows

param(
    [switch]$SkipBrowser,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
ChatGPT OCR Application Setup Script

Usage: .\setup_and_run_windows.ps1 [OPTIONS]

Options:
  -SkipBrowser    Don't automatically open the web browser
  -Help           Show this help message

This script will:
1. Check Python installation
2. Install required dependencies
3. Set up OpenAI API key configuration
4. Create necessary directories
5. Start the application
6. Open your web browser (unless -SkipBrowser is used)
"@
    exit 0
}

# Set window title
$Host.UI.RawUI.WindowTitle = "ChatGPT OCR - Setup and Launch"

# Function to write colored output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Function to check if a command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput "   ChatGPT OCR Application Setup" "Cyan"
Write-ColorOutput "========================================" "Cyan"
Write-Host ""

try {
    # Step 1: Check Python installation
    Write-ColorOutput "[1/6] Checking Python installation..." "Yellow"
    
    if (-not (Test-Command "python")) {
        Write-ColorOutput "ERROR: Python is not installed or not in PATH" "Red"
        Write-Host ""
        Write-ColorOutput "Please install Python from https://www.python.org/downloads/" "White"
        Write-ColorOutput "Make sure to check 'Add Python to PATH' during installation" "White"
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    $pythonVersion = python --version 2>&1
    Write-ColorOutput $pythonVersion "Green"
    Write-ColorOutput "Python is installed successfully!" "Green"
    Write-Host ""

    # Step 2: Check pip installation
    Write-ColorOutput "[2/6] Checking pip installation..." "Yellow"
    
    if (-not (Test-Command "pip")) {
        Write-ColorOutput "ERROR: pip is not installed or not in PATH" "Red"
        Write-Host ""
        Write-ColorOutput "Please reinstall Python with pip included" "White"
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Write-ColorOutput "pip is available!" "Green"
    Write-Host ""

    # Step 3: Install dependencies
    Write-ColorOutput "[3/6] Installing Python dependencies..." "Yellow"
    Write-ColorOutput "This may take a few minutes..." "White"
    Write-Host ""
    
    $installResult = & pip install -r requirements.txt 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-ColorOutput "ERROR: Failed to install dependencies" "Red"
        Write-ColorOutput "Error details: $installResult" "Red"
        Write-ColorOutput "Please check your internet connection and try again" "White"
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Write-Host ""
    Write-ColorOutput "Dependencies installed successfully!" "Green"
    Write-Host ""

    # Step 4: Setup OpenAI API configuration
    Write-ColorOutput "[4/6] Setting up OpenAI API configuration..." "Yellow"
    
    if (-not (Test-Path ".env")) {
        Write-ColorOutput "Creating environment configuration file..." "White"
        Write-Host ""
        Write-ColorOutput "You need an OpenAI API key to use this application." "White"
        Write-ColorOutput "You can get one from: https://platform.openai.com/api-keys" "Cyan"
        Write-Host ""
        
        $openaiKey = Read-Host "Please enter your OpenAI API key"
        
        if ([string]::IsNullOrWhiteSpace($openaiKey)) {
            Write-ColorOutput "ERROR: OpenAI API key is required" "Red"
            Write-Host ""
            Read-Host "Press Enter to exit"
            exit 1
        }
        
        "OPENAI_API_KEY=$openaiKey" | Out-File -FilePath ".env" -Encoding UTF8
        Write-Host ""
        Write-ColorOutput "Environment file created successfully!" "Green"
    }
    else {
        Write-ColorOutput "Environment file already exists!" "Green"
        
        # Check if API key is set
        $envContent = Get-Content ".env" -ErrorAction SilentlyContinue
        $hasApiKey = $envContent -match "OPENAI_API_KEY="
        
        if (-not $hasApiKey) {
            Write-Host ""
            Write-ColorOutput "WARNING: No OpenAI API key found in .env file" "Yellow"
            $updateKey = Read-Host "Would you like to add/update your API key? (y/n)"
            
            if ($updateKey -eq "y" -or $updateKey -eq "Y") {
                $openaiKey = Read-Host "Please enter your OpenAI API key"
                if (-not [string]::IsNullOrWhiteSpace($openaiKey)) {
                    "OPENAI_API_KEY=$openaiKey" | Add-Content -Path ".env"
                    Write-ColorOutput "API key added to .env file!" "Green"
                }
            }
        }
        else {
            Write-ColorOutput "OpenAI API key is configured!" "Green"
        }
    }
    Write-Host ""

    # Step 5: Create necessary directories
    Write-ColorOutput "[5/6] Creating necessary directories..." "Yellow"
    
    @("uploads", "images") | ForEach-Object {
        if (-not (Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ | Out-Null
        }
    }
    
    Write-ColorOutput "Directories ready!" "Green"
    Write-Host ""

    # Step 6: Start the application
    Write-ColorOutput "[6/6] Starting ChatGPT OCR Application..." "Yellow"
    Write-Host ""
    Write-ColorOutput "========================================" "Cyan"
    Write-ColorOutput "   Application is starting..." "Cyan"
    Write-ColorOutput "========================================" "Cyan"
    Write-Host ""
    Write-ColorOutput "The application will be available at:" "White"
    Write-ColorOutput "  http://localhost:5000" "Cyan"
    Write-Host ""
    Write-ColorOutput "Press Ctrl+C to stop the application" "White"
    Write-Host ""

    # Open browser unless skipped
    if (-not $SkipBrowser) {
        Write-ColorOutput "Opening web browser in 3 seconds..." "White"
        Start-Sleep -Seconds 3
        Start-Process "http://localhost:5000"
    }

    # Start the Flask application
    & python app.py
}
catch {
    Write-ColorOutput "An unexpected error occurred: $($_.Exception.Message)" "Red"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# If we get here, the app has stopped
Write-Host ""
Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput "   Application has stopped" "Cyan"
Write-ColorOutput "========================================" "Cyan"
Write-Host ""
Read-Host "Press Enter to exit" 