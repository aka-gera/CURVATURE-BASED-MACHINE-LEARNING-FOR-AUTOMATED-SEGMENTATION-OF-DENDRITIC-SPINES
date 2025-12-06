# Ensure script stops on errors
$ErrorActionPreference = "Stop"

Write-Host "Checking Python installation..."

# Check if Python 3.11 exists
$python = (Get-Command python 2>$null)
if ($python -eq $null -or -not ((python --version) -match "3\.11")) {
    Write-Host "Installing Python 3.11..."

    winget install -e --id Python.Python.3.11
} else {
    Write-Host "Python 3.11 already installed."
}

# Create a virtual environment
Write-Host "Creating virtual environment..."
python -m venv dsa_venv

# Activate venv
Write-Host "Activating environment..."
./dsa_venv/Scripts/Activate.ps1

# Install requirements if available
if (Test-Path "requirements_win.txt") {
    Write-Host "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements_win.txt
    Write-Host "Requirements installed!"
} else {
    Write-Host "Creating requirements.txt..."
    "# Add your dependencies here" | Out-File requirements_win.txt
}

Write-Host "`nSetup complete! To start working: `n`n    ./dsa_venv/Scripts/Activate.ps1`n"
