#!/bin/bash

set -e

if ! command -v brew &>/dev/null; then
  echo "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  echo "Homebrew already installed."
 fi

if ! brew list python@3.9 &>/dev/null; then
  echo "Installing Python 3.9..."
  brew install python@3.9
else
  echo "Python 3.9 already installed."
fi
  
if [ -f ../dsa_venv/bin/activate ]; then
  echo "Activating virtual environment..."
  source ../dsa_venv/bin/activate
else
  echo "Virtual environment not found. Creating one..."
  python3 -m venv ../dsa_venv
  source ../dsa_venv/bin/activate
fi 

if [ -f  requirements.txt ]; then
  pip install --upgrade pip
  pip install -r  requirements.txt
  echo "Requirements installed from  requirements.txt"
else 
  echo "# Add your dependencies here" > requirements.txt
fi

echo "Setup complete! To start working, run: source dsa_venv/bin/activate"
