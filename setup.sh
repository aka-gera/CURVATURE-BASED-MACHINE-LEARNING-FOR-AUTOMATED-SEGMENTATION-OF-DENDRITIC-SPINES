#!/bin/bash

set -e
 
if ! command -v brew &>/dev/null; then
  echo "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
  echo "Homebrew already installed."
fi
  
if ! brew list python@3.11 &>/dev/null; then
  echo "Installing Python 3.11..."
  brew install python@3.11
else
  echo "Python 3.11 already installed."
fi
 
echo "Activating environment and installing requirements..."
python3.11 -m venv dsa_venv
source ./dsa_venv/bin/activate

if [ -f requirements.txt ]; then
  pip install --upgrade pip
  pip install -r requirements.txt
  echo "Requirements installed from requirements.txt"
else
  echo "# Add your dependencies here" > requirements.txt
fi

echo "Setup complete! To start working, run: source dsa_venv/bin/activate"

