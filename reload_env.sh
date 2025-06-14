#!/bin/bash

set -e  # Stop on any error
set -u  # Stop on undefined variables

# Configuration
VENV_DIR=".venv"
REQUIRED_PYTHON_VERSION="3.11"
REQUIREMENTS_IN="requirements.in"
REQUIREMENTS_TXT="requirements.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check Python version
print_message "$YELLOW" "🔍 Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "$REQUIRED_PYTHON_VERSION" ]]; then
    print_message "$RED" "❌ Python version $REQUIRED_PYTHON_VERSION is required, but found $PYTHON_VERSION"
    exit 1
fi
print_message "$GREEN" "✅ Python version $PYTHON_VERSION is compatible"

# Cleanup old environment
print_message "$YELLOW" "🧹 Cleaning up old environment (if it exists)..."
if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
    print_message "$GREEN" "✅ Old environment removed"
else
    print_message "$YELLOW" "ℹ️  No existing environment found"
fi

# Remove old requirements.txt if it exists
if [ -f "$REQUIREMENTS_TXT" ]; then
    print_message "$YELLOW" "🧹 Removing old requirements.txt..."
    rm "$REQUIREMENTS_TXT"
fi

# Create new virtual environment
print_message "$YELLOW" "🐍 Creating new virtual environment in '$VENV_DIR'..."
python3 -m venv "$VENV_DIR"

# Activate virtual environment
print_message "$YELLOW" "✅ Environment created. Activating..."
source "$VENV_DIR/bin/activate"

# Upgrade pip and install pip-tools
print_message "$YELLOW" "📦 Upgrading pip and installing pip-tools..."
python -m pip install --upgrade pip
pip install pip-tools

# Compile requirements
print_message "$YELLOW" "📝 Compiling requirements from $REQUIREMENTS_IN..."
pip-compile "$REQUIREMENTS_IN"

# Install dependencies
print_message "$YELLOW" "📦 Installing dependencies from $REQUIREMENTS_TXT..."
pip install -r "$REQUIREMENTS_TXT"

print_message "$GREEN" "✅ Environment setup completed successfully!"
print_message "$YELLOW" "🧪 To activate the environment manually later, use:"
echo "   source $VENV_DIR/bin/activate"

# Verify installation
print_message "$YELLOW" "🔍 Verifying key package installations..."
python -c "import torch; import transformers; import diffusers; import accelerate; print(f'✅ PyTorch {torch.__version__}\n✅ Transformers {transformers.__version__}\n✅ Diffusers {diffusers.__version__}\n✅ Accelerate {accelerate.__version__}')"