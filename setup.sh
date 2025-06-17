#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create workspace directory if it doesn't exist
WORKSPACE_DIR="/workspace"

# Create pip cache directory in workspace
PIP_CACHE_DIR="/workspace/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

# If running as root, first install sudo if needed
if [ "$EUID" -eq 0 ]; then
    # Check if sudo is installed
    if ! command -v sudo &> /dev/null; then
        echo -e "${YELLOW}Installing sudo...${NC}"
        apt-get update
        apt-get install -y sudo
    fi

    echo -e "${YELLOW}Running as root. Setting up user...${NC}"
    
    # Check if diffusionuser already exists
    if id "diffusionuser" &>/dev/null; then
        echo -e "${YELLOW}User diffusionuser already exists${NC}"
        # Ensure user has sudo access
        if ! groups diffusionuser | grep -q sudo; then
            echo -e "${YELLOW}Adding diffusionuser to sudo group...${NC}"
            usermod -aG sudo diffusionuser
        fi
    else
        echo -e "${YELLOW}Creating user diffusionuser...${NC}"
        # Create user with home directory and shell
        useradd -m -s /bin/bash diffusionuser
        # Set password (you'll be prompted to enter it)
        passwd diffusionuser
        # Add to sudo group
        usermod -aG sudo diffusionuser
    fi

    # Configure sudoers to allow passwordless sudo for diffusionuser
    echo -e "${YELLOW}Configuring sudo access...${NC}"
    echo "diffusionuser ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/diffusionuser
    chmod 440 /etc/sudoers.d/diffusionuser

    # Set proper ownership of workspace directory
    echo -e "${YELLOW}Setting workspace permissions...${NC}"
    chown -R diffusionuser:diffusionuser "$WORKSPACE_DIR"
    
    echo -e "${YELLOW}Switching to diffusionuser...${NC}"
    # Switch to the new user and run the script with a proper terminal
    exec su - diffusionuser -c "cd $(pwd) && bash -c '$0 $@'"
    exit 0
fi

echo -e "${GREEN}Starting DiffusionAPI setup...${NC}"

# Check if running as root (this check is now after user creation)
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}Please do not run this script as root${NC}"
    exit 1
fi

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
sudo -n apt-get update
sudo -n apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git

# Function to check available disk space
check_disk_space() {
    local required_space=$1  # in GB
    local mount_point=$2    # directory to check
    local available_space=$(df -BG "$mount_point" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "$available_space" -lt "$required_space" ]; then
        echo -e "${RED}Error: Not enough disk space in $mount_point. Required: ${required_space}GB, Available: ${available_space}GB${NC}"
        echo -e "${YELLOW}Please free up some space or mount a larger volume and try again.${NC}"
        exit 1
    fi
}

# Change to workspace directory
cd "$WORKSPACE_DIR"

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Initializing git repository...${NC}"
    git init
    git remote add origin https://github.com/mozartdiniz/DiffusionAPI.git
    git fetch
    git checkout -b main
    git reset --hard origin/main
else
    echo -e "${YELLOW}Repository already exists, updating...${NC}"
    git pull
fi

# Create and activate virtual environment
echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
python3.11 -m venv .venv
source .venv/bin/activate

# Configure pip to use workspace for cache
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export TMPDIR="/workspace/tmp"
mkdir -p "$TMPDIR"

# Upgrade pip
python -m pip install --upgrade pip

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --cache-dir "$PIP_CACHE_DIR" -r requirements.txt

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p stable_diffusion/models
mkdir -p stable_diffusion/loras
mkdir -p stable_diffusion/upscalers
mkdir -p outputs
mkdir -p queue

# Download models if not present
echo -e "${YELLOW}Checking models...${NC}"
# Check for 16GB of free space (models are about 7-8GB each)
check_disk_space 16 "/workspace"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "DEST_DIR=stable_diffusion/models" > .env
fi

# Download amanatsu-illustrious-v11-sdxl
if [ ! -d "stable_diffusion/models/models--John6666__amanatsu-illustrious-v11-sdxl" ]; then
    echo -e "${YELLOW}Downloading amanatsu-illustrious-v11-sdxl model...${NC}"
    python download_model.py --model "John6666/amanatsu-illustrious-v11-sdxl"
else
    echo -e "${GREEN}amanatsu-illustrious-v11-sdxl model already exists${NC}"
fi

# Download ilustmix-v6-sdxl
if [ ! -d "stable_diffusion/models/models--John6666--ilustmix-v6-sdxl" ]; then
    echo -e "${YELLOW}Downloading ilustmix-v6-sdxl model...${NC}"
    python download_model.py --model "John6666/ilustmix-v6-sdxl"
else
    echo -e "${GREEN}ilustmix-v6-sdxl model already exists${NC}"
fi

# Download digiplay/ChikMix_V3  
if [ ! -d "stable_diffusion/models/models--digiplay--ChikMix_V3" ]; then
    echo -e "${YELLOW}Downloading digiplay/ChikMix_V3 model...${NC}"
    python download_model.py --model "digiplay/ChikMix_V3"
else
    echo -e "${GREEN}digiplay/ChikMix_V3 model already exists${NC}"
fi

# Download mirroring/pastel-mix   
if [ ! -d "stable_diffusion/models/models--mirroring--pastel-mix" ]; then
    echo -e "${YELLOW}Downloading mirroring/pastel-mix model...${NC}"
    python download_model.py --model "mirroring/pastel-mix"
else
    echo -e "${GREEN}mirroring/pastel-mix model already exists${NC}"
    fi

# Download stablediffusionapi/realcartoon-xl-v6 
if [ ! -d "stable_diffusion/models/models--stablediffusionapi--realcartoon-xl-v6" ]; then
    echo -e "${YELLOW}Downloading stablediffusionapi/realcartoon-xl-v6 model...${NC}"
    python download_model.py --model "stablediffusionapi/realcartoon-xl-v6"
else
    echo -e "${GREEN}stablediffusionapi/realcartoon-xl-v6 model already exists${NC}"
fi

# Download Meina/MeinaMix_V11   
if [ ! -d "stable_diffusion/models/models--Meina--MeinaMix_V11" ]; then
    echo -e "${YELLOW}Downloading Meina/MeinaMix_V11 model...${NC}"
    python download_model.py --model "Meina/MeinaMix_V11"
else
    echo -e "${GREEN}Meina/MeinaMix_V11 model already exists${NC}"
fi

# Download gsdf/Counterfeit-V2.5
if [ ! -d "stable_diffusion/models/models--gsdf--Counterfeit-V2.5" ]; then
    echo -e "${YELLOW}Downloading gsdf/Counterfeit-V2.5 model...${NC}"
    python download_model.py --model "gsdf/Counterfeit-V2.5"
else
    echo -e "${GREEN}gsdf/Counterfeit-V2.5 model already exists${NC}"
fi


echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}To start the server:${NC}"
echo "1. Activate the virtual environment: source .venv/bin/activate"
echo "2. Run the server: python -m diffusionapi.main"
echo -e "${YELLOW}The API will be available at http://localhost:8000${NC}"

# Instructions in Portuguese
echo -e "\n${GREEN}Instruções em Português:${NC}"
echo -e "${YELLOW}Para iniciar o servidor:${NC}"
echo "1. Ative o ambiente virtual: source .venv/bin/activate"
echo "2. Execute o servidor: python -m diffusionapi.main"
echo -e "${YELLOW}A API estará disponível em http://localhost:8000${NC}" 