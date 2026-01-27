#!/bin/bash

# Configuration
# Replace 'group4-node' with the IP address if the hostname isn't in your /etc/hosts
REMOTE_USER="cc"
REMOTE_HOST="chameleon"
REMOTE_DIR="~/"  # Downloads from the home directory where the ls -l was taken
LOCAL_DIR="./group4_data_backup"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting download from ${REMOTE_HOST}...${NC}"
echo "Target: ${REMOTE_DIR}"
echo "Destination: ${LOCAL_DIR}"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# RSYNC Command
# -a: archive mode (preserves permissions, timestamps, symbolic links, and works recursively)
# -v: verbose
# -z: compress file data during the transfer
# -P: same as --partial --progress (keeps partially transferred files and shows progress bar)
rsync -avzP "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_DIR"

echo -e "${GREEN}Download complete! Files are located in ${LOCAL_DIR}${NC}"
