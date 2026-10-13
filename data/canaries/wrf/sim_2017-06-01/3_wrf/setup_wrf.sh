#!/bin/bash
# Script to setup WRF in the current working directory

# Parse arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <MET_EM_DIR> <WRF_TEMPLATE_DIR>"
    echo "  MET_EM_DIR: Directory with met_em* files for this simulation"
    echo "  WRF_TEMPLATE_DIR: Directory with main and run dirs of compiled WRF"
    exit 1
fi

MET_EM_DIR=$1
WRF_TEMPLATE_DIR=$2

# Check that directories can be accessed
if [ ! -d $MET_EM_DIR ]; then
    echo "Error: MET_EM_DIR '$1' does not exist or is not a directory."
    exit 1
fi
if [ ! -d $WRF_TEMPLATE_DIR ]; then
    echo "Error: WRF_TEMPLATE_DIR '$2' does not exist or is not a directory."
    exit 1
fi

# Copy files from WRF/run directory
echo "Copying files from $WRF_TEMPLATE_DIR/run directory..."
rsync -av --exclude='namelist.input' --exclude='myoutfields.txt' --exclude='*.exe' --exclude='wrfout*' --exclude='wrfrst*' --exclude='rsl*' $WRF_TEMPLATE_DIR/run/* .

# Link .exe files from WRF/main directory
echo "Linking .exe files from $WRF_TEMPLATE_DIR/main directory..."
ln -sf $WRF_TEMPLATE_DIR/main/*.exe .

# Link met_em* files
echo "Linking met_em* files from $MET_EM_DIR directory..."
ln -sf $MET_EM_DIR/met_em*.nc .

echo "WRF setup successfully!"
