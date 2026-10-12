#!/usr/bin/env bash
# Based on https://github.com/HarishBaki/EU_SCORES_project/blob/main/WRFV4.4/WPS_pipeline.sh

WPS_DIR="$1"

usage () {
    echo "Usage: $0 <WPS_DIR>"
    echo "  WPS_DIR: Directory containing WPS source files"
    exit 1
}

if [ -z "$WPS_DIR" ]; then
    usage
fi

# Copy WPS to current dir
rsync -av --exclude='namelist.wps' --exclude='Vtable' $WPS_DIR/* .
