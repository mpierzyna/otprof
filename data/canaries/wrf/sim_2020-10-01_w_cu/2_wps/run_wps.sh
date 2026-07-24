#!/usr/bin/env bash
# Based on https://github.com/HarishBaki/EU_SCORES_project/blob/main/WRFV4.4/WPS_pipeline.sh

FORCING_DIR="$1"

usage () {
    echo "Usage: $0 <FORCING_DIR>"
    echo "  FORCING_DIR: Directory containing forcing data (CERRA and ERA5)"
    exit 1
}

check_log () {
    local log_file="$1"
    if ! grep -q "Successful completion" "$log_file"; then
        echo "$log_file does not contain 'Successful completion'. Exiting." 1>&2
        exit 99
    fi
}

log_ok () {
    # Gate re-runs on the step's own success message rather than on the presence of its output files: a step that
    # crashed part-way leaves partial artifacts (some intermediate files, some met_em) that a file-existence check
    # mistakes for a finished step and wrongly skips. The WPS tools don't set a non-zero exit code on failure, so
    # the log is the authoritative "did this complete" signal.
    local log_file="$1"
    [ -f "$log_file" ] && grep -q "Successful completion" "$log_file"
}

if [ -z "$FORCING_DIR" ]; then
    usage
fi

# Run geogrid, unless it already completed successfully.
if ! log_ok geogrid.log; then
    ln -sf namelist.wps.CERRA namelist.wps
    ./geogrid.exe
    check_log geogrid.log  # check for successful completion because geogrib doesn't exit with error code on failure
fi

# Process CERRA data, unless ungrib already completed successfully for it.
if ! log_ok ungrib_CERRA.log; then
    rm -f GRIBFILE*  # delete potentially leftover GRIBFILES
    find $FORCING_DIR -name 'CERRA*.grb' | xargs ./link_grib.csh  # link CERRA files
    ln -sf Vtable.CERRA Vtable  # enable CERRA Vtable
    ln -sf namelist.wps.CERRA namelist.wps  # enable CERRA namelist
    ./ungrib.exe
    mv ungrib.log ungrib_CERRA.log
    check_log ungrib_CERRA.log  # check for successful completion because ungrib doesn't exit with error code on failure
fi

# Process ERA5 data, unless ungrib already completed successfully for it.
if ! log_ok ungrib_ERA5.log; then
    rm -f GRIBFILE*  # delete potentially leftover GRIBFILES
    find $FORCING_DIR -name 'ERA5*.grb' | xargs ./link_grib.csh  # link ERA5 files
    ln -sf ungrib/Variable_Tables/Vtable.ERA-interim.pl Vtable  # enable ERA5 Vtable
    ln -sf namelist.wps.ERA5 namelist.wps  # enable ERA5 namelist
    ./ungrib.exe
    mv ungrib.log ungrib_ERA5.log
    check_log ungrib_ERA5.log  # check for successful completion because ungrib doesn't exit with error code on failure
fi

# Run metgrid and delete intermediate files after completion, unless metgrid already completed successfully.
if ! log_ok metgrid.log; then
    ./metgrid.exe
    check_log metgrid.log  # check for successful completion because metgrid doesn't exit with error code on failure
    rm -r CERRA:*
    rm -r ERA5:*
fi
