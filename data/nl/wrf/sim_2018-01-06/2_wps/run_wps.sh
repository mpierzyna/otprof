#!/usr/bin/env bash
# Based on https://github.com/HarishBaki/EU_SCORES_project/blob/main/WRFV4.4/WPS_pipeline.sh

FORCING_DIR="$1"

usage () {
    echo "Usage: $0 <FORCING_DIR>"
    echo "  FORCING_DIR: Directory containing forcing data (CERRA and ERA5)"
    exit 1
}

if [ -z "$FORCING_DIR" ]; then
    usage
fi

# Run geogrid, if not done, yet.
if [ ! -f geo_em.d01.nc ]; then
    ln -sf namelist.wps.CERRA namelist.wps
    ./geogrid.exe || exit 2
fi

# Process CERRA data, if not done, yet.
if [[ $(find . -name 'CERRA:*' | wc -l) -eq 0 ]]; then
    rm GRIBFILE*  # delete potentially leftover GRIBFILES
    find $FORCING_DIR -name 'CERRA*.grb' | xargs ./link_grib.csh  # link CERRA files
    ln -sf Vtable.CERRA Vtable  # enable CERRA Vtable
    ln -sf namelist.wps.CERRA namelist.wps  # enable CERRA namelist
    ./ungrib.exe || exit 3
    mv ungrib.log ungrib_CERRA.log
fi

# Process ERA5 data, if not done, yet.
if [[ $(find . -name 'ERA5:*' | wc -l) -eq 0 ]]; then
    rm GRIBFILE*  # delete potentially leftover GRIBFILES
    find $FORCING_DIR -name 'ERA5*.grb' | xargs ./link_grib.csh  # link ERA5 files
    ln -sf ungrib/Variable_Tables/Vtable.ERA-interim.pl Vtable  # enable ERA5 Vtable
    ln -sf namelist.wps.ERA5 namelist.wps  # enable ERA5 namelist
    ./ungrib.exe || exit 4
    mv ungrib.log ungrib_ERA5.log
fi

# Run metgrid and delete intermediate files after completion
if [[ $(ls met_em* | wc -l) -eq 0 ]]; then
    ./metgrid.exe || exit 5
    rm -r CERRA:*
    rm -r ERA5:*
fi
