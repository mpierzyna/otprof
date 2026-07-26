#!/usr/bin/env bash
# Execute from workspace root (ie one level above)

# Pass 'CONFIRM' to actually run
if [ "$1" = "CONFIRM" ]; then
    DRYRUN=''
else
    DRYRUN='--dry-run'
fi

# Due to storage limitations, data is partially in project folder...
rclone copy ${DRYRUN} --progress --transfers=8 --include 'met_em*' --include 'wrfout*cn2.nc' snellius:/home/mpierzyna/prjs1785/otprof/data/nl/wrf .
# ...and scratch (no met_em on scratch)
rclone copy ${DRYRUN} --progress --transfers=8 --include 'wrfout*cn2.nc' snellius:/scratch-shared/mpierzyna/ .
