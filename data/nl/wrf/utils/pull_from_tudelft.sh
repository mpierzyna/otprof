#!/usr/bin/env bash
# Pull processed Cn2 data from TU Delft project drive.
# Execute from workspace root (ie one level above)

# Pass 'CONFIRM' to actually run
if [ "$1" = "CONFIRM" ]; then
    DRYRUN=''
else
    DRYRUN='--dry-run'
fi

rclone copy \
    ${DRYRUN} \
    --progress \
    --transfers=8 \
    --multi-thread-streams=1 \
    --include 'wrfout*cn2.nc' \
    tudelft:staff-umbrella/mlatmosphericmodelling/otprof_rerun/data/nl/wrf .
