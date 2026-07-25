#!/usr/bin/env bash
# Print simulations where WPS has not been completed

for sim_dir in $@; do
    wps_dir="$sim_dir/2_wps"

    # Doesn't exist
    if [ ! -d "$wps_dir" ]; then
        echo $sim_dir
        continue
    fi

    # If symlink, WPS crashed
    if [[ -h "$wps_dir" ]]; then
        echo $sim_dir
        continue
    fi

    # If metgrid.log.0000 doesn't exist, WPS crashed
    metgrid_log="$wps_dir/metgrid.log.0000"
    if [ ! -f "$metgrid_log" ]; then
        echo $sim_dir
        continue
    fi

    # If metgrid.log.0000 doesn't contain "Successful completion", WPS crashed
    if ! grep -q "Successful completion" "$metgrid_log"; then
        echo $sim_dir
        continue
    fi
done
