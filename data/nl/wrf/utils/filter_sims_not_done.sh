#!/usr/bin/env bash
# Print simulations that have not been completed yet
# Usage: filter_sims_not_done.sh sim_dir1 sim_dir2 ...

for sim_dir in $@; do
    if [ -d "$sim_dir" ]; then
        if [ ! -f "$sim_dir/.done" ]; then
            echo $sim_dir
        fi
    fi
done