#!/bin/bash
#
# Call this if colons were removed with colon_remove.sh
#
# Recursively finds met_em.* files (in all subdirectories) whose time
# portion uses '_' or '-' as separators (e.g. from a Windows share) and
# renames them back to the standard WRF colon format:
#   met_em.d01.2017-02-09_12_00_00.nc  -> met_em.d01.2017-02-09_12:00:00.nc
#   met_em.d01.2017-02-09_12-00-00.nc  -> met_em.d01.2017-02-09_12:00:00.nc
#
# Usage:
#   ./fix_met_em_colons.sh [start_dir]
#
# If start_dir is omitted, the current directory is used.

set -euo pipefail

start_dir="${1:-.}"

find "$start_dir" -type f -name 'met_em.*.nc' | while IFS= read -r f; do
    dir=$(dirname "$f")
    base=$(basename "$f")
    newbase=$(echo "$base" | sed -E 's/_([0-9]{2})[-_]([0-9]{2})[-_]([0-9]{2})\.nc$/_\1:\2:\3.nc/')

    if [ "$base" != "$newbase" ]; then
        mv -- "$f" "$dir/$newbase"
        echo "renamed: $f -> $dir/$newbase"
    fi
done