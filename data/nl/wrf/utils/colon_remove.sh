#!/bin/bash
# Windows filesystems cannot handle the colons in met_em or wrfout filenames.
# Rename them. Execute this from workspace root (i.e. one level up)
find . -type f -name "met_em*" | while IFS= read -r f; do
    dir=$(dirname "$f")
    base=$(basename "$f")
    newbase="${base//:/-}"
    if [ "$base" != "$newbase" ]; then
        mv -n -- "$f" "$dir/$newbase"
    fi
done