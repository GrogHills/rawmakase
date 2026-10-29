#!/usr/bin/env bash
# Runs a build of RAWmakase in a sandbox of its own: its own HOME, data folder
# and default catalog, and copies of sample photos. A test build then never
# reaches your library, catalogs, session, previews or preferences, and file
# operations (move, rename, trash) only touch the copies.
#
#   scripts/sandbox.sh --sample ~/Pictures/2021-2030 --limit 300
#   scripts/sandbox.sh                  # build and launch the same sandbox again
#   scripts/sandbox.sh --reset --sample DIR --limit 500
#
# The sandbox is $RAWMAKASE_SANDBOX, by default ~/rawmakase-sandbox:
#   home/    HOME (and XDG_*_HOME under it) while the app runs
#   data/    RAWMAKASE_DATA_DIR: the default catalog, session and previews
#   photos/  the copied sample; add this folder to the catalog in the app
#
# The sample folder is only read. Photos (RAW, JPEG, PNG, TIFF, WebP, AVIF,
# JPEG XL) are copied with the files beside them that share their name (XMP
# sidecars, the JPEG of a RAW+JPEG pair, legacy .rawmakase.json edits),
# keeping the folder layout. Hidden files and Lightroom preview folders
# (*.lrdata) are skipped. Copies are clones where the file system supports
# them (APFS, Btrfs, XFS), so a sample on the sandbox's volume takes almost
# no space. Copying again adds only files the sandbox lacks.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/sandbox.sh [--sample DIR] [--limit N] [--reset] [--no-build] [-- APP_ARGS...]

  --sample DIR  copy photos from DIR (read only) into the sandbox
  --limit N     copy only N photos, spread evenly over DIR's sorted paths
  --reset       delete the sandbox first (only one this script made)
  --no-build    launch the existing target/release/rawmakase without building
  APP_ARGS      passed to the app, such as a catalog to open

The sandbox is $RAWMAKASE_SANDBOX, by default ~/rawmakase-sandbox.
EOF
}

die() {
    echo "sandbox: $*" >&2
    exit 1
}

sample=
limit=
reset=0
build=1
while [ $# -gt 0 ]; do
    case $1 in
        --sample) [ $# -ge 2 ] || die "--sample needs a folder"; sample=$2; shift 2 ;;
        --limit) [ $# -ge 2 ] || die "--limit needs a number"; limit=$2; shift 2 ;;
        --reset) reset=1; shift ;;
        --no-build) build=0; shift ;;
        -h | --help) usage; exit 0 ;;
        --) shift; break ;;
        *) usage >&2; exit 2 ;;
    esac
done
app_args=("$@")
case $limit in
    '') ;;
    *[!0-9]* | 0) die "--limit must be a positive number" ;;
esac

repo=$(cd "$(dirname "$0")/.." && pwd -P)
marker=.rawmakase-sandbox

# The absolute path of $1 with symlinks resolved, for a path that may not
# exist yet; nothing is created.
physical() {
    local p=$1 rest=
    case $p in /*) ;; *) p=$PWD/$p ;; esac
    while [ ! -d "$p" ]; do
        rest=/$(basename "$p")$rest
        p=$(dirname "$p")
    done
    printf '%s%s\n' "$(cd "$p" && pwd -P)" "$rest"
}
# Whether path $1 is $2 or inside it.
within() {
    case $1/ in "$2"/*) return 0 ;; *) return 1 ;; esac
}

root=$(physical "${RAWMAKASE_SANDBOX:-$HOME/rawmakase-sandbox}")
[ "$root" != / ] && [ "$root" != "$(physical "$HOME")" ] ||
    die "$root can't be the sandbox; set RAWMAKASE_SANDBOX to a folder of its own"

if [ -n "$sample" ]; then
    [ -d "$sample" ] || die "no folder $sample"
    sample=$(physical "$sample")
    ! within "$root" "$sample" || die "the sandbox $root is inside the sample $sample"
    ! within "$sample" "$root" || die "the sample $sample is inside the sandbox $root"
fi

# Never adopt or delete a folder this script didn't make.
if [ -d "$root" ] && [ ! -e "$root/$marker" ] && [ -n "$(ls -A "$root")" ]; then
    die "$root exists and isn't a sandbox; set RAWMAKASE_SANDBOX to another folder"
fi
if [ "$reset" = 1 ] && [ -e "$root/$marker" ]; then
    echo "Deleting the sandbox $root"
    rm -rf "$root"
fi
mkdir -p "$root/home" "$root/data" "$root/photos"
touch "$root/$marker"

# Copies file $1 to $2 unless $2 exists: a clone where possible, else a copy.
clone() {
    [ -e "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    if [ "$(uname -s)" = Darwin ]; then
        cp -c -p "$1" "$2" 2>/dev/null || cp -p "$1" "$2"
    else
        cp --reflink=auto -p "$1" "$2"
    fi
}
# The device holding $1, on macOS.
device() {
    stat -f %d "$1"
}

if [ -n "$sample" ]; then
    names=()
    for ext in 3fr arw cr2 cr3 crw dcr dng erf fff gpr iiq k25 kdc mef mos mrw nef nrw \
        orf pef raf raw rw2 rwl sr2 srf srw x3f jpg jpeg png tif tiff webp avif jxl; do
        [ ${#names[@]} -eq 0 ] || names+=(-o)
        names+=(-iname "*.$ext")
    done
    list=$(mktemp)
    trap 'rm -f "$list" "$list.all"' EXIT
    find "$sample" -mindepth 1 \( -name '.*' -o -name '*.lrdata' \) -prune -o \
        -type f \( "${names[@]}" \) -print | LC_ALL=C sort >"$list.all"
    total=$(wc -l <"$list.all" | tr -d ' ')
    [ "$total" -gt 0 ] || die "no photos in $sample"
    if [ -n "$limit" ] && [ "$limit" -lt "$total" ]; then
        awk -v n="$limit" -v t="$total" \
            'BEGIN { last = -1 } { k = int((NR - 1) * n / t); if (k != last) { print; last = k } }' \
            "$list.all" >"$list"
    else
        cp "$list.all" "$list"
    fi
    rm -f "$list.all"
    count=$(wc -l <"$list" | tr -d ' ')

    # A clone on the same APFS volume takes no space; anything else is a full copy.
    if [ "$(uname -s)" != Darwin ] || [ "$(device "$sample")" != "$(device "$root")" ]; then
        need=$(tr '\n' '\0' <"$list" | xargs -0 du -k | awk '{ s += $1 } END { print s + 0 }')
        free=$(df -Pk "$root" | awk 'NR == 2 { print $4 }')
        [ "$need" -lt "$((free - 1048576))" ] ||
            die "$count photos need $((need / 1024)) MB and $root has $((free / 1024)) MB free; use --limit, or put RAWMAKASE_SANDBOX on another volume"
    fi

    echo "Copying $count of $total photos from $sample to $root/photos"
    while IFS= read -r photo; do
        rel=${photo#"$sample"/}
        dir=$(dirname "$photo")
        name=$(basename "$photo")
        clone "$photo" "$root/photos/$rel"
        for companion in "$dir/${name%.*}".*; do
            if [ -f "$companion" ]; then
                clone "$companion" "$root/photos/${companion#"$sample"/}"
            fi
        done
    done <"$list"
fi

if [ "$build" = 1 ]; then
    (cd "$repo" && cargo build --release --locked)
fi
bin=$repo/target/release/rawmakase
[ -x "$bin" ] || die "no $bin; build it or drop --no-build"

cat <<EOF
Sandbox:  $root
Data:     $root/data (default catalog Photos.rawmakase)
Photos:   $root/photos (add this folder in the Library)
EOF

export HOME=$root/home
export XDG_DATA_HOME=$HOME/.local/share
export XDG_CONFIG_HOME=$HOME/.config
export XDG_CACHE_HOME=$HOME/.cache
export XDG_STATE_HOME=$HOME/.local/state
export RAWMAKASE_DATA_DIR=$root/data
exec "$bin" ${app_args[@]+"${app_args[@]}"}
