#!/usr/bin/env bash
# Extracts COUNT frames evenly spread over a video, named in capture order, as PNGs.
# With --cutout, also cuts the background out of each frame (macOS 14+, see cutout.swift),
# and drops frames where no foreground is found.
#
# Usage: tools/video_frames.sh <video> <count> <out_dir> [--cutout]
# Then:  mapanything-rs <out_dir>/*.png --focal-35mm <mm>
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 || ($# -eq 4 && $4 != --cutout) ]]; then
    echo "usage: $0 <video> <count> <out_dir> [--cutout]" >&2
    exit 2
fi
video=$1
count=$2
out_dir=$3
cutout=${4:-}
if ! [[ $count =~ ^[0-9]+$ ]] || (( count < 2 )); then
    echo "count must be an integer >= 2, got '$count'" >&2
    exit 2
fi

duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$video")
mkdir -p "$out_dir"
# ffmpeg applies the video's rotation metadata, so frames come out upright.
ffmpeg -v error -i "$video" -vf "fps=$count/$duration" -frames:v "$count" \
    "$out_dir/frame_%04d.png"
echo "extracted $(ls "$out_dir"/frame_*.png | wc -l | tr -d ' ') frames to $out_dir"

if [[ $cutout == --cutout ]]; then
    tools_dir=$(cd "$(dirname "$0")" && pwd)
    cutout_bin=${TMPDIR:-/tmp}/mapanything-cutout
    if [[ ! -x $cutout_bin || $tools_dir/cutout.swift -nt $cutout_bin ]]; then
        swiftc -O "$tools_dir/cutout.swift" -o "$cutout_bin"
    fi
    for frame in "$out_dir"/frame_*.png; do
        status=0
        "$cutout_bin" "$frame" "$frame" >/dev/null || status=$?
        if (( status == 3 )); then
            rm "$frame"
            echo "dropped $frame: no foreground found"
        elif (( status != 0 )); then
            exit "$status"
        fi
    done
    echo "cut the background out of $(ls "$out_dir"/frame_*.png | wc -l | tr -d ' ') frames"
fi
