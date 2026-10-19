#!/usr/bin/env bash
# Installs the desktop app (instantsplat-rs/app) as `splatting-rs`: the binary with
# `cargo install`, then a `splatting-rs.app` launcher for Spotlight, Launchpad and the Dock.
#
# The bundle's executable is a link to the installed binary, not a copy, so rerunning this
# script (or `cargo install` alone) upgrades what the launcher opens. macOS only.
set -euo pipefail
cd "$(dirname "$0")"

NAME=splatting-rs
IDENTIFIER=com.antoinemopa.splatting-rs
BINARY=instantsplat-app
ICON=docs/app_icon.png
# The shared folder first; the per-user one for an account that cannot write it.
APPLICATIONS_DIRS=(/Applications "$HOME/Applications")

[[ $(uname) == Darwin ]] || { echo "install-app.sh: macOS only" >&2; exit 1; }

cargo install --locked --path instantsplat-rs/app
executable="$(command -v "$BINARY" || true)"
[[ -n $executable ]] || executable="${CARGO_HOME:-$HOME/.cargo}/bin/$BINARY"
[[ -x $executable ]] || { echo "install-app.sh: $executable was not installed" >&2; exit 1; }

applications=""
for dir in "${APPLICATIONS_DIRS[@]}"; do
    mkdir -p "$dir" 2>/dev/null || continue
    if [[ -w $dir ]]; then applications="$dir"; break; fi
done
[[ -n $applications ]] || { echo "install-app.sh: none of ${APPLICATIONS_DIRS[*]} is writable" >&2; exit 1; }

bundle="$applications/$NAME.app"
if [[ -e $bundle ]] && ! grep -q "<string>$IDENTIFIER</string>" "$bundle/Contents/Info.plist" 2>/dev/null; then
    echo "install-app.sh: $bundle exists and is not ours; not touching it" >&2
    exit 1
fi
rm -rf "$bundle"
mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources"

cat > "$bundle/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$NAME</string>
    <key>CFBundleDisplayName</key><string>$NAME</string>
    <key>CFBundleIdentifier</key><string>$IDENTIFIER</string>
    <key>CFBundleExecutable</key><string>$BINARY</string>
    <key>CFBundleIconFile</key><string>$NAME</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

# The icon: every size an .icns carries, from the one PNG, with the tools macOS ships.
iconset="$(mktemp -d)/$NAME.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256 512; do
    sips -z $size $size "$ICON" --out "$iconset/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) "$ICON" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$bundle/Contents/Resources/$NAME.icns"
rm -rf "$(dirname "$iconset")"

ln -s "$executable" "$bundle/Contents/MacOS/$BINARY"
# Finder and the Dock cache the icon against the bundle's mtime.
touch "$bundle"
echo "installed $bundle -> $executable"
