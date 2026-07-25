#!/usr/bin/env bash
# Builds and packages a Dictata release for Linux x86_64.
#
# STATUS: UNVERIFIED. This script has never been executed — the project is
# developed on Windows and no Linux machine was available. It reflects what
# the dependency documentation requires, not a build that was observed to
# succeed. Run it on a real Linux box before trusting it, and read
# LINUX.md first: several features are known NOT to work on Linux.
#
# Produces dist/dictata-<version>-linux-x86_64[-cpu].tar.gz + .sha256.
# Nothing is committed, tagged or uploaded.
#
# Usage:
#   scripts/release-linux.sh            # CPU build (no Vulkan SDK needed)
#   scripts/release-linux.sh --gpu      # Vulkan build (needs the Vulkan SDK)
#   scripts/release-linux.sh --skip-tests

set -euo pipefail

cd "$(dirname "$0")/.."
root="$PWD"

variant="cpu"
skip_tests=0
for arg in "$@"; do
    case "$arg" in
        --gpu) variant="gpu" ;;
        --cpu) variant="cpu" ;;
        --skip-tests) skip_tests=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done

step() { printf '==> %s\n' "$1"; }
fail() { printf '!!! %s\n' "$1" >&2; exit 1; }

# Build-time system packages (Debian/Ubuntu names). Sources:
#   enigo      -> libxdo-dev, libX11-dev  (X11 input simulation)
#   tray-icon  -> libgtk-3-dev, libxdo-dev, libayatana-appindicator3-dev
#   cpal       -> libasound2-dev (ALSA)
#   eframe     -> libgtk-3-dev / X11 dev headers (file dialogs via rfd)
#   whisper-rs -> cmake, a C++ toolchain (+ the Vulkan SDK for --gpu)
step 'Required build packages (Debian/Ubuntu), install them first:'
cat <<'EOF'
    sudo apt install build-essential cmake pkg-config \
        libasound2-dev libgtk-3-dev libxdo-dev libx11-dev \
        libayatana-appindicator3-dev
    # --gpu additionally requires the Vulkan SDK (libvulkan-dev, glslc/shaderc)
EOF

# ------------------------------------------------------------------ metadata
step 'Reading cargo metadata'
command -v cargo >/dev/null || fail 'cargo not found in PATH'
meta=$(cargo metadata --format-version 1 --no-deps)
# Version straight from the manifest: deterministic, no JSON parsing needed.
version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
[ -n "$version" ] || fail 'could not read the version from Cargo.toml'
# target-dir may be redirected by .cargo/config.toml, so ask cargo for it.
target_dir=$(printf '%s' "$meta" | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
[ -n "$target_dir" ] || target_dir="$root/target"
echo "    dictata $version"
echo "    target-dir: $target_dir"

step 'Toolchain'
rustc --version
cargo --version

# --------------------------------------------------------------------- tests
if [ "$skip_tests" -eq 0 ]; then
    step 'cargo test'
    cargo test || fail 'tests failed - release aborted'
else
    echo '    (tests skipped)'
fi

# --------------------------------------------------------------------- build
step "Building release ($variant)"
if [ "$variant" = "cpu" ]; then
    cargo build --release --no-default-features
else
    cargo build --release
fi

bin="$target_dir/release/dictata"
[ -f "$bin" ] || fail "binary not found: $bin"

# ------------------------------------------------------------------- package
suffix=""
[ "$variant" = "cpu" ] && suffix="-cpu"
name="dictata-$version-linux-x86_64$suffix"
out="$root/dist"
stage="$out/$name"

mkdir -p "$out"
rm -rf "$stage"
mkdir -p "$stage"

cp "$bin" "$stage/dictata"
chmod +x "$stage/dictata"
cp "$root/README.md" "$stage/"
cp "$root/LICENSE" "$stage/"
[ -f "$root/CHANGELOG.md" ] && cp "$root/CHANGELOG.md" "$stage/"
[ -f "$root/LINUX.md" ] && cp "$root/LINUX.md" "$stage/"

# No config.json is shipped: the app writes its own defaults on first run, and
# a developer's config.json carries personal data (app rules, LLM endpoint,
# vocabulary).

tar -C "$out" -czf "$out/$name.tar.gz" "$name"
( cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256" )

step 'Done'
echo "    $out/$name.tar.gz"
cat "$out/$name.tar.gz.sha256"
echo 'Reminder: nothing was committed, tagged or uploaded.'
