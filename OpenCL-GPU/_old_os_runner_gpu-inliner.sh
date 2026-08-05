#!/bin/bash

###############################################################################
#  run_vanity_gpu.sh
#  Wrapper for ssh_ed25519_vanity_gpu_opencl.py - For Some OLD Systems (Fedora 36 etc)
#
#  Problem:  vanity_sshgen.cl uses #include "./sha512.cl" etc.
#            NVIDIA OpenCL compiler cannot resolve these because pyopencl
#            passes the source as a string without CWD context.
#  Solution: Recursively inline all #include directives into a single
#            self-contained .cl file, copy the Python script alongside,
#            and run the copy (so __file__ resolves to the working dir).
#
#  Original Python script is NEVER modified — only copied to a temp dir.
#
#  Usage:
#    ./run_vanity_gpu.sh <pattern> [options...]
#    ./run_vanity_gpu.sh --patterns-file file.txt [options...]
###############################################################################

set -euo pipefail

# Directory containing the original Python script and .cl kernel files.
# Override with VANITY_CL_DIR env var if needed.
CL_DIR="${VANITY_CL_DIR:-/opt/OpenCL-GPU}"
PYTHON_SCRIPT="$CL_DIR/ssh_ed25519_vanity_gpu_opencl.py"
KERNEL="$CL_DIR/vanity_sshgen.cl"

# Working directory (user-writable, cleaned up on exit)
WORK_DIR=$(mktemp -d "$HOME/.vanity_gpu_XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT

# ---- inline_cl: recursively inline #include + fix NVIDIA compat ----
# - Resolves all #include directives
# - Replaces __generic (OpenCL 2.0) → __global (NVIDIA supports only 1.2)
# - Replaces __inline → inline (some NVIDIA drivers prefer plain inline)
inline_cl() {
    local file="$1"
    local basedir="$2"
    local visited="$3"

    local bname
    bname="$(basename "$file")"

    # Already included?
    if grep -qFx "$bname" "$visited" 2>/dev/null; then
        return 0
    fi
    echo "$bname" >> "$visited"

    local filepath
    if [[ "$file" == /* ]]; then
        filepath="$file"
    else
        filepath="$basedir/$file"
    fi

    if [[ ! -f "$filepath" ]]; then
        echo "[-] Include file not found: $filepath" >&2
        return 1
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ ^[[:space:]]*#include[[:space:]]+[\"\'](.*)[\"\'] ]]; then
            local inc="${BASH_REMATCH[1]}"
            local inc_dir
            if [[ "$inc" == */* ]]; then
                inc_dir="$basedir/$(dirname "$inc")"
            else
                inc_dir="$basedir"
            fi
            inline_cl "$inc" "$inc_dir" "$visited"
        else
            # NVIDIA OpenCL 1.2 compatibility fixes:
            # - __generic (OpenCL 2.0) → removed (unqualified pointer accepts any address space)
            # - __inline → inline (some NVIDIA drivers)
            # - ULL suffix (unsigned long long) → stripped (NVIDIA doesn't support long long)
            line="${line//__generic/}"
            line="${line//__inline/inline}"
            line="${line//ULL/}"
            printf '%s\n' "$line"
        fi
    done < "$filepath"
}

# ---- Usage ----

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <pattern> [options...]"
    echo "       $0 --patterns-file file.txt [options...]"
    echo ""
    echo "Options (passed to ssh_ed25519_vanity_gpu_opencl.py):"
    echo "  -i                  Case insensitive"
    echo "  -w N                Number of GPU workers (default: all GPUs)"
    echo "  -o PATH             Output file path"
    echo "  --debug             Debug mode"
    echo "  --opencl-devices a,b,c  Use specific device IDs"
    echo "  --load-percent 1-100    GPU load percentage (default: 100)"
    exit 1
fi

# ---- Pre-flight checks ----

if [[ ! -f "$KERNEL" ]]; then
    echo "[-] Kernel file not found: $KERNEL"
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "[-] Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

for PKG in pyopencl numpy cryptography; do
    if ! python3 -c "import ${PKG}" 2>/dev/null; then
        echo "[-] Python package '$PKG' not installed."
        echo "    Run: bash setup_vanity_gpu_env.sh"
        exit 1
    fi
done

# ---- Inline the kernel ----

VISITED=$(mktemp "$WORK_DIR/visited.XXXXXX")

echo "[*] Inlining OpenCL kernel includes..."

inline_cl "$KERNEL" "$CL_DIR" "$VISITED" > "$WORK_DIR/vanity_sshgen.cl"

if [[ ! -s "$WORK_DIR/vanity_sshgen.cl" ]]; then
    echo "[-] Failed to generate inlined kernel"
    exit 1
fi

# Report
INCLUDED=$(wc -l < "$VISITED")
ORIG_LINES=$(wc -l < "$KERNEL")
INLINE_LINES=$(wc -l < "$WORK_DIR/vanity_sshgen.cl")
echo "[*] Inlined $INCLUDED .cl file(s): $ORIG_LINES -> $INLINE_LINES lines"

# ---- Copy the Python script to working dir ----
# This is a copy, NOT a modification — original script stays untouched.
cp -f "$PYTHON_SCRIPT" "$WORK_DIR/"
chmod +x "$WORK_DIR/ssh_ed25519_vanity_gpu_opencl.py"

echo "[*] Working directory: $WORK_DIR"
echo "[*] Starting vanity key search..."
echo "----------------------------------------"

# ---- Run ----
python3 "$WORK_DIR/ssh_ed25519_vanity_gpu_opencl.py" "$@"
EXIT_CODE=$?

echo "----------------------------------------"
exit $EXIT_CODE
