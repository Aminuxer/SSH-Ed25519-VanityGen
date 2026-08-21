#!/usr/bin/env bash

# ==============================================================================
# Detect container runtime, build and run CPU vanity key image
# ==============================================================================
# Usage: ./run-podman-cpu.sh [pattern] [--patterns-file <file>] [-o <path>] [args...]
#   - No GPU required
#   - Build context: top-level Ed-25519-SSH-Vanity/ (parent of OpenCL-GPU/)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Resolve a path to absolute form
# ---------------------------------------------------------------------------
resolve_path() {
    local p="$1"
    if [[ "$p" == ~* ]]; then
        echo "$HOME/${p:2}"
    elif [[ ! "$p" == /* ]]; then
        echo "$(pwd)/$p"
    else
        echo "$p"
    fi
}

# ---------------------------------------------------------------------------
# Detect container runtime (podman preferred, docker as fallback)
# ---------------------------------------------------------------------------
detect_container_tool() {
    if command -v podman &>/dev/null; then
        echo "podman"
    elif command -v docker &>/dev/null; then
        echo "docker"
    else
        echo "none"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    if [ "$EUID" = "0" ]; then
        echo "ERROR: must NOT run as root."
        exit 1
    fi

    if [[ $# -eq 0 ]]; then
        echo "Usage: $0 [pattern] [--patterns-file <file>] [-o <path>] [args...]"
        echo "  CPU-only vanity key generator. No GPU required."
        echo "  Pattern is optional when --patterns-file is provided."
        exit 1
    fi

    local PATTERN=""
    if [[ "${1:0:1}" != "-" ]]; then
        PATTERN="$1"
        shift
    fi

    echo "=== Detecting environment ==="

    local container_tool
    container_tool="$(detect_container_tool)"
    echo "Container : ${container_tool}"

    if [[ "$container_tool" == "none" ]]; then
        echo "ERROR: No container runtime found (tried podman, docker)" >&2
        exit 1
    fi

    local CONTAINER_TOOL="$container_tool"

    local dockerfile="${SCRIPT_DIR}/Dockerfile.cpu"
    local image_tag="aminuxer-ssh-ed25519-cpu-vanity:latest"
    local BUILD_CONTEXT="$(cd "${SCRIPT_DIR}/.." && pwd)"

    echo "Dockerfile: ${dockerfile}"
    echo "Image tag : ${image_tag}"
    echo ""

    # -----------------------------------------------------------------------
    # Build image — skip if already cached
    # -----------------------------------------------------------------------
    echo "=== Checking image ==="
    if $CONTAINER_TOOL image inspect "${image_tag}" &>/dev/null; then
        echo "Image '${image_tag}' already exists, skipping build."
    else
        echo "=== Building image ==="
        if [[ "$CONTAINER_TOOL" == "podman" ]]; then
            if $CONTAINER_TOOL build --squash --rm -f "${dockerfile}" -t "${image_tag}" "${BUILD_CONTEXT}" 2>&1; then
                : # success
            else
                echo "Build failed (podkit auth), retrying with --isolation chroot..."
                $CONTAINER_TOOL build --squash --rm --isolation chroot -f "${dockerfile}" -t "${image_tag}" "${BUILD_CONTEXT}"
            fi
        else
            $CONTAINER_TOOL build --rm -f "${dockerfile}" -t "${image_tag}" "${BUILD_CONTEXT}" 2>&1
        fi
    fi

    echo ""
    echo "=== Running (CPU) ==="

    # -----------------------------------------------------------------------
    # Parse --patterns-file and -o from remaining args
    # -----------------------------------------------------------------------
    local pattern_file_mount=""
    local output_dir_mount=""
    local -a rebuilt=()
    local -a ARGS=("$@")
    local argc=${#ARGS[@]}

    local pfile_remote=""
    local outdir_real=""
    local outpath_real=""

    for ((i=0; i<argc; i++)); do
        case "${ARGS[$i]}" in
            --patterns-file)
                if ((i + 1 >= argc)); then
                    echo "ERROR: --patterns-file requires an argument" >&2
                    exit 1
                fi
                local pfile
                pfile="$(resolve_path "${ARGS[$((i+1))]}")"
                if [[ -f "$pfile" ]]; then
                    local base
                    base="$(basename "$pfile")"
                    pfile_remote="/input_patterns/${base}"
                    pattern_file_mount="-v ${pfile}:${pfile_remote}:ro"
                    echo "Patterns file: ${pfile} -> ${pfile_remote}"
                else
                    echo "ERROR: Patterns file not found: ${pfile}" >&2
                    exit 1
                fi
                ;;
            -o)
                if ((i + 1 >= argc)); then
                    echo "ERROR: -o requires an argument" >&2
                    exit 1
                fi
                local outpath
                outpath="$(resolve_path "${ARGS[$((i+1))]}")"
                local outdir
                outdir="$(dirname "$outpath")"
                outdir="$(cd "$outdir" 2>/dev/null && pwd)" || outdir="$(resolve_path "$outdir")"
                if [[ -d "$outdir" ]]; then
                    outdir_real="$outdir"
                    outpath_real="$outpath"
                else
                    echo "ERROR: Output directory does not exist: ${outdir}" >&2
                    exit 1
                fi
                ;;
        esac
    done

    if [[ -n "$outdir_real" ]]; then
        local mount_point="/output"
        output_dir_mount="-v ${outdir_real}:${mount_point}:rw"
        echo "Output dir: ${outdir_real} -> ${mount_point}"
    fi

    # --- Pass 2: rebuild args, replacing host paths with container paths ---
    for ((i=0; i<argc; i++)); do
        case "${ARGS[$i]}" in
            --patterns-file)
                rebuilt+=("--patterns-file" "$pfile_remote")
                ((i++)) || true
                ;;
            -o)
                rebuilt+=("-o" "/output/$(basename "$outpath_real")")
                ((i++)) || true
                ;;
            *)
                rebuilt+=("${ARGS[$i]}")
                ;;
        esac
    done

    set -- "${rebuilt[@]}"

    # Clean up stale container from previous runs
    if $CONTAINER_TOOL container inspect ssh-25519-vanity-cpu &>/dev/null; then
        echo "Removing stale container ssh-25519-vanity-cpu ..."
        $CONTAINER_TOOL rm -f ssh-25519-vanity-cpu 2>/dev/null || true
    fi

    # Build final argument list
    local -a final_args=()
    if [[ -n "$PATTERN" ]]; then
        final_args+=("$PATTERN")
    fi
    final_args+=("$@")

    exec $CONTAINER_TOOL run \
        --name ssh-25519-vanity-cpu \
        --read-only \
        --tmpfs /tmp \
        --network none \
        ${pattern_file_mount} \
        ${output_dir_mount} \
        "${image_tag}" "${final_args[@]}"
}

main "$@"
