#!/usr/bin/env bash

# ==============================================================================
# Detect host OS + GPU vendor + container runtime, build and run image
# ==============================================================================
# Usage: ./run-podman-gpu.sh <pattern> [args...]
#   - Automatically detects:
#     1. Container runtime (podman prefer, docker as fallback)
#     2. OS family (Fedora/RHEL-family vs Debian/Ubuntu)
#     3. GPU vendor (NVIDIA vs AMD)
#   - Builds the appropriate Dockerfile.*
#   - Runs with correct device mounts
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
# Detect OS family
# ---------------------------------------------------------------------------
detect_os_family() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        source /etc/os-release
        case "${ID,,}" in
            fedora|rhel|centos|almalinux|rocky|amzn|opencloudos|tencentos)
                echo "fedora"
                ;;
            debian|ubuntu)
                echo "debian"
                ;;
            *)
                echo "unknown"
                ;;
        esac
    elif command -v rpm &>/dev/null; then
        echo "fedora"
    elif command -v dpkg &>/dev/null; then
        echo "debian"
    else
        echo "unknown"
    fi
}

# ---------------------------------------------------------------------------
# Detect GPU vendor
# ---------------------------------------------------------------------------
detect_gpu_vendor() {
    # Check NVIDIA first
    if command -v nvidia-smi &>/dev/null; then
        echo "nvidia"
        return
    fi

    # Check lspci for NVIDIA or AMD
    if command -v lspci &>/dev/null; then
        if lspci 2>/dev/null | grep -iqi "nvidia"; then
            echo "nvidia"
            return
        fi
        if lspci 2>/dev/null | grep -iqi -e "amd.*vga" -e "advanced.*micro.*devices.*vga" -e "rdna" -e "radeon"; then
            echo "amd"
            return
        fi
    fi

    # Check /proc for NVIDIA
    if [[ -d /dev/nvidia* ]]; then
        echo "nvidia"
        return
    fi

    echo "none"
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
# Get Dockerfile and image tag based on OS family
# ---------------------------------------------------------------------------
get_dockerfile_info() {
    local os_family="$1"
    local gpu_vendor="$2"

    local dockerfile=""
    local image_tag=""

    case "${os_family}" in
        fedora)
            dockerfile="${SCRIPT_DIR}/Dockerfile.fedora"
            ;;
        debian)
            dockerfile="${SCRIPT_DIR}/Dockerfile.debian"
            ;;
        *)
            echo "ERROR: Unsupported OS family: ${os_family}" >&2
            echo "Supported: Fedora/RHEL-family, Debian/Ubuntu" >&2
            exit 1
            ;;
    esac

    case "${gpu_vendor}" in
        nvidia)
            image_tag="aminuxer-ssh-ed25519-gpu-vanity:latest"
            ;;
        amd)
            image_tag="aminuxer-ssh-ed25519-gpu-vanity-amd:latest"
            ;;
        *)
            echo "ERROR: No GPU detected (tried nvidia-smi, lspci, /dev/nvidia*)" >&2
            exit 1
            ;;
    esac

    echo "${dockerfile}|${image_tag}"
}

# ---------------------------------------------------------------------------
# Get podman run flags for GPU
# ---------------------------------------------------------------------------
get_run_flags() {
    local gpu_vendor="$1"

    case "${gpu_vendor}" in
        nvidia)
            echo "--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-caps -v /usr/lib64:/usr/lib64:ro -v /etc/OpenCL:/etc/OpenCL:ro"
            ;;
        amd)
            echo "--device /dev/dri:/dev/dri --device /dev/kfd:/dev/kfd -v /usr/lib64:/usr/lib64:ro -v /etc/OpenCL:/etc/OpenCL:ro"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    # --- Must not run as root ---
    if [ "$EUID" = "0" ]; then
       echo "ERROR: must NOT run as root."
       exit 1
    fi

    if [[ $# -eq 0 ]]; then
        echo "Usage: $0 [pattern] [--patterns-file <file>] [-o <path>] [args...]"
        echo "  Detects host OS family (Fedora/Debian), GPU vendor (NVIDIA/AMD),"
        echo "  and container runtime (podman/docker)."
        echo "  Pattern is optional when --patterns-file is provided."
        exit 1
    fi

    local PATTERN=""
    if [[ "${1:0:1}" != "-" ]]; then
        PATTERN="$1"
        shift
    fi

    echo "=== Detecting environment ==="

    local os_family
    os_family="$(detect_os_family)"
    echo "OS family : ${os_family}"

    local gpu_vendor
    gpu_vendor="$(detect_gpu_vendor)"
    echo "GPU vendor: ${gpu_vendor}"

    local container_tool
    container_tool="$(detect_container_tool)"
    echo "Container : ${container_tool}"

    if [[ "$container_tool" == "none" ]]; then
        echo "ERROR: No container runtime found (tried podman, docker)" >&2
        exit 1
    fi

    local CONTAINER_TOOL="$container_tool"


    # Resolve Dockerfile and image tag
    local info
    info="$(get_dockerfile_info "${os_family}" "${gpu_vendor}")"
    local dockerfile image_tag
    dockerfile="$(echo "${info}" | cut -d'|' -f1)"
    image_tag="$(echo "${info}" | cut -d'|' -f2)"

    echo "Dockerfile: ${dockerfile}"
    echo "Image tag : ${image_tag}"
    echo ""

    # Build context is the parent of SCRIPT_DIR (where OpenCL-GPU/ lives).
    local BUILD_CONTEXT="$(cd "${SCRIPT_DIR}/.." && pwd)"

    # -----------------------------------------------------------------------
    # Build image — skip if already cached; fallback isolation if needed
    # -----------------------------------------------------------------------
    echo "=== Checking image ==="
    if $CONTAINER_TOOL image inspect "${image_tag}" &>/dev/null; then
        echo "Image '${image_tag}' already exists, skipping build."
    else
        echo "=== Building image ==="
        if [[ "$CONTAINER_TOOL" == "podman" ]]; then
            # Podman supports --squash (merge all layers into one).
            if $CONTAINER_TOOL build --squash --rm -f "${dockerfile}" -t "${image_tag}" "${BUILD_CONTEXT}" 2>&1; then
                : # success
            else
                echo "Build failed (podkit auth), retrying with --isolation chroot..."
                $CONTAINER_TOOL build --squash --rm --isolation chroot -f "${dockerfile}" -t "${image_tag}" "${BUILD_CONTEXT}"
            fi
        else
            # Docker CE — no --squash, just build normally.
            $CONTAINER_TOOL build --rm -f "${dockerfile}" -t "${image_tag}" "${BUILD_CONTEXT}" 2>&1
        fi
    fi

    echo ""
    echo "=== Running (GPU: ${gpu_vendor}) ==="

    # Run container with appropriate flags
    local run_flags
    run_flags="$(get_run_flags "${gpu_vendor}")"

    # -----------------------------------------------------------------------
    # Parse --patterns-file and -o from remaining args.
    # Two-pass: first collect mounts, then rebuild args with container paths.
    # -----------------------------------------------------------------------
    local pattern_file_mount=""
    local output_dir_mount=""
    local -a rebuilt=()
    local -a ARGS=("$@")
    local argc=${#ARGS[@]}

    # --- Pass 1: detect --patterns-file and -o, prepare mounts ---
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
                fi
                ;;
        esac
    done

    # Build output_dir_mount after we know outdir_real
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

    # Restore positional parameters with rebuilt args
    set -- "${rebuilt[@]}"

    # Cache dir — must exist before container mount (host side).
    mkdir -p /tmp/aminuxer-gpu-vanity-cache
    chmod 1777 /tmp/aminuxer-gpu-vanity-cache

    # Clean up stale container from previous runs
    if $CONTAINER_TOOL container inspect ssh-25519-vanity-gpu &>/dev/null; then
        echo "Removing stale container ssh-25519-vanity-gpu ..."
        $CONTAINER_TOOL rm -f ssh-25519-vanity-gpu 2>/dev/null || true
    fi

    # Build final argument list: pattern (if any) + rebuilt args
    local -a final_args=()
    if [[ -n "$PATTERN" ]]; then
        final_args+=("$PATTERN")
    fi
    final_args+=("$@")

    exec $CONTAINER_TOOL run \
        --name ssh-25519-vanity-gpu \
        --read-only \
        --tmpfs /tmp \
        --network none \
        ${pattern_file_mount} \
        ${output_dir_mount} \
        -v /tmp/aminuxer-gpu-vanity-cache:/home/aminuxer/.cache:rw \
        -v /tmp/aminuxer-gpu-vanity-cache:/home/aminuxer/.nv/ComputeCache:rw \
        ${run_flags} \
        "${image_tag}" "${final_args[@]}"
}

main "$@"

