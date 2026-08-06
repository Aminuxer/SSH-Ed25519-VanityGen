#!/usr/bin/env python3

"""
Ed-25519 SSH Vanity Key Generator [OpenCL GPU]
Port of ssh_ed25519_vanity_multicpu.py to GPU via OpenCL.
** Inspired by Aminuxer
** Version: 2026-08-07--N-GPU

Usage:
    python3 ssh_ed25519_vanity_gpu_opencl.py <pattern> [-i] [-w <workers>] [-o output] [--debug]
    python3 ssh_ed25519_vanity_gpu_opencl.py --patterns-file <file> [-i] [-w <workers>] [-o output] [--debug]

GPU-specific options:
    --opencl-devices a,b,c     Use specific device IDs (ignores -w)
    --load-percent 1-100       % of GPU cores to use (default: 100)
"""

import os
import sys
import time
import re
import pyopencl as cl
import numpy as np
from multiprocessing import Process, Queue, Event, Array
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


# --- OpenCL kernel inliner -----------------------------------------------

"""Recursively inline #include directives and apply NVIDIA OpenCL 1.2 compatibility fixes.

NVIDIA OpenCL 1.2 does NOT support:
  - __generic   (OpenCL 2.0 address-space qualifier)  → removed
  - __inline                           → replaced with "inline"
  - unsigned long long (ULL suffix)    → stripped
"""
def inline_cl(filepath, basedir, visited=None):
    """Recursively inline #include directives from the given .cl file.

    Returns the fully inlined and NVIDIA-compatible source as a string.
    """
    if visited is None:
        visited = set()

    bname = os.path.basename(filepath)
    if bname in visited:
        return ""
    visited.add(bname)

    # Resolve relative path
    if not os.path.isabs(filepath):
        filepath = os.path.join(basedir, filepath)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Include file not found: {filepath}")

    lines = []
    file_basedir = os.path.dirname(filepath)
    with open(filepath, 'r') as f:
        for line in f:
            m = re.match(r'\s*#include\s+[\"\'](.*)[\"\']', line)
            if m:
                inc = m.group(1)
                if '/' in inc:
                    inc_basedir = os.path.join(file_basedir, os.path.dirname(inc))
                else:
                    inc_basedir = file_basedir
                lines.append(inline_cl(inc, inc_basedir, visited))
            else:
                # NVIDIA OpenCL 1.2 compatibility
                line = line.replace('__generic', '')
                line = line.replace('__inline', 'inline')
                line = line.replace('ULL', '')
                lines.append(line)

    return "".join(lines)

# Valid Base64 characters for OpenSSH public key
B64_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")


# --- Helpers ------------------------------------------------------------

"""Format seconds into human-readable duration string (dd:hh:mm:ss or shorter)."""
def format_duration(seconds):
    """Format duration as days, hours, minutes, seconds."""
    if seconds < 0:
        return "0s"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


"""Validate vanity pattern: length 2-30, base64 chars only."""
def validate_pattern(pattern):
    """Check if pattern contains only valid Base64 characters."""
    if len(pattern) > 44:
        return False
    return all(c in B64_CHARS for c in pattern)


"""Sanitize pattern string for use in filename."""
def sanitize_filename(name):
    """Replace invalid filename characters with underscores."""
    return re.sub(r'[^a-zA-Z0-9\-_.]', '_', name)


"""Discover all available GPU OpenCL devices across platforms."""
def get_all_gpu_devices():
    """Return list of (platform, device) tuples for all GPU devices."""
    devices = []
    for platform in cl.get_platforms():
        for dev in platform.get_devices(device_type=cl.device_type.GPU):
            devices.append((platform, dev))
    return devices


# --- GPU Worker ---------------------------------------------------------

"""Worker function for one GPU: compile kernel, run vanity search loop, report progress."""
def worker_gpu(device_idx, patterns, case_insensitive,
                result_queue, stop_event,
                found_flags, kernel_path, load_percent):
    """GPU worker process -- runs OpenCL kernel in a loop.

    SEED LIFECYCLE:
      - Random seeds generated ONCE at startup via os.urandom()
      - Uploaded to GPU READ_WRITE buffer ONCE
      - Kernel increments each seed by 1 (256-bit LE) after each launch
      - NO H2D seed transfers in the main loop
      - On match: kernel writes the matching seed to foundSeeds buffer
    """
    import traceback

    pat_count = len(patterns)
    # Prepare pattern data for GPU (32-byte padded slots)
    pat_bytes = bytearray()
    pat_lens = bytearray()
    pat_ci = bytearray()
    for p in patterns:
        pbytes = p.lower().encode() if case_insensitive else p.encode()
        pat_bytes.extend(pbytes)
        pat_bytes.extend(b'\x00' * (32 - len(pbytes)))
        pat_lens.append(len(pbytes))
        pat_ci.append(1 if case_insensitive else 0)

    pat_bytes_np = np.frombuffer(bytes(pat_bytes), dtype=np.uint8)
    pat_lens_np  = np.frombuffer(bytes(pat_lens),  dtype=np.uint8)
    pat_ci_np    = np.frombuffer(bytes(pat_ci),    dtype=np.uint8)

    # Init OpenCL
    try:
        platforms = cl.get_platforms()
        all_devices = get_all_gpu_devices()
        if device_idx >= len(all_devices):
            result_queue.put(('error', f"Device index {device_idx} out of range"))
            return
        platform, device = all_devices[device_idx]
        dev_name = device.name

        mf = cl.mem_flags
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)

        kernel_dir = os.path.dirname(kernel_path)
        kernel_src = inline_cl(kernel_path, kernel_dir)
        program = cl.Program(ctx, kernel_src).build()
        kernel = cl.Kernel(program, 'vanity_search')

        # Determine work size
        max_wg = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        max_cu = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
        local_size = 32
        desired_global = int(max_wg * max_cu * load_percent / 100)
        batch_size = (desired_global // local_size) * local_size
        if batch_size == 0:
            batch_size = local_size

        print(f"[*] GPU [{device_idx}] {dev_name}: global={batch_size}, "
              f"local={local_size}, load={load_percent}%, patterns={pat_count}",
              flush=True)
    except Exception as e:
        result_queue.put(('error', f"GPU init failed: {e}\n{traceback.format_exc()}"))
        return

    # -- Allocate persistent GPU buffers --------------------------------
    mf = cl.mem_flags
    seeds_buf     = cl.Buffer(ctx, mf.READ_WRITE, batch_size * 32)
    results_buf   = cl.Buffer(ctx, mf.READ_WRITE, batch_size * 4)
    pubkey_buf    = cl.Buffer(ctx, mf.READ_WRITE, batch_size * 32)
    found_seeds_buf = cl.Buffer(ctx, mf.READ_WRITE, batch_size * 32)
    pat_bytes_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pat_bytes_np)
    pat_lens_buf  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pat_lens_np)
    pat_ci_buf    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pat_ci_np)

    # -- Generate random seeds ONCE and upload ONCE ---------------------
    seeds_np = np.frombuffer(os.urandom(batch_size * 32), dtype=np.uint8)
    cl.enqueue_copy(queue, seeds_buf, seeds_np, is_blocking=True)

    # -- P0: Double-buffered pipeline (single queue, ping-pong) ----------
    # Pipeline: [Kernel N] -> [D2H N] queued, then wait D2H N-1, process N-1.
    # GPU computes batch N while CPU processes results of batch N-1.
    results_A = np.empty(batch_size, dtype=np.int32)
    results_B = np.empty(batch_size, dtype=np.int32)
    found_seeds_np = np.empty(batch_size * 32, dtype=np.uint8)

    # Ping-pong pointers: curr = buffer for next D2H, prev = buffer to process
    curr_results = results_A
    prev_results = results_B
    # pending_evt: event for D2H of "curr" that will become "prev" next iter
    pending_evt = None

    iterations = 0
    last_prog_time = time.monotonic()
    matched_count = 0
    last_prog_iter = 0

    try:
        while not stop_event.is_set():
            # -- 1. Launch kernel (GPU starts computing batch N) ---------
            kernel.set_args(
                seeds_buf,
                np.int32(batch_size),
                pat_bytes_buf, pat_lens_buf, pat_ci_buf,
                np.int32(pat_count),
                results_buf, pubkey_buf,
                found_seeds_buf,
            )
            cl.enqueue_nd_range_kernel(queue, kernel,
                                       (batch_size,),
                                       (local_size,))

            # -- 2. Queue D2H for current batch (async, non-blocking) ----
            copy_evt = cl.enqueue_copy(queue, curr_results, results_buf,
                                        is_blocking=False)

            # -- 3. Wait for previous batch's D2H to complete -----------
            if pending_evt is not None:
                pending_evt.wait()

            # -- 4. Process PREVIOUS batch results on CPU ---------------
            iterations += batch_size

            match_indices = np.where(prev_results >= 0)[0]
            if len(match_indices) > 0:
                # P3: download found seeds async + explicit finish
                cl.enqueue_copy(queue, found_seeds_np, found_seeds_buf,
                                is_blocking=False)
                queue.finish()

                for i in match_indices:
                    pat_idx = int(prev_results[i])
                    try:
                        with found_flags.get_lock():
                            if found_flags[pat_idx] == 0:
                                found_flags[pat_idx] = 1
                            else:
                                continue
                    except Exception:
                        continue

                    seed_hex = found_seeds_np[i*32:(i+1)*32].tobytes().hex()
                    result_queue.put(('found', {
                        'pattern_idx': pat_idx,
                        'seed': seed_hex,
                        'iterations': iterations,
                    }))
                    matched_count += 1

            # -- 5. Swap ping-pong buffers for next iteration -----------
            curr_results, prev_results = prev_results, curr_results
            pending_evt = copy_evt

            # Progress reporting (every ~5 seconds)
            now = time.monotonic()
            if now - last_prog_time >= 5.0:
                try:
                    result_queue.put(('progress', iterations - last_prog_iter), block=False)
                except Exception:
                    pass
                last_prog_time = now
                last_prog_iter = iterations

    except Exception as e:
        result_queue.put(('error', f"Worker loop error: {e}\n{traceback.format_exc()}"))
    finally:
        for buf in [seeds_buf, results_buf, pubkey_buf, found_seeds_buf,
                    pat_bytes_buf, pat_lens_buf, pat_ci_buf]:
            try:
                buf.release()
            except Exception:
                pass
        try:
            ctx.release()
        except Exception:
            pass

    result_queue.put(('done', iterations))


# --- Main logic ---------------------------------------------------------

"""Main entry: launch GPU workers, collect matches, write SSH keys to disk."""
def generate_vanity_key(patterns, case_insensitive=False,
                        num_workers=None, output_file=None,
                        debug_mode=False, opencl_devices=None,
                        load_percent=100):
    """Generate vanity SSH keys using GPU workers."""
    if not patterns:
        print("[-] No valid patterns to search for")
        return None

    print(f"[*] Accepted patterns: {', '.join(patterns)}")
    print(f"[*] Case insensitive: {case_insensitive}")
    print(f"[*] Debug mode: {debug_mode}")
    print(f"[*] Load percent: {load_percent}%")

    # Enumerate devices
    devices = get_all_gpu_devices()
    if not devices:
        print("[-] No GPU devices found")
        return None

    print(f"[*] Available OpenCL devices:")
    for i, (plat, dev) in enumerate(devices):
        cu = dev.get_info(cl.device_info.MAX_COMPUTE_UNITS)
        wg = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        print(f"  [{i}] {plat.name}: {dev.name} (CU={cu}, WG={wg})")

    # Select devices
    if opencl_devices is not None:
        selected = []
        for idx in opencl_devices:
            if 0 <= idx < len(devices):
                selected.append(idx)
            else:
                print(f"[-] Invalid device index: {idx}")
                return None
        num_workers = len(selected)
        print(f"[*] Using {num_workers} GPU(s) (selected: {opencl_devices})")
    else:
        # If num_workers not specified, use ALL available GPUs by default
        if num_workers is None:
            num_workers = len(devices)
        num_workers = min(num_workers, len(devices))
        selected = list(range(num_workers))
        print(f"[*] Using {num_workers} GPU(s)")

    kernel_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'vanity_sshgen.cl')
    if not os.path.exists(kernel_path):
        print(f"[-] Kernel not found: {kernel_path}")
        return None

    # Set up multiprocessing
    try:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    result_queue  = Queue()
    stop_event    = Event()
    found_flags   = Array('i', [0] * len(patterns))  # shared flag per pattern

    # Start workers
    processes = []
    for dev_idx in selected:
        p = Process(
            target=worker_gpu,
            args=(dev_idx, patterns, case_insensitive,
                  result_queue, stop_event,
                  found_flags, kernel_path, load_percent),
        )
        p.start()
        processes.append(p)

    # Monitor loop
    total_iterations = 0
    start_time = time.monotonic()
    last_prog_time = start_time
    last_prog_iter = 0
    remaining = set(range(len(patterns)))
    first_printed = False
    progress_line = ""
    last_pub = None
    last_pem = None

    try:
        while len(remaining) > 0:
            # Check results
            try:
                msg_type, data = result_queue.get(timeout=3)
            except Exception:
                # Periodic progress display on timeout
                now = time.monotonic()
                if now - last_prog_time >= 5.0:
                    elapsed = now - start_time
                    rate = total_iterations / elapsed if elapsed > 0 else 0
                    rem = f" ({len(remaining)} left)" if remaining else ""
                    progress_line = (
                        f"\r[+] Progress: {total_iterations:,} keys "
                        f"({format_duration(elapsed)}) "
                        f"(~{rate:,.0f} keys/sec){rem}")
                    print(progress_line, end="", flush=True)
                    first_printed = True
                    last_prog_time = now
                    last_prog_iter = total_iterations
                continue

            if msg_type == 'progress':
                total_iterations += data
                continue

            if msg_type == 'done':
                total_iterations += data
                continue

            if msg_type == 'found':
                pat_idx = data['pattern_idx']
                seed_hex = data['seed']
                total_iterations = data.get('iterations', total_iterations)

                if pat_idx not in remaining:
                    continue  # already handled

                remaining.discard(pat_idx)
                matched_pat = patterns[pat_idx]
                elapsed = time.monotonic() - start_time

                if first_printed and progress_line:
                    print(f"\r{' ' * len(progress_line)}\r", end="", flush=True)

                # Generate key on CPU from seed
                seed_bytes = bytes.fromhex(seed_hex)
                priv_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed_bytes)
                pub_key  = priv_key.public_key()

                pub_bytes = pub_key.public_bytes(
                    encoding=serialization.Encoding.OpenSSH,
                    format=serialization.PublicFormat.OpenSSH,
                )
                pub_str = pub_bytes.decode()

                priv_pem_bytes = priv_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.OpenSSH,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                priv_pem_str = priv_pem_bytes.decode()

                print(f"\n[+] Found match for '{matched_pat}'!")
                print(f"[+] Public key: {pub_str} {matched_pat}")
                if debug_mode:
                    print(f"[+] Seed (hex): {seed_hex}")

                # Save to file or console
                saved = False
                if output_file:
                    try:
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        safe = sanitize_filename(matched_pat)
                        base = f"{output_file}-{safe}-{ts}"
                        out_dir = os.path.dirname(output_file) or '.'
                        with open(base + '.pub', 'w') as f:
                            f.write(pub_str + ' ' + matched_pat + '\n')
                        with open(base, 'w') as f:
                            f.write(priv_pem_str)
                        os.chmod(base, 0o600)
                        print(f"[+] Written: {base}.pub and {base} (mode 600)")
                        saved = True
                    except Exception as e:
                        print(f"[-] Save failed: {e}")

                if not saved:
                    print("[!] Output to console:")
                    print(priv_pem_str)

                last_pub = pub_str
                last_pem = priv_pem_str

                if remaining:
                    print("[*] Continuing search for remaining patterns...")
                else:
                    print("[*] All patterns found!")
                    stop_event.set()

            elif msg_type == 'error':
                print(f"[-] Worker error: {data}")
                stop_event.set()

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        stop_event.set()

    # Shutdown
    stop_event.set()
    for p in processes:
        try:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
        except Exception:
            pass

    if first_printed and progress_line:
        print()

    elapsed = time.monotonic() - start_time
    if last_pub:
        rate = total_iterations / elapsed if elapsed > 0 else 0
        print(f"\n[+] Checked keys: {total_iterations:,} "
              f"({format_duration(elapsed)}) (~{rate:,.0f} keys/sec)")
        return last_pub, last_pem, total_iterations, elapsed
    else:
        print(f"[+] Search completed. Iterations: {total_iterations:,}")
        return None


# --- CLI ----------------------------------------------------------------

"""CLI entry point: parse args, validate patterns, call generate_vanity_key."""
def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__.strip())
        sys.exit(0)

    pattern = None
    patterns_file = None
    case_insensitive = '-i' in sys.argv or '--ignore-case' in sys.argv
    debug_mode = '--debug' in sys.argv
    num_workers = None
    output_file = None
    opencl_devices = None
    load_percent = 100

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--patterns-file' and i + 1 < len(sys.argv):
            patterns_file = sys.argv[i + 1]
            i += 2
        elif arg in ('-w', '--workers') and i + 1 < len(sys.argv):
            try:
                num_workers = int(sys.argv[i + 1])
            except ValueError:
                pass
            i += 2
        elif arg in ('-o', '--output') and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif arg == '--opencl-devices' and i + 1 < len(sys.argv):
            try:
                opencl_devices = [int(x.strip())
                                  for x in sys.argv[i + 1].split(',')]
            except ValueError:
                print(f"[-] Invalid device list: {sys.argv[i+1]}")
                sys.exit(1)
            i += 2
        elif arg == '--load-percent' and i + 1 < len(sys.argv):
            try:
                load_percent = int(sys.argv[i + 1])
                if load_percent < 1 or load_percent > 100:
                    print("[-] --load-percent must be 1..100")
                    sys.exit(1)
            except ValueError:
                print("[-] --load-percent must be an integer")
                sys.exit(1)
            i += 2
        elif arg in ('-i', '--ignore-case', '--debug'):
            i += 1
        elif arg.startswith('-'):
            i += 1
        else:
            if pattern is None:
                pattern = sys.argv[i]
            i += 1

    # Build pattern list
    valid_patterns = []
    if patterns_file:
        if not os.path.isfile(patterns_file):
            print(f"[-] Patterns file not found: {patterns_file}")
            sys.exit(1)
        with open(patterns_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and validate_pattern(line):
                    valid_patterns.append(line)
                elif line:
                    print(f"[-] Warning: Skipping invalid pattern: '{line}'")
    elif pattern:
        if validate_pattern(pattern):
            valid_patterns.append(pattern)
        else:
            print(f"[-] Invalid pattern: '{pattern}'")
            sys.exit(1)
    else:
        print("[-] No pattern or patterns-file provided")
        sys.exit(1)

    # Output dir check
    if output_file:
        out_dir = os.path.dirname(output_file) or '.'
        if not os.path.isdir(out_dir):
            print(f"[-] Warning: Output directory does not exist: {out_dir}")
        elif not os.access(out_dir, os.W_OK):
            print(f"[-] Warning: No write permission for: {out_dir}")

    result = generate_vanity_key(
        valid_patterns,
        case_insensitive=case_insensitive,
        num_workers=num_workers,
        output_file=output_file,
        debug_mode=debug_mode,
        opencl_devices=opencl_devices,
        load_percent=load_percent,
    )

    if result:
        pub, pem, total_iter, duration = result
        print(f"[+] Total time: {format_duration(duration)}")
        print(f"[+] Checked keys: {total_iter:,}")


if __name__ == "__main__":
    main()
