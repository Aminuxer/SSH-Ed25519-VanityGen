#!/usr/bin/env python3

"""
Ed-25519 SSH Vanity Key Generator [OpenCL GPU]
Port of ssh_ed25519_vanity_multicpu.py to GPU via OpenCL.
** Inspired by Aminuxer
** Version: 2026-09-11


Usage:
    python3 ssh_ed25519_vanity_gpu_opencl.py <pattern> [-i] [-w <workers>] [-o output] [--debug]
    python3 ssh_ed25519_vanity_gpu_opencl.py --patterns-file <file> [-i] [-w <workers>] [-o output] [--debug]

GPU-specific options:
    --opencl-devices a,b,c     Use specific device IDs (ignores -w)
    --load-percent 1-100       % of GPU cores to use (default: 100)
    --batch-mult 1-16          Multiply the per-launch batch (default: 1)
"""

# Fault-tolerance constants (seconds)
KERNEL_TIMEOUT   = 120  # one pipeline event wait beyond this => GPU dead
WATCHDOG_TIMEOUT = 60   # worker silent this long => main kills + retries
MAX_GPU_ATTEMPTS = 10   # respawn attempts per GPU, then excluded
RETRY_BACKOFF_SEC = (30, 60, 120, 300)  # capped at 300s
PROGRESS_FLUSH_SEC = 0.25

import os
import sys
import time
import re
import signal
import array
import base64
import struct
import ctypes
import pyopencl as cl
from multiprocessing import Process, Queue, Event, Array
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

"""Recursively inline #include directives and apply NVIDIA OpenCL 1.2 compatibility fixes.

NVIDIA OpenCL 1.2 does NOT support:
  - __generic   (OpenCL 2.0 address-space qualifier)  -> removed
  - __inline                           -> replaced with "inline"
  - unsigned long long (ULL suffix)    -> stripped
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
    """Check pattern contains only valid Base64 characters."""
    if len(pattern) > 44:
        return False
    return all(c in B64_CHARS for c in pattern)


"""Sanitize pattern string for use in filename."""
def sanitize_filename(name):
    """Replace invalid filename characters with underscores."""
    return re.sub(r'[^a-zA-Z0-9\-_.]', '_', name)


def _ssh_str(b):
    """OpenSSH wire format: uint32 length prefix + bytes."""
    return struct.pack(">I", len(b)) + b


"""Build an unencrypted openssh-key-v1 ed25519 private key from seed + public key.

The cryptography library cannot serialize a key from a raw (seed, pubkey)
pair, so the OpenSSH wire format is assembled directly. Layout follows
sshkey.c sshkey_private_to_blob2() byte-for-byte:

    magic "openssh-key-v1\0"
    string  ciphername "none"
    string  kdfname    "none"
    string  kdf        ""
    uint32  number of keys (1)          <- RAW u32, not a string
    string  public key blob
    uint32  private section length
    private section:
        uint32  check                   <- random, written TWICE
        uint32  check
        string  keytype "ssh-ed25519"
        string  pubkey  (32 bytes)
        string  privkey (64 bytes: seed32 || pubkey32)
        string  comment
        padding bytes 1,2,3,... so the section length % 8 == 0

OpenSSH stores the RFC 8032 seed and re-derives the clamped scalar from it
at signing time, so seed32 must be the actual seed that produced pubkey32.
"""
def build_openssh_private_key(pubkey32, seed32, comment=b""):
    keytype = b"ssh-ed25519"
    public_blob = _ssh_str(keytype) + _ssh_str(pubkey32)
    check = os.urandom(4)
    private = (check + check
               + _ssh_str(keytype)
               + _ssh_str(pubkey32)
               + _ssh_str(seed32 + pubkey32)
               + _ssh_str(comment))
    pad_len = 8 - (len(private) % 8)
    private += bytes(range(1, pad_len + 1))
    blob = (b"openssh-key-v1\x00"
            + _ssh_str(b"none") + _ssh_str(b"none") + _ssh_str(b"")
            + struct.pack(">I", 1)
            + _ssh_str(public_blob) + _ssh_str(private))
    b64 = base64.b64encode(blob).decode()
    # ssh-keygen wraps the base64 body at 70 columns
    b64 = "\n".join(b64[i:i + 70] for i in range(0, len(b64), 70))
    return ("-----BEGIN OPENSSH PRIVATE KEY-----\n" + b64
            + "\n-----END OPENSSH PRIVATE KEY-----\n")


"""Discover all available GPU OpenCL devices across platforms."""
def get_all_gpu_devices():
    """Return list of (platform, device) tuples for all GPU devices."""
    devices = []
    for platform in cl.get_platforms():
        for dev in platform.get_devices(device_type=cl.device_type.GPU):
            devices.append((platform, dev))
    return devices


# --- GPU Worker ---------------------------------------------------------

"""Worker function for one GPU: compile kernel, run vanity search loop, report.

Fault model (one worker = one GPU = one process):
  - init failure / device list changed  -> ('gpu-dead', reason), exit
  - pipeline event wait timeout         -> ('gpu-dead', reason), exit
  - parent process died                 -> silent exit (orphan guard)
  - stop_event set                      -> clean ('done', iterations)
The main process owns all retries; a worker never retries itself.
"""
def worker_gpu(device_idx, patterns, case_insensitive,
                result_queue, stop_event,
                found_flags, kernel_path, load_percent,
                n_devices_expected, batch_mult):
    import threading
    import traceback

    ppid0 = os.getppid()

    def parent_alive():
        return os.getppid() == ppid0

    def gpu_dead(reason):
        """Report death and exit. The process exit is the only reliable
        cleanup of a wedged OpenCL context, so this never returns."""
        try:
            result_queue.put(('gpu-dead', reason), timeout=5)
        except Exception:
            pass
        os._exit(1)

    def wait_event(evt, timeout=KERNEL_TIMEOUT):
        """evt.wait() with a wall-clock timeout (driver has none).

        Returns True if the event completed in time, False otherwise.
        """
        done = []

        def _w(e=evt):
            try:
                e.wait()
                done.append(True)
            except Exception:
                done.append(False)
        th = threading.Thread(target=_w, daemon=True)
        th.start()
        th.join(timeout=timeout)
        return bool(done) and done[0]

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

    # -- Init OpenCL ----------------------------------------------------
    try:
        devices = get_all_gpu_devices()
        if len(devices) != n_devices_expected:
            gpu_dead(f"device list changed: expected {n_devices_expected} GPUs, "
                     f"seen {len(devices)}")
        if device_idx >= len(devices):
            gpu_dead(f"device index {device_idx} out of range ({len(devices)})")
        platform, device = devices[device_idx]
        dev_name = device.name

        mf = cl.mem_flags
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)

        kernel_dir = os.path.dirname(kernel_path)
        print(f"[+] Compiling OpenCL kernel on GPU [{device_idx}] {dev_name} ...", flush=True)
        kernel_src = inline_cl(kernel_path, kernel_dir)
        program = cl.Program(ctx, kernel_src).build()
        kernel = cl.Kernel(program, 'vanity_search')

        # Determine work size
        max_wg = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        max_cu = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
        local_size = 32
        desired_global = int(max_wg * max_cu * load_percent / 100) * batch_mult
        batch_size = (desired_global // local_size) * local_size
        if batch_size == 0:
            batch_size = local_size

        print(f"[*] GPU [{device_idx}] {dev_name}: global={batch_size}, "
              f"local={local_size}, load={load_percent}%, batch_mult={batch_mult}, "
              f"patterns={pat_count}", flush=True)
        # Signal main process that this worker is ready
        result_queue.put(('ready', device_idx))
    except Exception as e:
        gpu_dead(f"GPU init failed: {e}\n{traceback.format_exc()}")

    # -- Allocate persistent GPU buffers --------------------------------
    mf = cl.mem_flags
    seeds_buf   = cl.Buffer(ctx, mf.READ_WRITE, batch_size * 32)
    # results/pubkey: written by the kernel for debugging/verification,
    # the host NEVER reads them in the search loop.
    results_buf = cl.Buffer(ctx, mf.READ_WRITE, batch_size * 4)
    pubkey_buf  = cl.Buffer(ctx, mf.READ_WRITE, batch_size * 32)
    # Ring of 3 (deeper pipeline than ping-pong): batch N uses slot N%3;
    # the host processes batch N-2's counter, which is guaranteed
    # complete by then. The GPU queue stays full -> no starvation.
    found_bufs  = [cl.Buffer(ctx, mf.READ_WRITE, batch_size * 68) for _ in range(3)]
    count_bufs  = [cl.Buffer(ctx, mf.READ_WRITE, 4) for _ in range(3)]
    pat_bytes_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bytes(pat_bytes))
    pat_lens_buf  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bytes(pat_lens))
    pat_ci_buf    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bytes(pat_ci))

    # -- Generate random seeds ONCE and upload ONCE ---------------------
    cl.enqueue_copy(queue, seeds_buf, os.urandom(batch_size * 32), is_blocking=True)
    zeros4 = array.array('i', [0])
    for b in count_bufs:
        cl.enqueue_copy(queue, b, zeros4, is_blocking=True)

    count_host = [array.array('i', [0]) for _ in range(3)]

    iterations = 0
    n_batches = 0
    last_prog_iter = 0
    last_prog_put = time.monotonic()
    evt_ring = [None, None, None]  # counter D2H event per ring slot

    # Optional per-phase timing (VANTITY_PROFILE=1): printed every
    # 500 batches as averages in microseconds.
    prof = os.environ.get('VANTITY_PROFILE') == '1'
    prof_acc = [0.0, 0.0, 0.0, 0.0]  # launch, wait, process, reset
    prof_n = 0

    try:
        while not stop_event.is_set():
            # Orphan guard: parent (the farm) died -> release the GPU.
            if not parent_alive():
                os._exit(0)

            idx = n_batches % 3
            if prof:
                _t = time.monotonic()
            # -- 1. Launch kernel (batch N) + 4-byte counter D2H --------
            kernel.set_args(
                seeds_buf,
                ctypes.c_int32(batch_size),
                pat_bytes_buf, pat_lens_buf, pat_ci_buf,
                ctypes.c_int32(pat_count),
                results_buf, pubkey_buf,
                found_bufs[idx], count_bufs[idx],
            )
            cl.enqueue_nd_range_kernel(queue, kernel,
                                       (batch_size,), (local_size,))
            evt = cl.enqueue_copy(queue, count_host[idx], count_bufs[idx],
                                  is_blocking=False)
            evt_ring[idx] = evt
            if prof:
                prof_acc[0] += time.monotonic() - _t

            # -- 2. Wait for batch N-2's counter + process it -----------
            #    Two batches behind its own kernel: guaranteed complete,
            #    and the GPU queue already holds newer kernels, so the
            #    CPU wait never starves the GPU.
            if n_batches >= 2:
                proc_idx = (n_batches - 2) % 3
                if prof:
                    _t = time.monotonic()
                if not wait_event(evt_ring[proc_idx]):
                    gpu_dead(f"pipeline event wait > {KERNEL_TIMEOUT}s "
                             f"(GPU or driver wedged)")
                if prof:
                    prof_acc[1] += time.monotonic() - _t
                n_matches = count_host[proc_idx][0]
                if prof:
                    _t = time.monotonic()
                if n_matches > 0:
                    # CPU work only here: final assembly data for found
                    # keys. The host buffer is exactly as big as needed.
                    found_host = bytearray(n_matches * 68)
                    f_evt = cl.enqueue_copy(queue, found_host,
                                            found_bufs[proc_idx],
                                            is_blocking=False)
                    if not wait_event(f_evt):
                        gpu_dead("found-entries D2H wait timeout")
                    for k in range(n_matches):
                        entry = found_host[k * 68:(k + 1) * 68]
                        pat_idx = int.from_bytes(entry[64:68], 'little')
                        if pat_idx >= pat_count:
                            continue  # defensive: never happens
                        with found_flags.get_lock():
                            if found_flags[pat_idx] != 0:
                                continue  # another worker got it first
                            found_flags[pat_idx] = 1
                        # Sync accumulated progress so the farm total is
                        # accurate by the time the main reports the key.
                        if iterations - last_prog_iter > 0:
                            try:
                                result_queue.put(
                                    ('progress', iterations - last_prog_iter),
                                    block=False)
                            except Exception:
                                pass
                            last_prog_iter = iterations
                        result_queue.put(('found', {
                            'pattern_idx': pat_idx,
                            'seed': entry[:32].hex(),
                            'pub': entry[32:64].hex(),
                            'iterations': iterations,
                        }))
                    # Reset the consumed counter. Rare (only on match):
                    # the 4-byte H2D enqueue stalls ~7ms on this driver
                    # while a kernel is in flight, so it must NOT run
                    # on the common no-match path (counter is already 0).
                    # Queue ordering guarantees it lands before the next
                    # kernel that reuses this counter (batch N+1).
                    if prof:
                        prof_acc[2] += time.monotonic() - _t
                        _t = time.monotonic()
                    cl.enqueue_copy(queue, count_bufs[proc_idx], zeros4,
                                    is_blocking=False)
                    if prof:
                        prof_acc[3] += time.monotonic() - _t
                elif prof:
                    prof_acc[2] += time.monotonic() - _t

            # -- 3. Bookkeeping ------------------------------------------
            n_batches += 1
            iterations += batch_size

            if prof:
                prof_n += 1
                if prof_n % 500 == 0:
                    avg = [v * 1e6 / prof_n for v in prof_acc]
                    sys.stderr.write(
                        f"[prof] GPU{device_idx} n={prof_n} "
                        f"launch={avg[0]:.0f}us wait={avg[1]:.0f}us "
                        f"process={avg[2]:.0f}us reset={avg[3]:.0f}us "
                        f"cpu_total={sum(avg):.0f}us\n")
                    sys.stderr.flush()
                    prof_acc = [0.0, 0.0, 0.0, 0.0]
                    prof_n = 0

            now = time.monotonic()
            if now - last_prog_put >= PROGRESS_FLUSH_SEC:
                try:
                    result_queue.put(('progress', iterations - last_prog_iter),
                                     block=False)
                except Exception:
                    pass
                last_prog_iter = iterations
                last_prog_put = now

    except Exception as e:
        gpu_dead(f"worker loop error: {e}\n{traceback.format_exc()}")
    finally:
        for buf in [seeds_buf, results_buf, pubkey_buf,
                    found_bufs[0], found_bufs[1], found_bufs[2],
                    count_bufs[0], count_bufs[1], count_bufs[2],
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

"""Per-GPU lifecycle state owned by the main process."""
class _GpuSlot:
    __slots__ = ('device_idx', 'status', 'proc', 'queue', 'attempt',
                 'next_retry', 'iterations', 'last_msg',
                 'rate_iter', 'rate_time')
    LAUNCHING = 'launching'
    RUNNING = 'running'
    RETRY = 'retry'
    DEAD = 'dead'

    def __init__(self, device_idx):
        self.device_idx = device_idx
        self.status = self.LAUNCHING
        self.proc = None
        self.queue = None
        self.attempt = 0
        self.next_retry = 0.0
        self.iterations = 0
        self.last_msg = time.monotonic()
        self.rate_iter = 0
        self.rate_time = time.monotonic()


"""Main entry: launch GPU workers, collect matches, write SSH keys to disk."""
def generate_vanity_key(patterns, case_insensitive=False,
                        num_workers=None, output_file=None,
                        debug_mode=False, opencl_devices=None,
                        load_percent=100, batch_mult=1):
    """Generate vanity SSH keys using GPU workers.

    Farm semantics: a GPU dying (init failure, hang, driver loss) only
    affects that GPU. It is respawned up to MAX_GPU_ATTEMPTS times with
    backoff, then excluded. The run ends when all patterns are found or
    no GPU is left.
    """
    if not patterns:
        print("[-] No valid patterns to search for")
        return None

    print(f"[*] Accepted patterns: {', '.join(patterns)}")
    print(f"[*] Case insensitive: {case_insensitive}")
    print(f"[*] Debug mode: {debug_mode}")
    print(f"[*] Load percent: {load_percent}%, batch mult: {batch_mult}")

    # Enumerate devices
    devices = get_all_gpu_devices()
    if not devices:
        print("[-] No GPU devices found")
        return None

    n_devices_expected = len(devices)
    print(f"[*] Available OpenCL devices ({n_devices_expected}):")
    for i, (plat, dev) in enumerate(devices):
        cu = dev.get_info(cl.device_info.MAX_COMPUTE_UNITS)
        wg = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        print(f"  [{i}] {plat.name}: {dev.name} (CU={cu}, WG={wg})")

    # Select devices
    if opencl_devices is not None:
        selected = []
        for idx in opencl_devices:
            if 0 <= idx < n_devices_expected:
                selected.append(idx)
            else:
                print(f"[-] Invalid device index: {idx}")
                return None
        num_workers = len(selected)
        print(f"[*] Using {num_workers} GPU(s) (selected: {opencl_devices})")
    else:
        # If num_workers not specified, use ALL available GPUs by default
        if num_workers is None:
            num_workers = n_devices_expected
        num_workers = min(num_workers, n_devices_expected)
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

    stop_event = Event()
    found_flags = Array('i', [0] * len(patterns))
    slots = [_GpuSlot(i) for i in selected]
    remaining = set(range(len(patterns)))

    def spawn_slot(slot):
        slot.queue = Queue()
        slot.proc = Process(
            target=worker_gpu,
            args=(slot.device_idx, patterns, case_insensitive,
                  slot.queue, stop_event, found_flags, kernel_path,
                  load_percent, n_devices_expected, batch_mult),
        )
        slot.attempt += 1
        slot.status = _GpuSlot.LAUNCHING
        slot.last_msg = time.monotonic()
        slot.iterations = 0
        slot.rate_iter = 0
        slot.rate_time = time.monotonic()
        slot.proc.start()
        print(f"[+] GPU [{slot.device_idx}] worker started "
              f"(attempt {slot.attempt}/{MAX_GPU_ATTEMPTS})", flush=True)

    def reap_proc(slot):
        proc = slot.proc
        if proc is None:
            return
        try:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive() and proc.pid:
                    os.kill(proc.pid, 9)
        except Exception:
            pass

    def mark_slot_dead(slot, reason):
        """One GPU is gone. Never touches the rest of the farm."""
        first_line = reason.splitlines()[0] if isinstance(reason, str) else str(reason)
        print(f"[!] GPU [{slot.device_idx}] LOST: {first_line}", flush=True)
        reap_proc(slot)
        slot.proc = None
        if stop_event.is_set() or slot.attempt >= MAX_GPU_ATTEMPTS:
            slot.status = _GpuSlot.DEAD
            print(f"[-] GPU [{slot.device_idx}] excluded from the run "
                  f"({slot.attempt} attempts used)", flush=True)
        else:
            backoff = RETRY_BACKOFF_SEC[min(slot.attempt - 1,
                                            len(RETRY_BACKOFF_SEC) - 1)]
            slot.status = _GpuSlot.RETRY
            slot.next_retry = time.monotonic() + backoff
            print(f"[*] GPU [{slot.device_idx}] retry {slot.attempt + 1}/{MAX_GPU_ATTEMPTS} "
                  f"in {backoff}s -- farm continues on the rest", flush=True)

    for slot in slots:
        spawn_slot(slot)

    total_iterations = 0
    start_time = time.monotonic()
    first_printed = False
    progress_line = ""
    last_prog_time = start_time
    last_pub = None
    last_pem = None

    def handle_found(data):
        """CPU's ONLY per-match job: assemble the key from GPU data."""
        nonlocal last_pub, last_pem
        pat_idx = data['pattern_idx']
        if pat_idx not in remaining:
            return  # already handled
        remaining.discard(pat_idx)
        matched_pat = patterns[pat_idx]
        elapsed = time.monotonic() - start_time

        if first_printed and progress_line:
            print(f"\r{' ' * len(progress_line)}\r", end="", flush=True)

        seed_bytes = bytes.fromhex(data['seed'])
        pubkey32 = bytes.fromhex(data['pub'])
        pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pubkey32)

        pub_bytes = pub_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )
        pub_str = pub_bytes.decode()
        priv_pem_str = build_openssh_private_key(pubkey32, seed_bytes)

        print(f"\n[+] Found match for '{matched_pat}'!", flush=True)
        print(f"[+] Public key: {pub_str} {matched_pat}", flush=True)
        if debug_mode:
            print(f"[+] Seed (hex): {data['seed']}", flush=True)

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
                print(f"[+] Written: {base}.pub and {base} (mode 600)", flush=True)
                saved = True
            except Exception as e:
                print(f"[-] Save failed: {e}", flush=True)

        if not saved:
            print("[!] Output to console:", flush=True)
            print(priv_pem_str, flush=True)

        last_pub = pub_str
        last_pem = priv_pem_str

        if remaining:
            print(f"[*] Continuing search for remaining "
                  f"({len(remaining)} left)...", flush=True)
        else:
            print("[*] All patterns found!", flush=True)
            stop_event.set()

    try:
        while True:
            now = time.monotonic()
            live = [s for s in slots
                    if s.status in (_GpuSlot.LAUNCHING, _GpuSlot.RUNNING)]
            if not live:
                break  # every GPU is dead or the global stop is set

            # -- Drain messages from live workers -------------------------
            for slot in list(live):
                while True:
                    try:
                        msg_type, data = slot.queue.get_nowait()
                    except Exception:
                        break
                    slot.last_msg = time.monotonic()
                    if msg_type == 'ready':
                        if slot.status == _GpuSlot.LAUNCHING:
                            slot.status = _GpuSlot.RUNNING
                            print(f"[+] GPU [{slot.device_idx}] ready", flush=True)
                    elif msg_type == 'progress':
                        slot.iterations += data
                        total_iterations += data
                    elif msg_type == 'found':
                        handle_found(data)
                    elif msg_type in ('gpu-dead', 'error'):
                        mark_slot_dead(slot, data)
                    elif msg_type == 'done':
                        # Worker exited cleanly (global stop in progress).
                        slot.status = _GpuSlot.DEAD
                        total_iterations += data
                        break

            # -- Watchdog: per-GPU silence (hang) --------------------------
            for slot in list(live):
                if slot.status not in (_GpuSlot.LAUNCHING, _GpuSlot.RUNNING):
                    continue
                # Fast path: the worker process is already gone (killed,
                # OOM, crash) and its queue is drained -- no point waiting
                # out the full heartbeat timeout.
                if (slot.proc is not None and not slot.proc.is_alive()
                        and not slot.queue.qsize()):
                    mark_slot_dead(slot, "worker process terminated "
                                         "unexpectedly (no message)")
                    continue
                if now - slot.last_msg > WATCHDOG_TIMEOUT:
                    mark_slot_dead(slot, f"no heartbeat for {WATCHDOG_TIMEOUT}s "
                                         f"(watchdog) -- process terminated")

            # -- Spawn due retries ----------------------------------------
            for slot in slots:
                if slot.status == _GpuSlot.RETRY and now >= slot.next_retry:
                    spawn_slot(slot)

            # -- Stop conditions -------------------------------------------
            if not remaining:
                break
            if all(s.status == _GpuSlot.DEAD for s in slots):
                print("\n[-] All GPUs are dead -- nothing left to compute on.",
                      flush=True)
                break

            # -- Periodic progress display (every 5s) -----------------------
            now = time.monotonic()
            if now - last_prog_time >= 5.0:
                elapsed = now - start_time
                rate = total_iterations / elapsed if elapsed > 0 else 0
                parts = []
                for s in slots:
                    if s.status in (_GpuSlot.LAUNCHING, _GpuSlot.RUNNING):
                        dt = now - s.rate_time
                        r = (s.iterations - s.rate_iter) / dt if dt > 0 else 0
                        parts.append(f"GPU{s.device_idx} {r / 1e6:.1f}M/s")
                        s.rate_iter, s.rate_time = s.iterations, now
                    elif s.status == _GpuSlot.RETRY:
                        parts.append(
                            f"GPU{s.device_idx} retry {s.attempt + 1}/{MAX_GPU_ATTEMPTS} "
                            f"in {max(0, int(s.next_retry - now))}s")
                    else:
                        parts.append(f"GPU{s.device_idx} LOST")
                rem = f" ({len(remaining)} left)" if remaining else ""
                progress_line = (
                    f"\r[+] {total_iterations:,} keys "
                    f"({format_duration(elapsed)}) "
                    f"(~{rate:,.0f}/s){rem} | " + " | ".join(parts))
                if first_printed:
                    print(f"\r{' ' * len(progress_line)}\r", end="", flush=True)
                print(progress_line, end="", flush=True)
                first_printed = True
                last_prog_time = now

            if not any(True for s in slots
                       if s.status in (_GpuSlot.LAUNCHING, _GpuSlot.RUNNING)):
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")

    # Shutdown: stop everything we own
    stop_event.set()
    for slot in slots:
        reap_proc(slot)
        slot.proc = None

    if first_printed:
        print()

    elapsed = time.monotonic() - start_time
    if last_pub:
        rate = total_iterations / elapsed if elapsed > 0 else 0
        print(f"\n[+] Checked keys: {total_iterations:,} "
              f"({format_duration(elapsed)}) (~{rate:,.0f} keys/sec)", flush=True)
        return last_pub, last_pem, total_iterations, elapsed
    else:
        print(f"[+] Search ended. Iterations: {total_iterations:,}", flush=True)
        return None


# --- CLI ----------------------------------------------------------------

"""CLI entry point: parse args, validate patterns, call generate_vanity_key."""
def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__.strip())
        sys.exit(0)

    # `kill <pid>` (SIGTERM) must stop the farm as cleanly as Ctrl-C.
    # Background jobs of non-interactive shells inherit SIG_IGN for
    # SIGINT, so Python never installs the KeyboardInterrupt handler
    # there -- route SIGTERM to the same shutdown path.
    def _term_to_kbi(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term_to_kbi)

    pattern = None
    patterns_file = None
    case_insensitive = '-i' in sys.argv or '--ignore-case' in sys.argv
    debug_mode = '--debug' in sys.argv
    num_workers = None
    output_file = None
    opencl_devices = None
    load_percent = 100
    batch_mult = 1

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
                    print("--load-percent must be 1..100")
                    sys.exit(1)
            except ValueError:
                print("--load-percent must be an integer")
                sys.exit(1)
            i += 2
        elif arg == '--batch-mult' and i + 1 < len(sys.argv):
            try:
                batch_mult = int(sys.argv[i + 1])
                if batch_mult < 1 or batch_mult > 16:
                    print("--batch-mult must be 1..16")
                    sys.exit(1)
            except ValueError:
                print("--batch-mult must be an integer")
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
        out_dir = os.path.dirname(output_file)
        if out_dir:
            if not os.path.isdir(out_dir):
                print(f"[-] Warning: Output directory does not exist: {out_dir}")
            elif not os.access(out_dir, os.W_OK):
                print(f"[-] Warning: No write permission to {out_dir}")

    result = generate_vanity_key(
        valid_patterns,
        case_insensitive=case_insensitive,
        num_workers=num_workers,
        output_file=output_file,
        debug_mode=debug_mode,
        opencl_devices=opencl_devices,
        load_percent=load_percent,
        batch_mult=batch_mult,
    )

    if result:
        pub, pem, total_iter, duration = result
        print(f"[+] Total time: {format_duration(duration)}")
        print(f"[+] Checked keys: {total_iter:,}")


if __name__ == "__main__":
    main()
