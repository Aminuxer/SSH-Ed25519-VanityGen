#!/usr/bin/env python3


"""
     Ed-25519 SSH Vanity Key Generator [Multi CPU thread, FAST]
     100% AI-Generated: QWEN-Coder-Next 80B/3B
     Inspired by Aminuxer     https://github.com/Aminuxer/SSH-Ed25519-VanityGen
     Version 2026-07-24-cN

Usage:
    python3 ssh_ed25519_vanity_multicpu.py <pattern> [-i] [-w <workers>] [-o output_file] [--debug]
    python3 ssh_ed25519_vanity_multicpu.py --patterns-file patterns.txt [-i] [-w <workers>] [-o output_prefix] [--debug]

Examples:
    python3 ssh_ed25519_vanity_multicpu.py User -i -w 8
    python3 ssh_ed25519_vanity_multicpu.py mykey -o mykey --debug

The generated OpenSSH private key can be used with:
    ssh -i mykey vanity@host

Options:
    -i          Case-insensitive pattern matching
    -w <N>      Number of worker processes (default: CPU count)
    -o <file>   Save keys to files (output_file.pub and output_file)
                With --patterns-file: prefix-pattern-YYYYMMDD-HHMMSS
    --debug     Show seed hex in output (for testing/diagnosis)
"""


import os
import sys
import multiprocessing as mp
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from typing import Optional, Tuple, List
import time
import base64
import re

# Use monotonic for reliable time measurement
try:
    _time_func = time.monotonic
except AttributeError:
    _time_func = time.time

# Valid Base64 characters for OpenSSH public key (no padding '=' used)
B64_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")

def format_duration(seconds: float) -> str:
    """Format duration as days, hours, minutes, seconds"""
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

def validate_pattern(pattern: str) -> bool:
    """Check if pattern contains only valid Base64 characters"""
    if len(pattern) > 44:
        return False
    for char in pattern:
        if char not in B64_CHARS:
            return False
    return True

def sanitize_filename(name: str) -> str:
    """Replace invalid filename characters with underscores"""
    return re.sub(r'[^a-zA-Z0-9\-_.]', '_', name)

def worker_loop(patterns: List[str], case_insensitive: bool, result_queue, stop_event, seed_queue, progress_queue, debug_mode: bool):
    """Worker process that generates keys and checks for pattern"""
    pat_checks = [(p, p.lower() if case_insensitive else p) for p in patterns]
    iterations = 0
    last_progress_time = _time_func()
    batch_size = 10000

    while not stop_event.is_set():
        try:
            # Get seed from queue if not empty, otherwise generate random
            try:
                seed = seed_queue.get_nowait()
            except:
                seed = os.urandom(32)

            # Generate ED25519 key from seed
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
            public_key = private_key.public_key()

            # Get public key in OpenSSH format (base64 part only)
            public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.OpenSSH,
                format=serialization.PublicFormat.OpenSSH
            )
            public_str = public_bytes.decode()
            pubkey_b64 = public_str.split()[1]

            # Ed25519 public key format: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI<32bytes>
            # The prefix "AAAAC3NzaC1lZDI1NTE5AAAAI" is always the same (25 chars)
            # We only search in the variable part after the fixed prefix
            variable_part_start = 25  # Position after "AAAAC3NzaC1lZDI1NTE5AAAAI"

            iterations += 1

            # Check against all patterns
            matched_pattern = None
            if len(pubkey_b64) > variable_part_start:
                search_str = pubkey_b64[variable_part_start:].lower() if case_insensitive else pubkey_b64[variable_part_start:]
            else:
                search_str = pubkey_b64.lower() if case_insensitive else pubkey_b64

            for pat, pat_chk in pat_checks:
                if pat_chk in search_str:
                    matched_pattern = pat
                    break

            if matched_pattern:
                # Send both public key, seed and matched pattern
                if iterations > 0:
                    progress_queue.put(iterations)
                result_queue.put(('found', (matched_pattern, public_str, seed.hex())))
                iterations = 0  # Reset counter, DO NOT return. Keep searching for other patterns.
                # continue to next key

            # Check progress by time - send every 5 seconds of wall-clock time
            current_time = _time_func()
            if current_time - last_progress_time >= 5.0:
                progress_queue.put(iterations)
                iterations = 0
                last_progress_time = current_time

        except Exception as e:
            result_queue.put(('error', str(e)))
            return

    result_queue.put(('done', iterations))

def generate_vanity_key(patterns: List[str], case_insensitive: bool = False,
                        num_workers: Optional[int] = None,
                        output_file: Optional[str] = None,
                        debug_mode: bool = False) -> Optional[Tuple[str, str, int, float]]:
    """
    Generate ED25519 vanity key with multi-threading

    Args:
        patterns: List of patterns to search for
        case_insensitive: If True, ignore case when matching pattern
        num_workers: Number of worker processes (default: CPU count)
        output_file: If provided, save keys to files
        debug_mode: If True, show seed hex in output

    Returns:
        Tuple of (public_key_string, private_key_pem, total_iterations, avg_rate) or None if failed
    """
    if not patterns:
        print("[-] No valid patterns to search for")
        return None

    # 1). Вывод списка принятых паттернов
    print(f"[*] Accepted patterns: {', '.join(patterns)}")
    print(f"[*] Case insensitive: {case_insensitive}")
    print(f"[*] Debug mode: {debug_mode}")

    # Default to CPU count workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    print(f"[*] Using {num_workers} worker processes")

    # Use spawn start method for better compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Create Queues and Events
    result_queue = mp.Queue()
    stop_event = mp.Event()
    seed_queue = mp.Queue()
    progress_queue = mp.Queue()

    # Pre-fill seed queue with random seeds
    for _ in range(num_workers * 100):
        seed_queue.put(os.urandom(32))

    # Start worker processes
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=worker_loop, args=(patterns, case_insensitive, result_queue, stop_event, seed_queue, progress_queue, debug_mode))
        p.start()
        workers.append(p)

    # Monitor progress and collect results
    total_iterations = 0
    start_time = _time_func()
    last_progress_time = _time_func()
    last_progress_iter = 0
    last_rate = 0
    first_progress_printed = False
    current_progress_line = ""

    # Track remaining patterns
    remaining = set(patterns)
    last_found_pub = None
    last_found_pem = None

    try:
        while remaining:
            # Always drain progress queue to keep total_iterations accurate
            while True:
                try:
                    total_iterations += progress_queue.get_nowait()
                except:
                    break

            try:
                msg_type, data = result_queue.get(timeout=5)
            except mp.queues.Empty:
                # Print progress every 5 seconds (without newline in same line)
                current_time = _time_func()
                elapsed_total = current_time - start_time
                elapsed_interval = current_time - last_progress_time
                if elapsed_interval >= 5.0:
                    if total_iterations > last_progress_iter:
                        rate = (total_iterations - last_progress_iter) / elapsed_interval
                        last_rate = rate
                        rem_info = f" ({len(remaining)} left)" if remaining else ""
                        current_progress_line = f"\r[+] Progress: {total_iterations:,} keys ({format_duration(elapsed_total)}) (avg ~{rate:,.0f} keys/sec){rem_info}"
                    else:
                        rem_info = f" ({len(remaining)} left)" if remaining else ""
                        current_progress_line = f"\r[+] Progress: {total_iterations:,} keys ({format_duration(elapsed_total)}) (no new keys){rem_info}"
                    print(current_progress_line, end="", flush=True)
                    last_progress_time = current_time
                    last_progress_iter = total_iterations
                    first_progress_printed = True
                continue

            if msg_type == 'found':
                matched_pat, public_str, seed_hex = data
                if matched_pat in remaining:
                    remaining.remove(matched_pat)

                    # Clear progress line
                    if first_progress_printed and current_progress_line:
                        print("\r" + " " * len(current_progress_line) + "\r", end="", flush=True)

                    elapsed_total = _time_func() - start_time
                    print(f"\n[+] Found match for '{matched_pat}'!")
                    print(f"[+] Public key: {public_str} {matched_pat}")
                    if debug_mode:
                        print(f"[+] Seed (hex): {seed_hex}")

                    # Generate private key immediately
                    seed = bytes.fromhex(seed_hex)
                    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
                    private_pem_bytes = private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.OpenSSH,
                        encryption_algorithm=serialization.NoEncryption()
                    )

                    # Вставка комментария после строки BEGIN
                    private_pem_str = private_pem_bytes.decode()
                    date_str = time.strftime("%Y-%m-%d__%T")
                    comment_line = f"# Generated by Aminuxer & AI SSH-Ed25519-VanityGen; Pattern {matched_pat}; Date {date_str}\n"
                    lines = private_pem_str.splitlines()
                    # lines.insert(1, comment_line)
                    private_pem_mod = "\n".join(lines) + "\n"

                    last_found_pub = public_str
                    last_found_pem = private_pem_mod

                    # Безопасное сохранение с фолбэком в консоль
                    saved_successfully = False
                    if output_file:
                        try:
                            ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                            safe_pat = sanitize_filename(matched_pat)
                            base_name = f"{output_file}-{safe_pat}-{ts}"
                            output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'

                            if not os.path.isdir(output_dir):
                                raise FileNotFoundError(f"Directory does not exist: {output_dir}")
                            if not os.access(output_dir, os.W_OK):
                                raise PermissionError(f"Cannot write to directory: {output_dir}")

                            with open(base_name + '.pub', 'w') as f:
                                f.write(public_str + ' '+ matched_pat + '\n')
                            with open(base_name, 'w') as f:
                                f.write(private_pem_mod)
                            os.chmod(base_name, 0o600)
                            print(f"[+] Written: {base_name}.pub and {base_name} (mode 600)")
                            saved_successfully = True
                        except Exception as e:
                            print(f"[-] Warning: Failed to save files: {e}")

                    if not saved_successfully:
                        print("[!] Output to console (file save skipped or failed):")
                        print(private_pem_mod)

                    print("[*] Continuing search for remaining patterns...")
                    if not remaining:
                        break

            elif msg_type == 'error':
                print(f"[-] Worker error: {data}")
                break
            elif msg_type == 'done':
                total_iterations += data

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        stop_event.set()

    # Stop all workers
    stop_event.set()
    for p in workers:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)

    # Clear progress line if printed
    if first_progress_printed and current_progress_line:
        print()  # Newline after progress bar

    # Print final stats
    if last_found_pub:
        elapsed_total = _time_func() - start_time
        avg_rate = total_iterations / elapsed_total if elapsed_total > 0 else 0
        print(f"\n[+] Checked keys: {total_iterations:,} (avg ~{avg_rate:,.0f} keys/sec)")
        return last_found_pub, last_found_pem, total_iterations, avg_rate
    else:
        elapsed = _time_func() - last_progress_time if 'last_progress_time' in locals() else 0
        if total_iterations == 0:
            total_iterations = 1
        if total_iterations > 0 and elapsed > 0:
            rate = last_rate if last_rate > 0 else total_iterations / elapsed
            print(f"[+] Search completed. Total iterations: {total_iterations:,} (avg ~{rate:,.0f} keys/sec)")
        else:
            print(f"[+] Search completed. Total iterations: {total_iterations:,}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: ssh_ed25519_vanity_multicpu.py <pattern> [-i] [-w <workers>] [-o output_file] [--debug]")
        print("Usage: ssh_ed25519_vanity_multicpu.py --patterns-file <pattern-file.txt> [-i] [-w <workers>] [-o output_filesnames_prefix] [--debug]")
        print()
        print("Examples:")
        print("    python3 ssh_ed25519_vanity_multicpu.py User -i -w 8")
        print("    python3 ssh_ed25519_vanity_multicpu.py mykey -o mykey --debug")
        print()
        print("The generated OpenSSH private key can be used with:")
        print("    ssh -i mykey vanity@host")
        print()
        print("Options:")
        print("    -i          Case-insensitive pattern matching")
        print("    -w <N>      Number of worker processes (default: CPU count)")
        print("    -o <file>   Save keys to files (output_file.pub and output_file)")
        print("    --debug     Show seed hex in output (for testing/diagnosis)")
        sys.exit(0)

    pattern = None
    patterns_file = None
    case_insensitive = '-i' in sys.argv
    debug_mode = '--debug' in sys.argv
    num_workers = None
    output_file = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--patterns-file' and i + 1 < len(sys.argv):
            patterns_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '-w' and i + 1 < len(sys.argv):
            try:
                num_workers = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                i += 1
        elif sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--debug':
            debug_mode = True
            i += 1
        elif sys.argv[i].startswith('-'):
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
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            if validate_pattern(line):
                valid_patterns.append(line)
            else:
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

    # Предварительная проверка каталога вывода
    if output_file:
        out_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        if not os.path.isdir(out_dir):
            print(f"[-] Warning: Output directory does not exist: {out_dir}. Keys will be printed to console.")
        elif not os.access(out_dir, os.W_OK):
            print(f"[-] Warning: No write permission for output directory: {out_dir}. Keys will be printed to console.")

    # Start timing
    start_time = _time_func()
    result = generate_vanity_key(valid_patterns, case_insensitive, num_workers, output_file, debug_mode)

    if result:
        duration = _time_func() - start_time
        public_str, private_pem, total_iterations, last_rate = result
        if total_iterations == 0:
            total_iterations = 1
        print(f"[+] Total time: {format_duration(duration)}")
        print(f"[+] Checked keys: {total_iterations:,}")
