#!/usr/bin/env python3

"""
     Ed-25519 SSH Vanity Key Generator [Multi CPU thread, FAST]
     100% AI-Generated: QWEN-Coder-Next 80B/3B
     Inspired by Aminuxer
     Version 2026-07-17-cN

Usage:
    python3 ssh_ed25519_vanity_multicpu.py <pattern> [-i] [-w <workers>] [-o output_file] [--debug]

Examples:
    python3 ssh_ed25519_vanity_multicpu.py User -i -w 8
    python3 ssh_ed25519_vanity_multicpu.py mykey -o mykey --debug

The generated OpenSSH private key can be used with:
    ssh -i mykey vanity@host

Options:
    -i          Case-insensitive pattern matching
    -w <N>      Number of worker processes (default: CPU count)
    -o <file>   Save keys to files (output_file.pub and output_file)
    --debug     Show seed hex in output (for testing/diagnosis)
"""



import os
import sys
import multiprocessing as mp
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from typing import Optional, Tuple
import time
import base64

# Use monotonic for reliable time measurement
try:
    _time_func = time.monotonic
except AttributeError:
    _time_func = time.time

# Valid Base64 characters for OpenSSH public key (no padding '=' used)
B64_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")

def validate_pattern(pattern: str) -> bool:
    """Check if pattern contains only valid Base64 characters"""
    for char in pattern:
        if char not in B64_CHARS:
            return False
    return True

def worker_loop(pattern: str, case_insensitive: bool, result_queue, stop_event, seed_queue, progress_queue, debug_mode: bool):
    """Worker process that generates keys and checks for pattern"""
    pattern_lower = pattern.lower() if case_insensitive else pattern
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

            # Check if pattern matches in the variable part only
            if len(pubkey_b64) > variable_part_start:
                search_str = pubkey_b64[variable_part_start:].lower() if case_insensitive else pubkey_b64[variable_part_start:]
            else:
                search_str = pubkey_b64.lower() if case_insensitive else pubkey_b64
            if pattern_lower in search_str:
                # Send both public key and seed (seed always for key reconstruction)
                # First send remaining iterations, then found result
                if iterations > 0:
                    progress_queue.put(iterations)
                result_queue.put(('found', (public_str, seed.hex())))
                return

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

def generate_vanity_key(pattern: str, case_insensitive: bool = False,
                        num_workers: Optional[int] = None,
                        output_file: Optional[str] = None,
                        debug_mode: bool = False) -> Optional[Tuple[str, str, int]]:
    """
    Generate ED25519 vanity key with multi-threading

    Args:
        pattern: Pattern to search for in base64 part of public key
        case_insensitive: If True, ignore case when matching pattern
        num_workers: Number of worker processes (default: CPU count)
        output_file: If provided, save keys to files
        debug_mode: If True, show seed hex in output

    Returns:
        Tuple of (public_key_string, private_key_pem, total_iterations) or None if failed
    """
    pattern_len = len(pattern)
    if pattern_len > 44:
        print("[-] Pattern too long (max 44 chars)")
        return None

    if pattern_len == 0:
        print("[-] Pattern cannot be empty")
        return None

    # Validate pattern contains only valid Base64 characters
    if not validate_pattern(pattern):
        invalid_chars = set(pattern) - B64_CHARS
        print(f"[-] Pattern contains invalid characters: {sorted(invalid_chars)}")
        print(f"    Valid Base64 characters: A-Za-z0-9+/")
        return None

    print(f"[*] Searching for pattern: {pattern}")
    print(f"[*] Case insensitive: {case_insensitive}")
    print(f"[*] Debug mode: {debug_mode}")

    # Default to CPU count workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    print(f"[*] Using {num_workers} worker processes")

    # Use spawn start method for better compatibility
    mp.set_start_method('spawn', force=True)

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
        p = mp.Process(target=worker_loop, args=(pattern, case_insensitive, result_queue, stop_event, seed_queue, progress_queue, debug_mode))
        p.start()
        workers.append(p)

    # Monitor progress and collect results
    total_iterations = 0
    found_result = None
    start_time = _time_func()
    last_progress_time = _time_func()
    last_progress_iter = 0
    last_rate = 0
    first_progress_printed = False
    current_progress_line = ""

    try:
        while True:
            try:
                msg_type, data = result_queue.get(timeout=5)

                if msg_type == 'found':
                    found_result = data
                    # Drain progress queue to get all accumulated iterations
                    try:
                        while True:
                            delta = progress_queue.get_nowait()
                            total_iterations += delta
                    except:
                        pass
                    break
                elif msg_type == 'error':
                    print(f"[-] Worker error: {data}")
                    break
                elif msg_type == 'done':
                    total_iterations += data

            except mp.queues.Empty:
                # Check progress from progress_queue
                try:
                    while True:
                        delta = progress_queue.get_nowait()
                        total_iterations += delta
                except:
                    pass

                # Debug: show what we have
                # print(f"DEBUG: elapsed={elapsed:.1f}, total={total_iterations}, last_iter={last_progress_iter}", file=sys.stderr)
                # sys.stderr.flush()

                # Print progress every 5 seconds (without newline in same line)
                current_time = _time_func()
                elapsed_total = current_time - start_time
                elapsed_interval = current_time - last_progress_time
                if elapsed_interval >= 5:
                    if total_iterations > last_progress_iter:
                        rate = (total_iterations - last_progress_iter) / elapsed_interval
                        last_rate = rate
                        current_progress_line = f"\r[+] Progress: {total_iterations:,} keys ({format_duration(elapsed_total)}) (avg ~{rate:,.0f} keys/sec)"
                    else:
                        current_progress_line = f"\r[+] Progress: {total_iterations:,} keys ({format_duration(elapsed_total)}) (no new keys)"
                    print(current_progress_line, end="", flush=True)
                    last_progress_time = current_time
                    last_progress_iter = total_iterations
                    first_progress_printed = True

            except Exception as e:
                print(f"[+] Monitoring error: {e}")
                break

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
    if found_result:
        public_str, seed_hex = found_result
        # Clear any remaining progress line
        if first_progress_printed and current_progress_line:
            print()  # Newline after progress bar
        print(f"\n[+] Found match!")
        print(f"[+] Pattern: {pattern}")
        print(f"[+] Public key: {public_str}")
        # Only show seed in debug mode
        if debug_mode:
            print(f"[+] Seed (hex): {seed_hex}")
        # Calculate and show average rate using last_rate if available
        elapsed_total = _time_func() - start_time
        rate = last_rate if last_rate > 0 else (total_iterations / elapsed_total if elapsed_total > 0 else 0)
        print(f"[+] Checked keys: {total_iterations:,} (avg ~{rate:,.0f} keys/sec)")
        # Generate private key from seed (always available in result)
        seed = bytes.fromhex(seed_hex)
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption()
        )
        print("\n" + private_pem.decode())

        # Write to files if output_file specified
        if output_file:
            # Validate output file path first
            output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
            if not os.path.isdir(output_dir):
                raise FileNotFoundError(f"Directory does not exist: {output_dir}")

            # Check write permission
            if not os.access(output_dir, os.W_OK):
                raise PermissionError(f"Cannot write to directory: {output_dir}")

            # Public key file
            try:
                with open(output_file + '.pub', 'w') as f:
                    f.write(public_str + '\n')
            except (IOError, OSError) as e:
                raise IOError(f"Failed to write public key file: {output_file}.pub - {e}")
            print(f"[+] Written: {output_file}.pub")

            # Private key file
            try:
                with open(output_file, 'w') as f:
                    f.write(private_pem.decode())
                os.chmod(output_file, 0o600)
            except (IOError, OSError) as e:
                raise IOError(f"Failed to write private key file: {output_file} - {e}")
            print(f"[+] Written: {output_file} (mode 600)")

        return public_str, private_pem.decode(), total_iterations, last_rate
    else:
        # Ensure we print a final newline after progress if it was overwritten
        if first_progress_printed and current_progress_line:
            print()  # Newline after progress bar
        elapsed = _time_func() - last_progress_time if 'last_progress_time' in locals() else 0
        # For very fast searches (< 5 seconds), total_iterations might be 0 if workers
        # didn't have time to send progress data. At minimum, we checked at least 1 key.
        if total_iterations == 0:
            total_iterations = 1
        # Use last_rate from progress if available, otherwise calculate from total
        if total_iterations > 0 and elapsed > 0:
            rate = last_rate if last_rate > 0 else total_iterations / elapsed
            print(f"[+] Search completed. Total iterations: {total_iterations:,} (avg ~{rate:,.0f} keys/sec)")
        else:
            print(f"[+] Search completed. Total iterations: {total_iterations:,}")

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

if __name__ == "__main__":
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: ssh_ed25519_vanity_multicpu.py <pattern> [-i] [-w <workers>] [-o output_file] [--debug]")
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

    pattern = sys.argv[1]
    case_insensitive = '-i' in sys.argv
    debug_mode = '--debug' in sys.argv
    num_workers = None
    output_file = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '-w' and i + 1 < len(sys.argv):
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
            i += 1

    # Start timing
    start_time = _time_func()
    result = generate_vanity_key(pattern, case_insensitive, num_workers, output_file, debug_mode)

    if result:
        duration = _time_func() - start_time
        public_str, private_pem, total_iterations, last_rate = result
        # For very fast searches (< 5 seconds), total_iterations might be 0 if workers
        # didn't have time to send progress data. At minimum, we found 1 key.
        if total_iterations == 0:
            total_iterations = 1
        print(f"[+] Total time: {format_duration(duration)}")
        print(f"[+] Checked keys: {total_iterations:,}")

