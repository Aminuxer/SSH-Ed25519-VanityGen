#!/usr/bin/env python3

"""
     Ed-25519 SSH Vanity Key Generator [Multi CPU thread, FAST]
     100% AI-Generated: QWEN-Coder-Next 80B/3B
     Inspired by Aminuxer
     Version 2026-07-09-cN

Usage:
    python3 ssh_ed25519_vanity_multicpu.py <pattern> [-i] [-w <workers>] [-o output_file]

Examples:
    python3 ssh_ed25519_vanity_multicpu.py User -i -w 8
    python3 ssh_ed25519_vanity_multicpu.py mykey -o mykey

The generated OpenSSH private key can be used with:
    ssh -i mykey vanity@host
"""


import os
import sys
import multiprocessing as mp
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from typing import Optional, Tuple
import time
import base64

# Valid Base64 characters for OpenSSH public key (no padding '=' used)
B64_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")

def validate_pattern(pattern: str) -> bool:
    """Check if pattern contains only valid Base64 characters"""
    for char in pattern:
        if char not in B64_CHARS:
            return False
    return True

def worker_loop(pattern: str, case_insensitive: bool, result_queue, stop_event, seed_queue, progress_queue):
    """Worker process that generates keys and checks for pattern"""
    pattern_lower = pattern.lower() if case_insensitive else pattern
    iterations = 0
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

            iterations += 1

            # Check if pattern matches
            search_str = pubkey_b64.lower() if case_insensitive else pubkey_b64
            if pattern_lower in search_str:
                # Send both public key and seed
                seed_hex = seed.hex()
                result_queue.put(('found', (public_str, seed_hex)))
                return

            # Check stop event between batches
            if iterations % 100000 == 0:
                progress_queue.put(iterations)

        except Exception as e:
            result_queue.put(('error', str(e)))
            return

    result_queue.put(('done', iterations))

def generate_vanity_key(pattern: str, case_insensitive: bool = False,
                        num_workers: Optional[int] = None,
                        output_file: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Generate ED25519 vanity key with multi-threading

    Args:
        pattern: Pattern to search for in base64 part of public key
        case_insensitive: If True, ignore case when matching pattern
        num_workers: Number of worker processes (default: CPU count)
        output_file: If provided, save keys to files

    Returns:
        Tuple of (public_key_string, private_key_pem) or None if failed
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
        p = mp.Process(target=worker_loop, args=(pattern, case_insensitive, result_queue, stop_event, seed_queue, progress_queue))
        p.start()
        workers.append(p)

    # Monitor progress and collect results
    total_iterations = 0
    found_result = None
    last_progress_time = time.time()
    last_progress_iter = 0

    try:
        while True:
            try:
                msg_type, data = result_queue.get(timeout=5)

                if msg_type == 'found':
                    found_result = data
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

                # Print progress every 10 seconds
                current_time = time.time()
                if current_time - last_progress_time >= 10:
                    elapsed = current_time - last_progress_time
                    if total_iterations > last_progress_iter:
                        rate = (total_iterations - last_progress_iter) / elapsed
                        print(f"[+] Progress: {total_iterations:,} keys checked (avg ~{rate:,.0f} keys/sec)")
                    else:
                        print(f"[+] Progress: {total_iterations:,} keys checked (no new keys)")
                    last_progress_time = current_time
                    last_progress_iter = total_iterations

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

    # Print final stats
    if found_result:
        public_str, seed_hex = found_result
        print(f"\n[+] Found match!")
        print(f"[+] Pattern: {pattern}")
        print(f"[+] Public key: {public_str}")
        print(f"[+] Seed (hex): {seed_hex}")

        # Generate private key from seed
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

        return public_str, private_pem.decode()
    else:
        elapsed = time.time() - last_progress_time if 'last_progress_time' in locals() else 0
        if total_iterations > 0 and elapsed > 0:
            rate = total_iterations / elapsed
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
    if len(sys.argv) < 2:
        print("Usage: ssh_ed25519_vanity_multicpu.py <pattern> [-i] [-w <workers>] [-o output_file]")
        sys.exit(1)

    pattern = sys.argv[1]
    case_insensitive = '-i' in sys.argv
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
        elif sys.argv[i].startswith('-'):
            i += 1
        else:
            i += 1

    # Start timing
    start_time = time.time()
    result = generate_vanity_key(pattern, case_insensitive, num_workers, output_file)

    if result:
        duration = time.time() - start_time
        print(f"[+] Total time: {format_duration(duration)}")

