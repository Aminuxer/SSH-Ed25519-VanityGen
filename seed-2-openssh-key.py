#!/usr/bin/env python3

"""
Generate OpenSSH Ed25519 key from seed hex
Usage: seed-2-openssh-key.py <seed_hex> [-o output_file]
"""


import sys
import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__.strip())
        sys.exit(0)

    seed_hex = sys.argv[1]

    # Validate seed
    if len(seed_hex) != 64:
        print(f"Error: seed must be exactly 64 hex characters, got {len(seed_hex)}", file=sys.stderr)
        sys.exit(1)

    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError:
        print("Error: invalid hex string for seed", file=sys.stderr)
        sys.exit(1)

    # Generate key
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
    public_key = private_key.public_key()

    # Get public key in OpenSSH format
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH
    )
    public_str = public_bytes.decode()

    # Print public key
    print(public_str)

    # Generate and print private key in PEM format
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption()
    )
    print()
    print(private_pem.decode(), end='')
    
    # Save to files if output_file specified
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
            output_dir = os.path.dirname(output_file) or '.'

            if not os.path.isdir(output_dir):
                print(f"Error: directory does not exist: {output_dir}", file=sys.stderr)
                sys.exit(1)
            if not os.access(output_dir, os.W_OK):
                print(f"Error: cannot write to directory: {output_dir}", file=sys.stderr)
                sys.exit(1)

            try:
                with open(output_file + '.pub', 'w') as f:
                    f.write(public_str + '\n')
            except (IOError, OSError) as e:
                print(f"Error: failed to write public key file: {output_file}.pub - {e}", file=sys.stderr)
                sys.exit(1)

            try:
                with open(output_file, 'w') as f:
                    f.write(private_pem.decode())
                os.chmod(output_file, 0o600)
            except (IOError, OSError) as e:
                print(f"Error: failed to write private key file: {output_file} - {e}", file=sys.stderr)
                sys.exit(1)

            print(f"Written: {output_file}.pub", file=sys.stderr)
            print(f"Written: {output_file} (mode 600)", file=sys.stderr)

if __name__ == "__main__":
    main()
