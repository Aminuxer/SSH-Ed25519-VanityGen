#!/usr/bin/env python3

"""Test suite for openssh.cl - OpenSSH Ed25519 public key generation

GPU generates the full SSH public key line from a 32-byte seed:
  "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5... <comment>"

The base64 blob (AAAAC3NzaC1lZDI1NTE5...) is computed INSIDE the GPU kernel.

Usage:
  --func BASE64       Run built-in base64 test suite
  --func BASE64 --data "hello"       Custom: base64 encode "hello"
  --func BASE64 --hex-data 48656c6c6f  Custom: base64 encode 0x48656c6c6f

  --func OPENKEY      Run built-in openkey test suite (STATUS.md seeds)
  --func OPENKEY --hex-data <64 hex chars>   Custom: test pubkey from seed

  --func BLOB         Run built-in blob test suite
  --func BLOB --hex-data <64 hex chars>      Custom: test blob from raw pubkey

  --func ALL          Run all built-in test suites
"""

import sys, argparse, base64, struct, hashlib

try:
    import pyopencl as cl
except ImportError as e:
    print("Error: pyopencl required - " + str(e)); sys.exit(1)

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# --- Reference: Base64 encoding -------------------------

"""CPU reference: base64 encode bytes using Python standard library."""
def ref_base64_encode(data: bytes) -> bytes:
    """Reference Base64 encoding via Python stdlib."""
    return base64.b64encode(data)

# --- Reference: SSH public key blob ---------------------

"""CPU reference: build SSH public key binary blob (51 bytes)."""
def ref_ssh_public_blob(pub_key: bytes) -> bytes:
    """Build SSH public key binary blob: uint32_BE(11) || "ssh-ed25519" || uint32_BE(32) || pubkey(32)"""
    type_str = b"ssh-ed25519"
    blob = struct.pack(">I", len(type_str)) + type_str
    blob += struct.pack(">I", len(pub_key)) + pub_key
    return blob

"""CPU reference: build full "ssh-ed25519 <b64> <comment>" line."""
def ref_ssh_public_line(pub_key: bytes, comment: bytes = b"") -> bytes:
    """Build full SSH public key line: 'ssh-ed25519 <base64> <comment>'"""
    blob = ref_ssh_public_blob(pub_key)
    b64 = base64.b64encode(blob)
    line = b"ssh-ed25519 " + b64
    if comment:
        line += b" " + comment
    return line

# --- GPU runners -----------------------------------------

"""Find and return the first GPU OpenCL device."""
def get_device():
    for p in cl.get_platforms():
        for d in p.get_devices():
            if d.type == cl.device_type.GPU:
                return d
    raise RuntimeError("No GPU")

"""Run GPU base64_encode kernel and return decoded string."""
def run_base64(kc, data: bytes):
    """Run base64_encode kernel."""
    import math
    dev = get_device()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    buf = struct.pack("<I", len(data)) + data
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=buf)
    out_size = (math.ceil(len(data) / 3) * 4) + 1
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out_size)
    prg.base64_encode(queue, (1,), None, bi, bo)
    queue.finish()
    r = bytearray(out_size)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)

"""Run GPU build_ssh_public_blob kernel."""
def run_build_blob(kc, pub_key: bytes):
    """Run build_ssh_public_blob kernel (wrapper from openssh_test_kernels.cl)."""
    dev = get_device()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=pub_key)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 51)
    prg.build_ssh_public_blob(queue, (1,), None, bi, bo)
    queue.finish()
    r = bytearray(51)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)

"""Run GPU seed_to_ssh_ed25519_pubkey kernel (full pipeline)."""
def run_openkey(kc, seed: bytes, comment: bytes = b""):
    """Run seed_to_ssh_ed25519_pubkey kernel -- returns (pubKey, pubLine)."""
    dev = get_device()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    bi_seed = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=seed)
    cbuf = struct.pack("<I", len(comment)) + comment + b'\x00' * (64 - len(comment))
    bi_comment = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=cbuf)
    bo_pubkey = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 32)
    bo_publine = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 256)
    prg.seed_to_ssh_ed25519_pubkey(queue, (1,), None,
                                    bi_seed, bi_comment, bo_pubkey, bo_publine)
    queue.finish()
    pub_key = bytearray(32)
    pub_line = bytearray(256)
    cl.enqueue_copy(queue, pub_key, bo_pubkey, is_blocking=True)
    cl.enqueue_copy(queue, pub_line, bo_publine, is_blocking=True)
    return bytes(pub_key), bytes(pub_line)

# --- Pipeline debug functions --------------------------------

"""Run GPU SHA512 on 32-byte seed, return 64-byte hash."""
def run_sha512_32(kc, seed: bytes) -> bytes:
    """Run SHA512-32 kernel."""
    dev = get_device()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=seed)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 64)
    prg.sha512_32(queue, (1,), None, bi, bo)
    queue.finish()
    r = bytearray(64)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)

"""Run GPU scalar_mult and return 96-byte projective point."""
def run_scalar_mult(kc, scalar: bytes) -> bytes:
    """Run scalar_mult wrapper: 32-byte LE scalar -> 96-byte projective point."""
    dev = get_device()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scalar)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 96)
    prg.scalar_mult_wrapper(queue, (1,), None, bi, bo)
    queue.finish()
    r = bytearray(96)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)

"""Run GPU point_to_affine_y and return 32-byte LE affine Y."""
def run_point_to_affine_y(kc, point: bytes) -> bytes:
    """Run point_to_affine_y wrapper: 96-byte projective -> 32-byte LE Y."""
    dev = get_device()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=point)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 32)
    prg.point_to_affine_y_wrapper(queue, (1,), None, bi, bo)
    queue.finish()
    r = bytearray(32)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)

"""Run GPU clamp_and_encode: scalar -> public key (32 bytes)."""
def run_clamp_and_encode(kc, scalar_in: bytes) -> bytes:
    """Run clamp+encode: 32-byte LE scalar -> 32-byte pubkey."""
    dev = get_device()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=scalar_in)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 32)
    prg.clamp_and_encode(queue, (1,), None, bi, bo)
    queue.finish()
    r = bytearray(32)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)

# --- Reference helpers for pipeline -------------------------

"""CPU reference: clamp 32-byte hash to Ed25519 scalar."""
def ref_clamp_scalar(hash_bytes: bytes) -> bytes:
    """Clamp first 32 bytes of SHA512 output to LE scalar (NaCl/cryptography)."""
    expanded = hash_bytes[:32]
    scalar = bytearray(expanded)  # no reversal, LE interpretation
    scalar[0] &= 0xF8     # clear bottom 3 bits of LSB
    scalar[31] &= 0x7F   # clear top bit of MSB
    scalar[31] |= 0x40   # set top-1 bit of MSB
    return bytes(scalar)

"""CPU reference: scalar multiplication on Ed25519, returns (x, y)."""
def ref_scalar_mult(scalar: bytes) -> tuple:
    """Compute scalar * B using Python (slower but correct)."""
    k = int.from_bytes(scalar, byteorder="little")
    # Ed25519 base point
    Bx = 15112221349535400772501151409588531511454012693041857206046113283949847762202
    By = 46316835694926478169428394003475163141307993866256225615783033603165251855960
    P = 2**255 - 19
    D = 37095705934669439343138083508754565189542113879843219016388785533085940283555

    def add(x1, y1, x2, y2):
        x1x2 = (x1 * x2) % P
        y1y2 = (y1 * y2) % P
        x1y2 = (x1 * y2) % P
        x2y1 = (x2 * y1) % P
        dterm = (D * x1x2 * y1y2) % P
        num_x = (x1y2 + x2y1) % P
        den_x = (1 + dterm) % P
        x3 = (num_x * pow(den_x, P - 2, P)) % P
        num_y = (y1y2 + x1x2) % P
        den_y = (1 - dterm) % P
        y3 = (num_y * pow(den_y, P - 2, P)) % P
        return x3, y3

    # Double-and-add
    rx, ry = 0, 1  # identity
    for i in range(255, -1, -1):
        rx, ry = add(rx, ry, rx, ry)  # double
        bit = (k >> i) & 1
        if bit:
            rx, ry = add(rx, ry, Bx, By)  # add base
    return rx, ry

"""CPU reference: build 32-byte public key from affine Y and X sign bit."""
def ref_point_to_pub(affine_y: bytes, x_sign: int) -> bytes:
    """Encode 32-byte LE Y + X sign bit as public key."""
    pub = bytearray(affine_y)
    pub[31] |= (x_sign << 7)
    return bytes(pub)

"""CPU reference: check if point is on Ed25519 curve."""
def ref_point_on_curve(x, y) -> bool:
    """Check if (x,y) is on Ed25519 curve."""
    P = 2**255 - 19
    D = 37095705934669439343138083508754565189542113879843219016388785533085940283555
    lhs = (-x*x + y*y) % P
    rhs = (1 + D*x*x*y*y) % P
    return lhs == rhs

# --- Standard test suites --------------------------------

"""Test GPU base64_encode against Python reference for random data."""
def test_base64(kc):
    """Standard base64 test suite."""
    print("Testing base64_encode...")
    ok = True
    test_cases = [
        ("empty", b""),
        ("single", b"A"),
        ("triple", b"ABC"),
        ("four", b"ABCD"),
        ("hello", b"Hello World!"),
        ("binary", bytes(range(256))),
        ("full_32", bytes([0xDE, 0xAD, 0xBE, 0xEF] * 8)),
    ]
    for name, data in test_cases:
        ref_out = ref_base64_encode(data)
        gpu_out = run_base64(kc, data)
        null_pos = gpu_out.find(b'\x00')
        if null_pos >= 0:
            gpu_out = gpu_out[:null_pos]
        if gpu_out == ref_out:
            print("  PASS: %s (%d bytes -> %s)" % (name, len(data), ref_out[:30].decode('ascii', errors='replace')))
        else:
            print("  FAIL: %s" % name)
            print("    Ref: %s" % ref_out[:40])
            print("    GPU: %s" % gpu_out[:40])
            ok = False
    return ok

"""Test GPU build_ssh_public_blob against CPU reference."""
def test_build_blob(kc):
    """Standard build_ssh_public_blob test suite."""
    print("Testing build_ssh_public_blob...")
    ok = True
    test_cases = [
        ("zeros", bytes(32)),
        ("ones", bytes([0xFF] * 32)),
        ("base_y", bytes([0x58] + [0] * 31)),
    ]
    for name, pub_key in test_cases:
        ref_blob = ref_ssh_public_blob(pub_key)
        gpu_blob = run_build_blob(kc, pub_key)
        if gpu_blob == ref_blob:
            print("  PASS: %s" % name)
        else:
            print("  FAIL: %s" % name)
            print("    Ref: %s" % ref_blob[:30].hex())
            print("    GPU: %s" % gpu_blob[:30].hex())
            ok = False
    return ok

"""Test GPU seed_to_ssh_ed25519_pubkey against CPU cryptography reference."""
def test_openkey(kc):
    """Open key test suite -- GPU seed->pubkey vs cryptography.hazmat reference.

    seeds_comments: test seeds with their expected comments.
    Reference: seed-2-openssh-key.py on the server produces the same public key.
    """
    print("Testing seed_to_ssh_ed25519_pubkey (STATUS.md seeds)...")
    ok = True
    seeds_comments = [
        (bytes.fromhex("5a72ad0ce00b6619c22cf504ffd6984811b8805759a4bc6701351d728ff5898d"), b"User"),
        (bytes.fromhex("9f699a703e466ad9e1018d82c7e7d7819730d761a26ea034d3f9d9cff40bd7c3"), b"User"),
        (bytes.fromhex("07cef4ba976cd9c1271dce7cb800a6e0e4902dc86dfc80a63161167999ca4a3b"), b"User"),
        (bytes.fromhex("6336a500a3c6c08ff785855bd79128d8d0049359780f256df6886267a942b926"), b"Amin"),
        (bytes.fromhex("de22a26b9f565597b881b4430506fdd3fd187f07781e740d4691cacda9c37488"), b"Amin"),
    ]
    for seed, comment in seeds_comments:
        if CRYPTO_AVAILABLE:
            key = Ed25519PrivateKey.from_private_bytes(seed)
            pub_key = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            ref_line = ref_ssh_public_line(pub_key, comment)
            gpu_pubkey, gpu_line = run_openkey(kc, seed, comment)
            null_pos = gpu_line.find(b'\x00')
            if null_pos > 0:
                gpu_line = gpu_line[:null_pos]
            if gpu_pubkey == pub_key:
                print("  PASS: pubkey seed=%s comment=%s" % (seed.hex()[:8], comment.decode()))
            else:
                print("  FAIL: pubkey seed=%s" % seed.hex()[:8])
                print("    Ref: %s" % pub_key.hex())
                print("    GPU: %s" % gpu_pubkey.hex())
                ok = False
            if gpu_line == ref_line:
                print("  PASS: publine seed=%s comment=%s" % (seed.hex()[:8], comment.decode()))
            else:
                print("  FAIL: publine seed=%s" % seed.hex()[:8])
                print("    Ref: %s" % ref_line.decode('ascii', errors='replace')[:80])
                print("    GPU: %s" % gpu_line.decode('ascii', errors='replace')[:80])
                ok = False
        else:
            print("  SKIP seed %s: cryptography not available" % seed.hex()[:8])
    return ok

"""Test full pipeline: SHA512 -> clamp -> scalar_mult -> affine -> pubkey."""
def test_pipeline(kc):
    """Detailed step-by-step pipeline debug for seed->pubkey.

    Tests each stage of the Ed25519 key generation pipeline:
    1. SHA512(seed)
    2. Scalar clamping
    3. Scalar multiplication (GPU vs Python)
    4. Point to affine Y conversion
    5. Public key encoding

    Uses first seed from STATUS.md for detailed debugging.
    """
    print("Testing pipeline step-by-step (seed 5a72ad0c)...")
    ok = True
    seed = bytes.fromhex("5a72ad0ce00b6619c22cf504ffd6984811b8805759a4bc6701351d728ff5898d")

    # --- Step 1: SHA512 ---
    print("\n[Step 1] SHA512(seed)")
    ref_hash = hashlib.sha512(seed).digest()
    gpu_hash = run_sha512_32(kc, seed)
    if ref_hash == gpu_hash:
        print("  PASS: SHA512 hash matches")
    else:
        print("  FAIL: SHA512 hash mismatch")
        print("    Ref: %s" % ref_hash.hex())
        print("    GPU: %s" % gpu_hash.hex())
        ok = False

    # --- Step 2: Scalar clamping ---
    print("\n[Step 2] Scalar clamping (RFC 8032)")
    ref_scalar = ref_clamp_scalar(ref_hash)
    # GPU clamping is in clamp_and_encode, so we test it indirectly
    print("  Ref scalar (LE): %s" % ref_scalar.hex())
    print("  expanded[0]=0x%02x -> clamped 0x%02x (0xF8 mask)" % (ref_hash[0], ref_hash[0] & 0xF8))
    print("  expanded[31]=0x%02x -> clamped 0x%02x (0x7F|0x40)" % (ref_hash[31], (ref_hash[31] & 0x7F) | 0x40))

    # --- Step 3: Scalar multiplication ---
    print("\n[Step 3] Scalar multiplication")
    gpu_point = run_scalar_mult(kc, ref_scalar)
    gpu_X = int.from_bytes(gpu_point[0:32], "little")
    gpu_Y = int.from_bytes(gpu_point[32:64], "little")
    gpu_Z = int.from_bytes(gpu_point[64:96], "little")
    print("  GPU projective point:")
    print("    X: %s" % gpu_point[0:32].hex()[:40] + "...")
    print("    Y: %s" % gpu_point[32:64].hex()[:40] + "...")
    print("    Z: %s" % gpu_point[64:96].hex()[:40] + "...")

    # Compute affine from projective
    P = 2**255 - 19
    D = 37095705934669439343138083508754565189542113879843219016388785533085940283555
    Z_inv = pow(gpu_Z, -1, P)
    affine_X = (gpu_X * Z_inv) % P
    affine_Y = (gpu_Y * Z_inv) % P
    print("  GPU affine point (computed from projective):")
    print("    X: %s..." % hex(affine_X))
    print("    Y: %s..." % hex(affine_Y))

    # Check if point is on curve
    if ref_point_on_curve(affine_X, affine_Y):
        print("  GPU point is on Ed25519 curve: YES")
    else:
        print("  GPU point is on Ed25519 curve: NO!")
        ok = False

    # --- Compare GPU scalar_mult with cryptography reference ---
    print("\n[Compare] GPU scalar_mult vs cryptography reference")
    # Test with small scalar k=2
    scalar_2 = bytes([2] + [0]*31)
    print("  Testing with k=2 (small scalar for exact match)")
    ref_2x, ref_2y = ref_scalar_mult(scalar_2)
    print("    Ref (X,Y): %s... %s..." % (hex(ref_2x), hex(ref_2y)))
    gpu_2pt = run_scalar_mult(kc, scalar_2)
    gpu_2X = int.from_bytes(gpu_2pt[0:32], "little")
    gpu_2Y = int.from_bytes(gpu_2pt[32:64], "little")
    gpu_2Z = int.from_bytes(gpu_2pt[64:96], "little")
    gpu_2Z_inv = pow(gpu_2Z, -1, P)
    gpu_2afX = (gpu_2X * gpu_2Z_inv) % P
    gpu_2afY = (gpu_2Y * gpu_2Z_inv) % P
    print("    GPU (X,Y): %s... %s..." % (hex(gpu_2afX), hex(gpu_2afY)))
    if gpu_2afX == ref_2x and gpu_2afY == ref_2y:
        print("    PASS: k=2 matches")
    else:
        print("    FAIL: k=2 mismatch - scalar_mult has bug!")
        ok = False

    # Compute reference scalar*B (using cryptography)
    if CRYPTO_AVAILABLE:
        key = Ed25519PrivateKey.from_private_bytes(seed)
        pub_key = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        # Recover affine point from public key
        sign = (pub_key[31] >> 7) & 1
        y_clean = bytearray(pub_key)
        y_clean[31] &= 0x7F
        ref_Y = int.from_bytes(bytes(y_clean), "little")

        # Recover X from curve equation
        y2 = pow(ref_Y, 2, P)
        num = (y2 - 1) % P
        den = (D * y2 + 1) % P
        x2 = (num * pow(den, P - 2, P)) % P
        ref_X = pow(x2, (P + 3) // 8, P)
        if (ref_X * ref_X) % P != x2:
            ref_X = (P - ref_X) % P
        if (ref_X & 1) != sign:
            ref_X = (P - ref_X) % P

        print("\n  Reference point (from cryptography):")
        print("    X: %s..." % hex(ref_X))
        print("    Y: %s..." % hex(ref_Y))

        if affine_X == ref_X and affine_Y == ref_Y:
            print("  PASS: GPU point matches reference")
        else:
            print("  FAIL: GPU point does not match reference")
            print("    X diff: %s" % hex(affine_X ^ ref_X)[:40])
            print("    Y diff: %s" % hex(affine_Y ^ ref_Y)[:40])
            ok = False

    # --- Step 4: Point to affine Y ---
    print("\n[Step 4] Point to affine Y conversion")
    gpu_Y_affine = run_point_to_affine_y(kc, gpu_point)
    print("  GPU affine Y: %s" % gpu_Y_affine.hex())
    ref_Y_bytes = ref_Y.to_bytes(32, "little")
    if gpu_Y_affine == ref_Y_bytes:
        print("  PASS: GPU affine Y matches reference")
    else:
        print("  FAIL: GPU affine Y mismatch")
        print("    Ref: %s" % ref_Y_bytes.hex())
        ok = False

    # --- Step 5: Public key encoding ---
    print("\n[Step 5] Public key encoding (Y + X sign bit)")
    gpu_pub = run_clamp_and_encode(kc, ref_scalar)
    print("  GPU pubkey: %s" % gpu_pub.hex())
    print("  Ref pubkey: %s" % pub_key.hex())

    # Check X sign bit
    gpu_X_sign = (gpu_pub[31] >> 7) & 1
    ref_X_sign = (pub_key[31] >> 7) & 1
    print("  GPU X sign: %d" % gpu_X_sign)
    print("  Ref X sign: %d" % ref_X_sign)

    if gpu_pub == pub_key:
        print("  PASS: GPU pubkey matches reference")
    else:
        print("  FAIL: GPU pubkey mismatch")
        ok = False

    return ok

"""Test debug_pipeline kernel with intermediate value outputs."""
def test_debug(kc):
    """Debug: compare intermediate values between seed_to_ssh_ed25519_pubkey and step-by-step."""
    print("Testing debug pipeline (intermediate values)...")
    ok = True
    seed = bytes.fromhex("5a72ad0ce00b6619c22cf504ffd6984811b8805759a4bc6701351d728ff5898d")

    # --- Step 1: SHA512 ---
    print("\n[Step 1] SHA512")
    ref_hash = hashlib.sha512(seed).digest()
    gpu_hash = run_sha512_32(kc, seed)
    print("  Ref hash: %s" % ref_hash.hex())
    print("  GPU hash: %s" % gpu_hash.hex())
    if ref_hash == gpu_hash:
        print("  PASS: SHA512 matches")
    else:
        print("  FAIL: SHA512 mismatch!")
        ok = False

    # --- Step 2: Scalar derivation ---
    print("\n[Step 2] Scalar derivation")
    ref_scalar = ref_clamp_scalar(ref_hash)
    print("  Ref scalar: %s" % ref_scalar.hex())

    # --- Step 3: debug_pipeline kernel ---
    print("\n[Step 3] debug_pipeline (full GPU path)")
    dev = get_device()
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    bi_seed = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=seed)
    bo_hash = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 64)
    bo_scalar = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 32)
    bo_pub = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 32)
    prg.debug_pipeline(queue, (1,), None, bi_seed, bo_hash, bo_scalar, bo_pub)
    queue.finish()
    d_hash = bytearray(64)
    d_scalar = bytearray(32)
    d_pub = bytearray(32)
    cl.enqueue_copy(queue, d_hash, bo_hash, is_blocking=True)
    cl.enqueue_copy(queue, d_scalar, bo_scalar, is_blocking=True)
    cl.enqueue_copy(queue, d_pub, bo_pub, is_blocking=True)
    d_hash = bytes(d_hash)
    d_scalar = bytes(d_scalar)
    d_pub = bytes(d_pub)

    print("  GPU hash:   %s" % d_hash.hex())
    print("  GPU scalar: %s" % d_scalar.hex())
    print("  GPU pubkey: %s" % d_pub.hex())

    # --- Step 4: Run seed_to_ssh_ed25519_pubkey ---
    print("\n[Step 4] seed_to_ssh_ed25519_pubkey (original kernel)")
    gpu_pubkey, gpu_line = run_openkey(kc, seed, b"")
    print("  GPU pubkey: %s" % gpu_pubkey.hex())
    print("  GPU line:   %s" % gpu_line.decode('ascii', errors='replace')[:80])

    # Compare
    print("\n[Compare]")
    if d_hash == ref_hash:
        print("  PASS: debug_pipeline hash == ref hash")
    else:
        print("  FAIL: debug_pipeline hash != ref hash")
        ok = False

    if d_scalar == ref_scalar:
        print("  PASS: debug_pipeline scalar == ref scalar")
    else:
        print("  FAIL: debug_pipeline scalar != ref scalar")
        print("    diff bytes:", [i for i in range(32) if d_scalar[i] != ref_scalar[i]])
        ok = False

    if d_pub == gpu_pubkey:
        print("  PASS: debug_pipeline pubkey == seed_to_ssh_ed25519_pubkey pubkey")
    else:
        print("  FAIL: debug_pipeline pubkey != seed_to_ssh_ed25519_pubkey pubkey!")
        print("    debug:   %s" % d_pub.hex())
        print("    openkey: %s" % gpu_pubkey.hex())
        ok = False

    if CRYPTO_AVAILABLE:
        key = Ed25519PrivateKey.from_private_bytes(seed)
        pub_key = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        if d_pub == pub_key:
            print("  PASS: debug_pipeline pubkey == cryptography pubkey")
        else:
            print("  FAIL: debug_pipeline pubkey != cryptography pubkey")
            print("    ref: %s" % pub_key.hex())
            ok = False

    # --- Compare scalar_mult with Python reference ---
    print("\n[Compare scalar_mult]")
    gpu_point = run_scalar_mult(kc, ref_scalar)
    gpu_X = int.from_bytes(gpu_point[0:32], "little")
    gpu_Y = int.from_bytes(gpu_point[32:64], "little")
    gpu_Z = int.from_bytes(gpu_point[64:96], "little")
    P = 2**255 - 19
    D = 37095705934669439343138083508754565189542113879843219016388785533085940283555
    Z_inv = pow(gpu_Z, -1, P)
    gpu_afX = (gpu_X * Z_inv) % P
    gpu_afY = (gpu_Y * Z_inv) % P
    print("  GPU affine: X=%s... Y=%s..." % (hex(gpu_afX)[:20], hex(gpu_afY)[:20]))

    # Recover from cryptography
    sign = (pub_key[31] >> 7) & 1
    y_clean = bytearray(pub_key)
    y_clean[31] &= 0x7F
    ref_Y = int.from_bytes(bytes(y_clean), "little")
    y2 = pow(ref_Y, 2, P)
    num = (y2 - 1) % P
    den = (D * y2 + 1) % P
    x2 = (num * pow(den, P - 2, P)) % P
    ref_X = pow(x2, (P + 3) // 8, P)
    if (ref_X * ref_X) % P != x2: ref_X = (P - ref_X) % P
    if (ref_X & 1) != sign: ref_X = (P - ref_X) % P
    print("  Ref affine: X=%s... Y=%s..." % (hex(ref_X)[:20], hex(ref_Y)[:20]))

    if gpu_afX == ref_X and gpu_afY == ref_Y:
        print("  PASS: scalar_mult matches cryptography")
    else:
        print("  FAIL: scalar_mult mismatch!")
        ok = False

    return ok

# --- Custom tests (--data / --hex-data) ------------------

"""Run custom base64 test with user-supplied data."""
def custom_base64(kc, data: bytes):
    """Custom base64 test with arbitrary data."""
    ref_b64 = ref_base64_encode(data)
    gpu_b64 = run_base64(kc, data)
    null_pos = gpu_b64.find(b'\x00')
    if null_pos >= 0:
        gpu_b64 = gpu_b64[:null_pos]
    print("Input (%d bytes): %s" % (len(data), data.hex()))
    print("  REF: %s" % ref_b64.decode('ascii', errors='replace'))
    print("  GPU: %s" % gpu_b64.decode('ascii', errors='replace'))
    match = "OK" if gpu_b64 == ref_b64 else "DIFF"
    print("  Match: %s" % match)
    print()
    return match == "OK"

"""Run custom openkey test with user-supplied seed."""
def custom_openkey(kc, data: bytes):
    """Custom openkey test. data = 32-byte seed.

    GPU generates: "ssh-ed25519 <base64-blob>"
    Reference: cryptography.hazmat produces the same base64 blob.
    """
    if len(data) != 32:
        print("Error: openkey requires exactly 32 bytes (use --hex-data with 64 hex chars)")
        sys.exit(1)
    seed = data
    comment = b""
    print("Seed: %s" % seed.hex())
    print()

    ok = True
    if not CRYPTO_AVAILABLE:
        print("WARN: cryptography not available -- GPU only mode")
        gpu_pubkey, gpu_line = run_openkey(kc, seed, comment)
        null_pos = gpu_line.find(b'\x00')
        if null_pos > 0:
            gpu_line = gpu_line[:null_pos]
        print("GPU pubkey: %s" % gpu_pubkey.hex())
        print("GPU pubLine: %s" % gpu_line.decode('ascii', errors='replace'))
        return True

    # Reference via cryptography.hazmat (same as seed-2-openssh-key.py)
    key = Ed25519PrivateKey.from_private_bytes(seed)
    pub_key = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    ref_line = ref_ssh_public_line(pub_key, comment)
    ref_blob = ref_ssh_public_blob(pub_key)
    ref_b64 = ref_base64_encode(ref_blob)

    # GPU
    gpu_pubkey, gpu_line = run_openkey(kc, seed, comment)
    null_pos = gpu_line.find(b'\x00')
    if null_pos >= 0:
        gpu_line = gpu_line[:null_pos]

    print("PUBLIC KEY (32 bytes raw)")
    print("  REF: %s" % pub_key.hex())
    print("  GPU: %s" % gpu_pubkey.hex())
    m = "OK" if gpu_pubkey == pub_key else "DIFF"
    print("  Match: %s" % m)
    if m == "DIFF": ok = False
    print()

    print("BASE64 (SSH public blob -- generated INSIDE GPU kernel)")
    print("  REF: %s" % ref_b64.decode())
    gpu_b64_raw = gpu_line[len(b"ssh-ed25519 "):]
    if comment:
        # strip " <comment>" from end
        comment_end = b" " + comment
        gpu_b64_raw = gpu_line[len(b"ssh-ed25519 "):-len(comment_end)]
    print("  GPU: %s" % gpu_b64_raw.decode('ascii', errors='replace'))
    m = "OK" if gpu_b64_raw == ref_b64 else "DIFF"
    print("  Match: %s" % m)
    if m == "DIFF": ok = False
    print()

    print("SSH PUBLIC KEY LINE (full)")
    print("  REF: %s" % ref_line.decode('ascii', errors='replace'))
    print("  GPU: %s" % gpu_line.decode('ascii', errors='replace'))
    m = "OK" if gpu_line == ref_line else "DIFF"
    print("  Match: %s" % m)
    if m == "DIFF": ok = False
    print()
    return ok

"""Run custom blob test with user-supplied pubkey."""
def custom_blob(kc, data: bytes):
    """Custom blob test. data = 32-byte raw pubkey."""
    if len(data) != 32:
        print("Error: blob requires exactly 32 bytes (raw pubkey, use --hex-data with 64 hex chars)")
        sys.exit(1)
    pub_key = data
    ref_blob = ref_ssh_public_blob(pub_key)
    gpu_blob = run_build_blob(kc, pub_key)
    print("PubKey: %s" % pub_key.hex())
    print("  Ref: %s" % ref_blob.hex())
    print("  GPU: %s" % gpu_blob.hex())
    match = "OK" if gpu_blob == ref_blob else "DIFF"
    print("  Match: %s" % match)
    print()
    return match == "OK"

# --- Main ------------------------------------------------

"""Parse user input: hex string or file path to bytes."""
def resolve_data(text):
    """Resolve --data input:
    - 64 hex chars (no 0x) -> seed bytes (32 bytes)
    - starts with 0x + hex chars -> hex decode
    - otherwise -> UTF-8 bytes
    """
    if text is None:
        return None
    # 64 hex chars -> seed
    if len(text) == 64 and all(c in '0123456789abcdefABCDEF' for c in text):
        return bytes.fromhex(text)
    # 0x prefix
    if text.startswith('0x') or text.startswith('0X'):
        hex_part = text[2:]
        if len(hex_part) > 0 and all(c in '0123456789abcdefABCDEF' for c in hex_part):
            return bytes.fromhex(hex_part)
    # UTF-8 text
    return text.encode('utf-8')

"""Entry point: parse args, run selected tests, report results."""
def main():
    pa = argparse.ArgumentParser(
        description="OpenSSH Ed25519 Public Key OpenCL Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""--data format:
  64 hex chars  -> seed (e.g. 5a72ad0ce00b6619c22cf504ffd6984811b8805759a4bc6701351d728ff5898d)
  0x prefix     -> hex dump (e.g. 0x48656c6c6f)
  otherwise     -> UTF-8 text (e.g. "hello")
Examples:
  python3 openssh.cl.test.py --func base64 --data "my-test-data"
  python3 openssh.cl.test.py --func openkey --data 5a72ad0ce00b6619c22cf504ffd6984811b8805759a4bc6701351d728ff5898d
  python3 openssh.cl.test.py --func blob --data 0x5cad1b490a3090a1bfded2c809c952c7abbbbdc3b3776fbca33d343b8e15b5ff
""")
    pa.add_argument("--func", choices=["all", "base64", "blob", "openkey", "pipeline", "debug"], default="all",
                    help="Function to test")
    pa.add_argument("--data", type=str, default=None,
                    help="Input data (64 hex chars = seed, 0x... = hex dump, else UTF-8 text)")
    a = pa.parse_args()

    print("=" * 70)
    print("OpenSSH Ed25519 Public Key OpenCL Test Suite")
    print("=" * 70)
    dev = get_device()
    print("Device: %s" % dev.name)
    if not CRYPTO_AVAILABLE:
        print("WARN: cryptography not installed -- limited tests")

    kc = open("./openssh_test_kernels.cl").read()
    print("Kernel: ./openssh_test_kernels.cl\n")

    input_data = resolve_data(a.data)

    # Custom test with data
    if input_data is not None:
        if a.func == "base64":
            ok = custom_base64(kc, input_data)
        elif a.func == "blob":
            ok = custom_blob(kc, input_data)
        elif a.func == "openkey":
            ok = custom_openkey(kc, input_data)
        elif a.func == "all":
            ok = custom_openkey(kc, input_data)
        else:
            ok = True
        print("=" * 70)
        print("RESULT: PASSED" if ok else "RESULT: FAILED")
        print("=" * 70)
        return 0 if ok else 1

    # Standard test suites (no data)
    ok = True
    if a.func == "base64":
        if not test_base64(kc): ok = False
    elif a.func == "blob":
        if not test_build_blob(kc): ok = False
    elif a.func == "openkey":
        if not test_openkey(kc): ok = False
    elif a.func == "pipeline":
        if not test_pipeline(kc): ok = False
    elif a.func == "debug":
        if not test_debug(kc): ok = False
    elif a.func == "all":
        if not test_base64(kc): ok = False
        print()
        if not test_build_blob(kc): ok = False
        print()
        if not test_openkey(kc): ok = False
        print()
        if not test_pipeline(kc): ok = False
        print()

    print("=" * 70)
    print("RESULT: ALL PASSED" if ok else "RESULT: SOME FAILED")
    print("=" * 70)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
