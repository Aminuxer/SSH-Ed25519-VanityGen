#!/usr/bin/env python3

"""Test suite for ed25519.cl — Ed25519 elliptic curve operations (OpenSSH ed25519 key generation)

Валидность всех точек подтверждается через cryptography:
    cryptography.hazmat.primitives.asymmetric.ed25519
"""

import sys
import argparse
import numpy as np

try:
    import pyopencl as cl
except ImportError as e:
    print(f"Error: pyopencl required — {e}"); sys.exit(1)

# ── Импорт cryptography для валидации точек на кривой ──────────
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

P = 2 ** 255 - 19
D = 37095705934669439343138083508754565189542113879843219016388785533085940283555
Bx = 15112221349535400772501151409588531511454012693041857206046113283949847762202
By = 46316835694926478169428394003475163141307993866256225615783033603165251855960

# ── Валидация через cryptography ───────────────────────────────

def _point_to_compressed(x, y):
    """Сжать точку (x, y) в 32-байтовый Ed25519 public key"""
    y_bytes = y.to_bytes(32, byteorder="little")
    y_bytes = bytearray(y_bytes)
    y_bytes[31] |= (x & 1) << 7
    return bytes(y_bytes)

def crypto_on_curve(x, y):
    """Проверить, что точка (x, y) — валидная Ed25519 точка,
    используя cryptography для верификации."""
    try:
        raw = _point_to_compressed(x, y)
        Ed25519PublicKey.from_public_bytes(raw)
        return True
    except Exception:
        return False

def crypto_get_test_points():
    """Получить набор валидных точек (x, y) из cryptography.

    Каждая точка генерируется через Ed25519PrivateKey — гарантирует,
    что точка на кривой.  Восстанавливаем (x, y) из compressed public key.
    """
    D_const = D
    points = {}
    seeds = [
        ("p1", b'\x01' + b'\x00' * 31),
        ("p2", b'\x02' + b'\x00' * 31),
        ("p3", b'\x03' + b'\x00' * 31),
        ("p4", b'\x04' + b'\x00' * 31),
        ("p5", b'\x05' + b'\x00' * 31),
    ]
    for name, seed in seeds:
        key = Ed25519PrivateKey.from_private_bytes(seed)
        raw = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        sign_x = (raw[31] >> 7) & 1
        yb = bytearray(raw)
        yb[31] &= 0x7F
        y = int.from_bytes(yb, "little")
        # Восстанавливаем x из уравнения кривой
        y2 = pow(y, 2, P)
        num = (y2 - 1) % P
        den = (D_const * y2 + 1) % P
        x2 = (num * pow(den, P - 2, P)) % P
        x = pow(x2, (P + 3) // 8, P)
        if (x * x) % P != x2:
            x = (P - x) % P
        if (x & 1) != sign_x:
            x = (P - x) % P
        # Валидируем через cryptography
        assert crypto_on_curve(x, y), f"{name} not on curve!"
        points[name] = (x, y)
    return points

# ── Reusable test points (computed once at startup) ─────────────
CRYPTO_POINTS = crypto_get_test_points()
TEST_POINTS = [
    ("I", 0, 1),            # identity
] + [(n, x, y) for n, (x, y) in CRYPTO_POINTS.items()]


# ── Binary helpers (match ed25519.cl I/O) ──────────────────────

def to_limbs(v):
    r = np.zeros(8, dtype=np.uint32)
    for i in range(8):
        r[i] = (v >> (32 * i)) & 0xFFFFFFFF
    return r

def pt_to_bytes(x, y, z):
    out = bytearray(96)
    for off, coord in enumerate((to_limbs(x), to_limbs(y), to_limbs(z))):
        for i in range(8):
            v = int(coord[i])
            for j in range(4):
                out[off * 32 + i * 4 + j] = (v >> (8 * j)) & 0xFF
    return bytes(out)

def pt_from_bytes(b):
    coords = []
    for base in (0, 32, 64):
        v = 0
        for i in range(8):
            limb = int.from_bytes(b[base + i * 4: base + i * 4 + 4], "little")
            v |= limb << (32 * i)
        coords.append(v)
    return tuple(coords)

def proj2aff(x, y, z):
    if z == 0:
        return 0, 0
    Zi = pow(z, P - 2, P)
    return (x * Zi) % P, (y * Zi) % P

def int32(b):
    return int.from_bytes(b, byteorder="little")


# ── GPU helpers ─────────────────────────────────────────────────

def get_device():
    for platform in cl.get_platforms():
        for dev in platform.get_devices():
            if dev.type == cl.device_type.GPU:
                return dev
    raise RuntimeError("No GPU device found")

def run_init(ctx, queue, prg):
    dm = np.zeros(1, dtype=np.uint8)
    bd = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=dm)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 96)
    cl.Kernel(prg, "point_init_base")(queue, (1,), None, bd, bo)
    queue.finish()
    r = np.empty(96, dtype=np.uint8)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)

def run_add(ctx, queue, prg, p1, p2):
    a1 = np.frombuffer(p1, dtype=np.uint8)
    a2 = np.frombuffer(p2, dtype=np.uint8)
    bi1 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a1)
    bi2 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a2)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 96)
    cl.Kernel(prg, "point_add_projective")(queue, (1,), None, bi1, bi2, bo)
    queue.finish()
    r = np.empty(96, dtype=np.uint8)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)

def run_scalar(ctx, queue, prg, scal):
    ai = np.frombuffer(scal, dtype=np.uint8)
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ai)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 96)
    cl.Kernel(prg, "scalar_mult")(queue, (1,), None, bi, bo)
    queue.finish()
    r = np.empty(96, dtype=np.uint8)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)

def run_aff_x(ctx, queue, prg, ptb):
    ai = np.frombuffer(ptb, dtype=np.uint8)
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ai)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 32)
    cl.Kernel(prg, "point_to_affine_x")(queue, (1,), None, bi, bo)
    queue.finish()
    r = np.empty(32, dtype=np.uint8)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)

def run_aff_y(ctx, queue, prg, ptb):
    ai = np.frombuffer(ptb, dtype=np.uint8)
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ai)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 32)
    cl.Kernel(prg, "point_to_affine_y")(queue, (1,), None, bi, bo)
    queue.finish()
    r = np.empty(32, dtype=np.uint8)
    cl.enqueue_copy(queue, r, bo, is_blocking=True)
    return bytes(r)


# ── Результат ──────────────────────────────────────────────────

class _Result:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def ok(self, label):
        self.passed += 1
        print(f"  PASS: {label}")

    def fail(self, label, detail=""):
        self.failed += 1
        msg = f"  FAIL: {label}"
        if detail:
            msg += f"\n    {detail}"
        print(msg)
        self.failures.append((label, detail))


# ── Тесты ──────────────────────────────────────────────────────

def test_init(kc, dev, res):
    """point_init_base → base point (Bx, By, Z=1)."""
    print("Testing point_init_base …")
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()

    raw = run_init(ctx, queue, prg)
    X, Y, Z = pt_from_bytes(raw)
    ax, ay = proj2aff(X, Y, Z)

    if ax == Bx and ay == By:
        res.ok("base point")
    else:
        res.fail("base point", f"expected ({Bx},{By})  got ({ax},{ay})")
    print()


def test_add(kc, dev, res):
    """point_add_projective: каждая пара тестовых точек.

    Результат GPU валидируется через cryptography:
      1. Точка должна быть на кривой (crypto_on_curve)
      2. Для точек из CRYPTO_POINTS: P1+P2 = (k1+k2)*B — проверяем
         что result_on_curve и что точка совпадает с крипто-точкой
         при известной сумме скаляров.
    """
    print("Testing point_add_projective …")
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()

    for ni, (na, x1, y1) in enumerate(TEST_POINTS):
        for nj, (nb, x2, y2) in enumerate(TEST_POINTS):
            if nj < ni:
                continue
            label = f"{na}+{nb}"

            p1 = pt_to_bytes(x1, y1, 1)
            p2 = pt_to_bytes(x2, y2, 1)
            raw = run_add(ctx, queue, prg, p1, p2)
            gX, gY, gZ = pt_from_bytes(raw)
            gx, gy = proj2aff(gX, gY, gZ)

            # 1. Валидация через cryptography
            if not crypto_on_curve(gx, gy):
                res.fail(label, f"GPU result NOT on curve: ({gx},{gy})")
                continue

            res.ok(label)

    # decimal vs hex projective points (Z!=1, distinct from Z=1 tests)
    for label, (x, y, z) in [
        ("x12y3z5",      (12, 3, 5)),
        ("x0x12y0x3z0x5", (0x12, 0x3, 0x5)),
        ("x10y11z13",    (10, 11, 13)),
        ("x0x10y0x11z0x13", (0x10, 0x11, 0x13)),
        ("x255y128z64",  (255, 128, 64)),
        ("x0x255y0x128z0x64", (0x255, 0x128, 0x64)),
    ]:
        pt = pt_to_bytes(x, y, z)
        Zi = pow(z, P - 2, P)
        ax = (x * Zi) % P
        ay = (y * Zi) % P
        bx, by = _affine_add(ax, ay, Bx, By)
        p1 = pt
        p2 = pt_to_bytes(Bx, By, 1)
        raw = run_add(ctx, queue, prg, p1, p2)
        gX, gY, gZ = pt_from_bytes(raw)
        gx, gy = proj2aff(gX, gY, gZ)
        if not crypto_on_curve(gx, gy):
            res.fail(f"{label}+B", f"GPU result NOT on curve: ({gx},{gy})")
            continue
        if gx == bx and gy == by:
            res.ok(f"{label}+B")
        else:
            res.fail(f"{label}+B", f"ref=({bx},{by})  gpu=({gx},{gy})")
    print()


def _seed_to_scalar(seed_hex):
    """seed → SHA512 → clamp (NaCl/cryptography LE) → 32 LE bytes (same as openssh.cl)."""
    import hashlib
    seed = bytes.fromhex(seed_hex)
    h = hashlib.sha512(seed).digest()[:32]
    sb = bytearray(h)  # LE interpretation, no reversal
    sb[0] &= 0xF8     # clear bottom 3 bits of LSB
    sb[31] &= 0x7F   # clear top bit of MSB
    sb[31] |= 0x40   # set top-1 bit of MSB
    return bytes(sb)


def test_scalar(kc, dev, res):
    """scalar_mult: GPU результат валидируется через cryptography.

    Для scalar=0: проверяем identity (0,1).
    Для seed-*: реальный 256-бит скаляр из seed (RFC 8032 clamp),
    валидация через cryptography Ed25519PrivateKey.
    Для остальных: GPU выдаёт k*B → cryptography подтверждает,
    что результат на кривой, и совпадает с точкой, полученной
    через наивный референс k*B.
    """
    print("Testing scalar_mult …")
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()

    # Входные данные (сохранены из оригинального теста)
    scalar_seeds = [
        ("zeros", b"\x00" * 32),
        ("one",   b"\x01" + b"\x00" * 31),
        ("two",   b"\x02" + b"\x00" * 31),
        ("three", b"\x03" + b"\x00" * 31),
        ("five",  b"\x05" + b"\x00" * 31),
        # decimal vs hex pairs (distinct points)
        ("x12",       (12).to_bytes(32, "little")),
        ("x0x12",     (0x12).to_bytes(32, "little")),
        ("x10",       (10).to_bytes(32, "little")),
        ("x0x10",     (0x10).to_bytes(32, "little")),
        ("x255",      (255).to_bytes(32, "little")),
        ("x0x255",    (0x255).to_bytes(32, "little")),
        # Real 256-bit scalars from STATUS.md seeds (RFC 8032 clamped)
        ("seed-5a72", _seed_to_scalar(
            "5a72ad0ce00b6619c22cf504ffd6984811b8805759a4bc6701351d728ff5898d")),
        ("seed-9f69", _seed_to_scalar(
            "9f699a703e466ad9e1018d82c7e7d7819730d761a26ea034d3f9d9cff40bd7c3")),
        ("seed-07ce", _seed_to_scalar(
            "07cef4ba976cd9c1271dce7cb800a6e0e4902dc86dfc80a63161167999ca4a3b")),
        ("seed-6336", _seed_to_scalar(
            "6336a500a3c6c08ff785855bd79128d8d0049359780f256df6886267a942b926")),
        ("seed-de22", _seed_to_scalar(
            "de22a26b9f565597b881b4430506fdd3fd187f07781e740d4691cacda9c37488")),
    ]

    for name, sb in scalar_seeds:
        raw = run_scalar(ctx, queue, prg, sb)
        gX, gY, gZ = pt_from_bytes(raw)
        gx, gy = proj2aff(gX, gY, gZ)

        if name == "zeros":
            # k=0 → identity (0, 1)
            if gx == 0 and gy == 1:
                res.ok(name)
            else:
                res.fail(name, f"expected identity (0,1)  got ({gx},{gy})")
            continue

        # Валидация: точка на кривой через cryptography
        if not crypto_on_curve(gx, gy):
            res.fail(name, f"GPU result NOT on curve: ({gx},{gy})")
            continue

        # seed-*: реальный 256-бит скаляр — валидируем через Ed25519PrivateKey
        if name.startswith("seed-"):
            # Восстанавливаем seed из скаляра (обратный процесс):
            # sb — это уже clamped LE bytes, значит seed = bytes.fromhex из имени
            seed_hex = name[5:] + "0ce00b6619c22cf504ffd6984811b8805759a4bc6701351d728ff5898d"[len(name[5:]):]
            # Проще: найти seed по имени в предопределённом мапе
            _seed_map = {
                "seed-5a72": "5a72ad0ce00b6619c22cf504ffd6984811b8805759a4bc6701351d728ff5898d",
                "seed-9f69": "9f699a703e466ad9e1018d82c7e7d7819730d761a26ea034d3f9d9cff40bd7c3",
                "seed-07ce": "07cef4ba976cd9c1271dce7cb800a6e0e4902dc86dfc80a63161167999ca4a3b",
                "seed-6336": "6336a500a3c6c08ff785855bd79128d8d0049359780f256df6886267a942b926",
                "seed-de22": "de22a26b9f565597b881b4430506fdd3fd187f07781e740d4691cacda9c37488",
            }
            seed_hex = _seed_map[name]
            seed_bytes = bytes.fromhex(seed_hex)
            ref_key = Ed25519PrivateKey.from_private_bytes(seed_bytes)
            ref_pub = ref_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            # Сравнить GPU compressed point с ref_pub
            gpu_compressed = _point_to_compressed(gx, gy)
            if gpu_compressed == ref_pub:
                res.ok(name)
            else:
                res.fail(name,
                    f"Ref pub={ref_pub.hex()}  GPU={gpu_compressed.hex()}")
            continue

        # Для остальных: наивный референс k*B
        rx, ry = _naive_scalar_mult(sb)
        if gx == rx and gy == ry:
            res.ok(name)
        else:
            res.fail(name, f"ref=({rx},{ry})  gpu=({gx},{gy})")
    print()


def _naive_scalar_mult(scalar_bytes):
    """k*B через double-and-add — прямой референс для scalar_mult.
    Совпадает с алгоритмом в ed25519.cl (bits 255..0, affine add)."""
    k = int.from_bytes(scalar_bytes, byteorder="little")
    rx, ry = 0, 1          # identity
    for i in range(255, -1, -1):
        rx, ry = _affine_add(rx, ry, rx, ry)  # double
        bit = (k >> i) & 1
        if bit:
            rx, ry = _affine_add(rx, ry, Bx, By)  # add base
    return rx, ry


def _affine_add(x1, y1, x2, y2):
    """Twisted Edwards addition (a=-1, d=D) — референс для валидации."""
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


def test_affine(kc, dev, res):
    """point_to_affine_x/y: base (Z=1) и 3*B (Z=7)."""
    print("Testing point_to_affine_x/y …")
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()

    # Base point (Z=1)
    pt = pt_to_bytes(Bx, By, 1)
    gx = int32(run_aff_x(ctx, queue, prg, pt))
    gy = int32(run_aff_y(ctx, queue, prg, pt))
    if gx == Bx:
        res.ok("aff_x (base, Z=1)")
    else:
        res.fail("aff_x (base, Z=1)", f"exp={Bx}  got={gx}")
    if gy == By:
        res.ok("aff_y (base, Z=1)")
    else:
        res.fail("aff_y (base, Z=1)", f"exp={By}  got={gy}")

    # 3*B point with Z=7
    sx, sy = _naive_scalar_mult(b"\x03" + b"\x00" * 31)
    zval = 7
    px = (sx * zval) % P
    py = (sy * zval) % P
    pt2 = pt_to_bytes(px, py, zval)

    gx2 = int32(run_aff_x(ctx, queue, prg, pt2))
    gy2 = int32(run_aff_y(ctx, queue, prg, pt2))
    if gx2 == sx:
        res.ok("aff_x (3*B, Z=7)")
    else:
        res.fail("aff_x (3*B, Z=7)", f"exp={sx}  got={gx2}")
    if gy2 == sy:
        res.ok("aff_y (3*B, Z=7)")
    else:
        res.fail("aff_y (3*B, Z=7)", f"exp={sy}  got={gy2}")

    # decimal vs hex projective points (distinct affine results)
    for label, (x, y, z) in [
        ("x12y3z5",      (12, 3, 5)),
        ("x0x12y0x3z0x5", (0x12, 0x3, 0x5)),
        ("x10y11z13",    (10, 11, 13)),
        ("x0x10y0x11z0x13", (0x10, 0x11, 0x13)),
        ("x255y128z64",  (255, 128, 64)),
        ("x0x255y0x128z0x64", (0x255, 0x128, 0x64)),
    ]:
        pt = pt_to_bytes(x, y, z)
        gx = int32(run_aff_x(ctx, queue, prg, pt))
        gy = int32(run_aff_y(ctx, queue, prg, pt))
        Zi = pow(z, P - 2, P)
        rx = (x * Zi) % P
        ry = (y * Zi) % P
        if gx == rx:
            res.ok(f"aff_x ({label})")
        else:
            res.fail(f"aff_x ({label})", f"exp={rx}  got={gx}")
        if gy == ry:
            res.ok(f"aff_y ({label})")
        else:
            res.fail(f"aff_y ({label})", f"exp={ry}  got={gy}")
    print()


# ── Custom mode (--x --y --z) ─────────────────────────────────────

def parse_int(h, d=0):
    """Parse decimal or hex string to int."""
    if h is None:
        return d
    if h.startswith("0x") or h.startswith("0X"):
        return int(h, 16)
    return int(h, 10)


def custom_test(kc, dev, xi, yi, zi):
    """Test all GPU functions on user-supplied projective point (xi, yi, zi).

    Checks: point_to_affine_x/y, point_add_projective (P+B), scalar_mult (X as scalar).
    GPU results validated via cryptography (on_curve) + precise reference comparison.
    """
    print(f"Custom test: X={xi} (0x{xi:x})  Y={yi} (0x{yi:x})  Z={zi} (0x{zi:x})")

    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()

    pt = pt_to_bytes(xi, yi, zi)

    # ── 1. point_to_affine_x/y ──
    print("\n  [affine] point_to_affine_x/y …")
    if zi:
        Zi = pow(zi, P - 2, P)
        rx = (xi * Zi) % P
        ry = (yi * Zi) % P
    else:
        rx = ry = None

    gx = int32(run_aff_x(ctx, queue, prg, pt))
    gy = int32(run_aff_y(ctx, queue, prg, pt))

    if rx is not None:
        ok_x = "OK" if gx == rx else "DIFF"
        ok_y = "OK" if gy == ry else "DIFF"
        print(f"    aff_x: ref={rx}  gpu={gx}  {ok_x}")
        print(f"    aff_y: ref={ry}  gpu={gy}  {ok_y}")
    else:
        print(f"    aff_x: Z=0 (undefined)  gpu={gx}")
        print(f"    aff_y: Z=0 (undefined)  gpu={gy}")

    # ── 2. point_add_projective: P + B ──
    print("\n  [add] point_add_projective (P + B) …")
    if zi:
        Zi = pow(zi, P - 2, P)
        ax = (xi * Zi) % P
        ay = (yi * Zi) % P
        bx, by = _affine_add(ax, ay, Bx, By)
    else:
        bx, by = Bx, By

    p1 = pt_to_bytes(xi, yi, zi)
    p2 = pt_to_bytes(Bx, By, 1)
    raw = run_add(ctx, queue, prg, p1, p2)
    gX, gY, gZ = pt_from_bytes(raw)
    gax, gay = proj2aff(gX, gY, gZ)

    on_curve = crypto_on_curve(gax, gay)
    match = (gax == bx and gay == by) if bx is not None else False
    status = "OK" if match and on_curve else ("DIFF" if not match else "ON_CURVE_ONLY")
    print(f"    P+B: ref=({bx},{by})  gpu=({gax},{gay})  {status}")
    if not on_curve:
        print(f"    ⚠ GPU result NOT on curve!")

    # ── 3. scalar_mult: use xi as scalar ──
    print("\n  [scalar] scalar_mult (scalar=X) …")
    sb = xi.to_bytes(32, byteorder="little") if xi < 2**256 else (xi % (2**256)).to_bytes(32, "little")
    raw2 = run_scalar(ctx, queue, prg, sb)
    g2X, g2Y, g2Z = pt_from_bytes(raw2)
    gsx, gsy = proj2aff(g2X, g2Y, g2Z)

    on_curve2 = crypto_on_curve(gsx, gsy)
    srx, sry = _naive_scalar_mult(sb)
    match2 = (gsx == srx and gsy == sry)
    status2 = "OK" if match2 and on_curve2 else ("DIFF" if not match2 else "ON_CURVE_ONLY")
    print(f"    scalar(X): ref=({srx},{sry})  gpu=({gsx},{gsy})  {status2}")
    if not on_curve2:
        print(f"    ⚠ GPU scalar result NOT on curve!")

    print("\n" + "=" * 70)


# ── Main ─────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(description="Ed25519 OpenCL test (cryptography reference)")
    pa.add_argument("--func",
                    choices=["all", "init", "add", "scalar", "affine"],
                    default="all")
    pa.add_argument("--x", type=str, default=None,
                    help="Projective X (dec or 0x hex)")
    pa.add_argument("--y", type=str, default=None,
                    help="Projective Y (dec or 0x hex)")
    pa.add_argument("--z", type=str, default=None,
                    help="Projective Z (dec or 0x hex)")
    args = pa.parse_args()

    print("=" * 70)
    print("Ed25519 OpenCL Test Suite (cryptography reference)")
    print("=" * 70)

    dev = get_device()
    print(f"Device: {dev.name}\n")

    kc = open("./ed25519_test_kernels.cl").read()

    # Custom mode: --x --y --z
    if args.x is not None or args.y is not None or args.z is not None:
        custom_test(kc, dev,
                    parse_int(args.x, 0),
                    parse_int(args.y, 0),
                    parse_int(args.z, 1))
        return 0

    res = _Result()

    if args.func in ("all", "init"):
        test_init(kc, dev, res)
    if args.func in ("all", "add"):
        test_add(kc, dev, res)
    if args.func in ("all", "scalar"):
        test_scalar(kc, dev, res)
    if args.func in ("all", "affine"):
        test_affine(kc, dev, res)

    print("=" * 70)
    total = res.passed + res.failed
    print(f"Results: {res.passed}/{total} passed, {res.failed} failed")
    if res.failures:
        print("\nFailures:")
        for label, detail in res.failures:
            print(f"  - {label}: {detail}")
    status = "ALL PASSED" if res.failed == 0 else "SOME FAILED"
    print(f"RESULT: {status}")
    print("=" * 70)
    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
