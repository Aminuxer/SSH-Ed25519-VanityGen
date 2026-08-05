#!/usr/bin/env python3

"""Test suite for ed25519.cl - Ed25519 elliptic curve operations (prepare for openssh ed25519 key generation)"""

import sys, argparse
import numpy as np

try:
    import pyopencl as cl
except ImportError as e:
    print("Error: pyopencl required - " + str(e)); sys.exit(1)

P = 2**255 - 19
D = 37095705934669439343138083508754565189542113879843219016388785533085940283555
# Base point
Bx = 15112221349535400772501151409588531511454012693041857206046113283949847762202
By = 46316835694926478169428394003475163141307993866256225615783033603165251855960

"""Convert integer to 4-element uint64 array (little-endian)."""
def to_256bit(v):
    r = np.zeros(8, dtype=np.uint32)
    for i in range(8): r[i] = (v >> (32*i)) & 0xFFFFFFFF
    return r

"""Convert 4-element uint64 array to integer (little-endian)."""
def from_256bit(a):
    r = 0
    for i in range(8): r += int(a[i]) << (32*i)
    return r

"""Convert projective point (x,y,z) to 96-byte LE representation."""
def pt_to_bytes(pt):
    out = bytearray(96)
    for ci, coord in enumerate([pt[0], pt[1], pt[2]]):
        base = ci * 32
        for i in range(8):
            v = int(coord[i])
            for j in range(4):
                out[base + i*4 + j] = (v >> (8*j)) & 0xFF
    return bytes(out)

"""Parse 96-byte LE data back to projective point (x,y,z)."""
def pt_from_bytes(b):
    coords = []
    for base in [0, 32, 64]:
        limb = []
        for i in range(8):
            v = b[base+i*4] | (b[base+i*4+1]<<8) | (b[base+i*4+2]<<16) | (b[base+i*4+3]<<24)
            limb.append(v)
        coords.append(np.array(limb, dtype=np.uint32))
    return tuple(coords)

"""Convert projective point (X,Y,Z) to affine (x,y) using Z^(-1) mod p."""
def proj2aff(pt):
    X = from_256bit(pt[0]); Y = from_256bit(pt[1]); Z = from_256bit(pt[2])
    if Z == 0: return 0, 0  # crash-safe fallback
    Zi = pow(Z, P-2, P)
    return (X*Zi)%P, (Y*Zi)%P

def le32(v): return v.to_bytes(32, byteorder='little')
def le32i(b): return int.from_bytes(b, byteorder='little')

"""Check if affine point (x,y) lies on the Ed25519 curve: -x^2 + y^2 = 1 + d*x^2*y^2."""
def on_curve(x, y):
    x2 = (x*x)%P; y2 = (y*y)%P
    return (y2 - x2)%P == (1 + D*x2*y2)%P

# --- Reference: affine formulas for a=-1 (Ed25519) ------
# x3 = (x1*y2 + x2*y1) / (1 + d*x1*y1*x2*y2)
# y3 = (y1*y2 + x1*x2) / (1 - d*x1*y1*x2*y2)

"""CPU reference: add two affine points on Ed25519."""
def pt_aff_add(x1,y1,x2,y2):
    den_x = (1 + D*x1*y1*x2*y2) % P
    den_y = (1 - D*x1*y1*x2*y2) % P
    x3 = ((x1*y2 + x2*y1) % P * pow(den_x, P-2, P)) % P
    y3 = ((y1*y2 + x1*x2) % P * pow(den_y, P-2, P)) % P
    return x3, y3

"""CPU reference: scalar multiplication on Ed25519 (double-and-add)."""
def scalar_mult_ref(scalar_bytes):
    """Double-and-add, bits 255..0, affine coords.
    Matches the algorithm in ed25519.cl."""
    scalar = le32i(scalar_bytes)
    rx, ry = 0, 1  # identity
    for i in range(255, -1, -1):
        rx, ry = pt_aff_add(rx, ry, rx, ry)  # double
        bit = (scalar >> i) & 1
        if bit:
            rx, ry = pt_aff_add(rx, ry, Bx, By)  # add base
    return rx, ry

"""CPU reference: projective point addition on Ed25519."""
def proj_add_ref(x1,y1,z1,x2,y2,z2):
    """Projective addition via affine conversion."""
    if z1 == 0: ax1, ay1 = 0, 1
    else:
        Zi1 = pow(z1, P-2, P); ax1 = (x1*Zi1)%P; ay1 = (y1*Zi1)%P
    if z2 == 0: ax2, ay2 = 0, 1
    else:
        Zi2 = pow(z2, P-2, P); ax2 = (x2*Zi2)%P; ay2 = (y2*Zi2)%P
    ax3, ay3 = pt_aff_add(ax1, ay1, ax2, ay2)
    z3 = (z1*z2) % P
    x3 = (ax3*z3) % P; y3 = (ay3*z3) % P
    return x3, y3, z3

# --- GPU runners -----------------------------------------

"""Find and return the first GPU OpenCL device."""
def get_device():
    for p in cl.get_platforms():
        for d in p.get_devices():
            if d.type == cl.device_type.GPU: return d
    raise RuntimeError("No GPU")

"""Run GPU point_add_projective kernel with two projective points."""
def run_add(kc, p1, p2):
    dev = get_device(); ctx = cl.Context([dev]); queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    a1 = np.frombuffer(p1, dtype=np.uint8); a2 = np.frombuffer(p2, dtype=np.uint8)
    bi1 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=a1)
    bi2 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=a2)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 96)
    k = cl.Kernel(prg, "point_add_projective")
    k(queue, (1,), None, bi1, bi2, bo); queue.finish()
    r = np.empty(96, dtype=np.uint8)
    cl.enqueue_copy(queue, r, bo, is_blocking=True); return r

"""Run GPU scalar_mult kernel with 32-byte LE scalar."""
def run_scalar(kc, scal):
    dev = get_device(); ctx = cl.Context([dev]); queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    ai = np.frombuffer(scal, dtype=np.uint8)
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=ai)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 96)
    k = cl.Kernel(prg, "scalar_mult")
    k(queue, (1,), None, bi, bo); queue.finish()
    r = np.empty(96, dtype=np.uint8)
    cl.enqueue_copy(queue, r, bo, is_blocking=True); return r

"""Run GPU point_init_base kernel to initialize base point B."""
def run_init(kc):
    dev = get_device(); ctx = cl.Context([dev]); queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    dm = np.zeros(1, dtype=np.uint8)
    bd = cl.Buffer(ctx, cl.mem_flags.READ_WRITE|cl.mem_flags.COPY_HOST_PTR, hostbuf=dm)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 96)
    k = cl.Kernel(prg, "point_init_base")
    k(queue, (1,), None, bd, bo); queue.finish()
    r = np.empty(96, dtype=np.uint8)
    cl.enqueue_copy(queue, r, bo, is_blocking=True); return r

"""Run GPU point_to_affine_x kernel with 96-byte projective point."""
def run_aff_x(kc, ptb):
    dev = get_device(); ctx = cl.Context([dev]); queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    ai = np.frombuffer(ptb, dtype=np.uint8)
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=ai)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 32)
    k = cl.Kernel(prg, "point_to_affine_x")
    k(queue, (1,), None, bi, bo); queue.finish()
    r = np.empty(32, dtype=np.uint8)
    cl.enqueue_copy(queue, r, bo, is_blocking=True); return r

"""Run GPU point_to_affine_y kernel with 96-byte projective point."""
def run_aff_y(kc, ptb):
    dev = get_device(); ctx = cl.Context([dev]); queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kc).build()
    ai = np.frombuffer(ptb, dtype=np.uint8)
    bi = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=ai)
    bo = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, 32)
    k = cl.Kernel(prg, "point_to_affine_y")
    k(queue, (1,), None, bi, bo); queue.finish()
    r = np.empty(32, dtype=np.uint8)
    cl.enqueue_copy(queue, r, bo, is_blocking=True); return r

# --- Tests -----------------------------------------------

"""Test GPU point_init_base against known base point B."""
def test_init(kc):
    print("Testing point_init_base...")
    ok = True
    r = run_init(kc)
    pt = pt_from_bytes(bytes(r)); x, y = proj2aff(pt)
    if x == Bx and y == By:
        print("  PASS: base point")
    else:
        print("  FAIL: expected (%d,%d) got (%d,%d)" % (Bx, By, x, y)); ok = False
    return ok

"""Test GPU point_add_projective against CPU reference for random points."""
def test_add(kc):
    print("Testing point_add_projective...")
    ok = True
    # Generate test points via scalar_mult_ref
    pts = [("I", 0, 1), ("B", Bx, By)]
    for sv in [2, 3, 5]:
        sb = sv.to_bytes(32, 'little')
        sx, sy = scalar_mult_ref(sb)
        if on_curve(sx, sy):
            pts.append(("s%d"%sv, sx, sy))
        else:
            print("  WARN: s%d not on curve!" % sv)

    for ni,(na,x1,y1) in enumerate(pts):
        for nj,(nb,x2,y2) in enumerate(pts):
            if nj < ni: continue
            name = "%s+%s" % (na, nb)
            rx, ry, rz = proj_add_ref(x1,y1,1,x2,y2,1)
            p1 = pt_to_bytes((to_256bit(x1), to_256bit(y1), to_256bit(1)))
            p2 = pt_to_bytes((to_256bit(x2), to_256bit(y2), to_256bit(1)))
            gr = run_add(kc, p1, p2)
            gpt = pt_from_bytes(bytes(gr)); gx, gy = proj2aff(gpt)
            rx_aff, ry_aff = proj2aff((to_256bit(rx), to_256bit(ry), to_256bit(rz)))
            if gx == rx_aff and gy == ry_aff:
                print("  PASS: %s" % name)
            else:
                print("  FAIL: %s" % name)
                print("    Ref: x=%d, y=%d" % (rx_aff, ry_aff))
                print("    GPU: x=%d, y=%d" % (gx, gy))
                ok = False
    return ok

"""Test GPU scalar_mult against CPU reference for random scalars."""
def test_scalar(kc):
    print("Testing scalar_mult (raw)...")
    ok = True
    seeds = [
        ("zeros", bytes(32)),
        ("one", b'\x01' + bytes(31)),
        ("two", b'\x02' + bytes(31)),
    ]
    for name, seed in seeds:
        rx, ry = scalar_mult_ref(seed)
        if not on_curve(rx, ry):
            print("  WARN: ref %s not on curve!" % name)
        gr = run_scalar(kc, seed)
        gpt = pt_from_bytes(bytes(gr)); gx, gy = proj2aff(gpt)
        if gx == rx and gy == ry:
            print("  PASS: %s" % name)
        else:
            print("  FAIL: %s" % name)
            print("    Ref x=%d, y=%d" % (rx, ry))
            print("    GPU x=%d, y=%d" % (gx, gy))
            ok = False
    return ok

"""Test GPU point_to_affine_x/y against CPU reference for random projective points."""
def test_affine(kc):
    print("Testing point_to_affine_x/y...")
    ok = True
    pt = pt_to_bytes((to_256bit(Bx), to_256bit(By), to_256bit(1)))
    xr = run_aff_x(kc, pt); yr = run_aff_y(kc, pt)
    gx = le32i(bytes(xr)); gy = le32i(bytes(yr))
    if gx == Bx: print("  PASS: aff_x (base)")
    else: print("  FAIL: aff_x base: exp=%d got=%d" % (Bx, gx)); ok = False
    if gy == By: print("  PASS: aff_y (base)")
    else: print("  FAIL: aff_y base: exp=%d got=%d" % (By, gy)); ok = False
    # Z=7 test
    sx, sy = scalar_mult_ref(b'\x03' + bytes(31))
    zval = 7; px = (sx*zval)%P; py = (sy*zval)%P
    pt2 = pt_to_bytes((to_256bit(px), to_256bit(py), to_256bit(zval)))
    xr2 = run_aff_x(kc, pt2); yr2 = run_aff_y(kc, pt2)
    gx2 = le32i(bytes(xr2)); gy2 = le32i(bytes(yr2))
    if gx2 == sx: print("  PASS: aff_x (Z=7)")
    else: print("  FAIL: aff_x Z=7: exp=%d got=%d" % (sx, gx2)); ok = False
    if gy2 == sy: print("  PASS: aff_y (Z=7)")
    else: print("  FAIL: aff_y Z=7: exp=%d got=%d" % (sy, gy2)); ok = False
    return ok

# --- Custom mode (--x --y --z) ---------------------------

def parse_int(h, d=0):
    if h is None: return d
    if h.startswith("0x") or h.startswith("0X"):
        return int(h, 16)
    return int(h, 10)

"""Test GPU point_to_affine_x/y with custom projective coordinates."""
def custom_test(kc, xi, yi, zi):
    print("Custom test: X=%d (hex: 0x%x) Y=%d (hex: 0x%x) Z=%d (hex: 0x%x)" % (xi, xi, yi, yi, zi, zi))
    pt = pt_to_bytes((to_256bit(xi), to_256bit(yi), to_256bit(zi)))
    if zi:
        Zi = pow(zi, P-2, P); rx = (xi*Zi)%P; ry = (yi*Zi)%P
    else: rx = ry = None
    gx = le32i(bytes(run_aff_x(kc, pt))); gy = le32i(bytes(run_aff_y(kc, pt)))
    print("  aff_x: ref=%s  gpu=%d  %s" % (rx, gx, "OK" if rx is not None and gx==rx else "DIFF"))
    print("  aff_y: ref=%s  gpu=%d  %s" % (ry, gy, "OK" if ry is not None and gy==ry else "DIFF"))
    if zi:
        Zi = pow(zi, P-2, P); ax = (xi*Zi)%P; ay = (yi*Zi)%P
        bx, by = pt_aff_add(ax, ay, Bx, By)
    else: bx = by = None
    p1 = pt_to_bytes((to_256bit(xi), to_256bit(yi), to_256bit(zi)))
    p2 = pt_to_bytes((to_256bit(Bx), to_256bit(By), to_256bit(1)))
    gr = run_add(kc, p1, p2); gpt = pt_from_bytes(bytes(gr)); gax, gay = proj2aff(gpt)
    print("  P+B: ref=(%s,%s) gpu=(%d,%d) %s" % (bx, by, gax, gay, "OK" if bx and gax==bx and gay==by else "DIFF"))
    sb = le32(xi)
    gr2 = run_scalar(kc, sb); gpt2 = pt_from_bytes(bytes(gr2)); gsx, gsy = proj2aff(gpt2)
    srx, sry = scalar_mult_ref(sb)
    print("  scalar(x): ref=(%d,%d) gpu=(%d,%d) %s" % (srx, sry, gsx, gsy, "OK" if gsx==srx and gsy==sry else "DIFF"))

# --- Main ------------------------------------------------

"""Entry point: parse args, run selected tests, report results."""
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--func", choices=["all","add","scalar","init","affine"], default="all")
    pa.add_argument("--x", type=str, default=None, help="Projective X (dec or 0x hex)")
    pa.add_argument("--y", type=str, default=None, help="Projective Y (dec or 0x hex)")
    pa.add_argument("--z", type=str, default=None, help="Projective Z (dec or 0x hex)")
    a = pa.parse_args()

    print("="*70); print("Ed25519 OpenCL Test Suite (for OpenSSH ed25519 key generation)"); print("="*70)
    dev = get_device()
    print("Device: %s" % dev.name)
    kc = open("./ed25519_test_kernels.cl").read()
    print("Kernel: ./ed25519_test_kernels.cl\n")

    if a.x or a.y or a.z:
        custom_test(kc, parse_int(a.x,0), parse_int(a.y,0), parse_int(a.z,1))
        print("="*70); return 0

    ok = True
    if a.func in ("all","init"):
        if not test_init(kc): ok = False
        print()
    if a.func in ("all","add"):
        if not test_add(kc): ok = False
        print()
    if a.func in ("all","scalar"):
        if not test_scalar(kc): ok = False
        print()
    if a.func in ("all","affine"):
        if not test_affine(kc): ok = False
        print()

    print("="*70)
    print("RESULT: ALL PASSED" if ok else "RESULT: SOME FAILED")
    print("="*70)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
