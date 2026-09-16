#!/usr/bin/env python3
"""Generate ed25519_static_tables.cl -- the fixed-base table for scalar_mult
(ed25519.cl): 4-bit signed-window ladder (donna method).

Self-contained: all curve data is embedded below, no external files.
Every entry (pos, w), pos = 0..31, w = 1..8, is the point
    w * 2^(8*pos) * B
where B is the Ed25519 base point, computed here by double-and-add on the
twisted Edwards curve -x^2 + y^2 = 1 + d*x^2*y^2 (a = -1).

Entry encoding (niels form, all mod p, 4 x 64-bit LE limbs):
    ysubx = y - x
    xaddy = y + x
    t2d   = 2*d*x*y
Layout in the generated file:
    ED_NIELS_TABLE[((pos*8 + (w-1)) * 12 + 0..3]  = ysubx
    ED_NIELS_TABLE[((pos*8 + (w-1)) * 12 + 4..7]  = xaddy
    ED_NIELS_TABLE[((pos*8 + (w-1)) * 12 + 8..11] = t2d
Also emitted: ED_INV_D = d^-1 mod p (the niels decode uses T = t2d * d^-1).

Checks performed before writing:
  1. embedded base point is valid (even x, on the curve)
  2. RFC 8032 section 6.1 test vectors 1-2 reproduce exactly
     (seed -> clamped scalar -> s*B -> 32-byte pubkey), proving the
     embedded B is the standard Ed25519 base point
  3. every table entry lies on the curve
  4. entry(pos, w1) + entry(pos, w2) == entry(pos, w1+w2), all w1<w2, w1+w2<=8
  5. entry(pos+1, w) == 256 * entry(pos, w)   (8 doublings), all w
  6. t2d * d^-1 == 2*x*y for every entry      (decode convention)
"""
import hashlib
import os

P = 2 ** 255 - 19
D = (-121665 * pow(121666, P - 2, P)) % P
INV2 = (P + 1) // 2
INV_D = pow(D, P - 2, P)

# Ed25519 base point B, 4 x 64-bit LE limbs (identical to ED_BASE_X / ED_BASE_Y
# in ed25519.cl). Values:
#   x_B = 0x216936D3CD6E53FEC0A4E231FDD6DC5C692CC7609525A7B2C9562D608F25D51A
#   y_B = 0x6666666666666666666666666666666666666666666666666666666666666658
BX = (0xC9562D608F25D51A, 0x692CC7609525A7B2,
      0xC0A4E231FDD6DC5C, 0x216936D3CD6E53FE)
BY = (0x6666666666666658, 0x6666666666666666,
      0x6666666666666666, 0x6666666666666666)


def from_limbs(l):
    return sum(v << (64 * i) for i, v in enumerate(l))


def to_limbs(v):
    return [(v >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(4)]


def on_curve(pt_):
    x, y = pt_
    # twisted edwards: -x^2 + y^2 = 1 + d*x^2*y^2
    return (y * y - x * x - 1 - D * x * x * y * y) % P == 0


def add_pt(A, C):
    x1, y1 = A
    x2, y2 = C
    x3 = (x1 * y2 + y1 * x2) * pow(1 + D * x1 * x2 * y1 * y2, P - 2, P) % P
    y3 = (y1 * y2 + x1 * x2) * pow(1 - D * x1 * x2 * y1 * y2, P - 2, P) % P
    return (x3, y3)


def dbl_pt(A):
    return add_pt(A, A)


def decode_niels(ysubx, xaddy):
    x = (xaddy - ysubx) * INV2 % P
    y = (xaddy + ysubx) * INV2 % P
    return (x, y)


def point_to_pub_bytes(pt_):
    # RFC 8032 5.1.5: 32-byte little-endian encoding of y; the least
    # significant bit of the first byte is set iff x is odd.
    x, y = pt_
    b = bytearray((y & ((1 << 256) - 1)).to_bytes(32, 'little'))
    if x & 1:
        b[0] |= 1
    return bytes(b)


def clamp_seed(seed32):
    h = hashlib.sha512(seed32).digest()[:32]
    h = bytearray(h)
    h[0] &= 248
    h[31] &= 127
    h[31] |= 64
    return int.from_bytes(bytes(h), 'little')


def scalar_mult_plain(s_int, base):
    # MSB-first double-and-add, affine
    IDENT = (0, 1)
    r = IDENT
    nbits = s_int.bit_length()
    for i in range(nbits - 1, -1, -1):
        r = dbl_pt(r)
        if (s_int >> i) & 1:
            r = add_pt(r, base)
    return r


B = (from_limbs(BX), from_limbs(BY))

# ---------------- self-check of the embedded base point ---------------------
assert B[0] % 2 == 0, "BX must be even"
assert on_curve(B), "base point B not on curve"

# RFC 8032 section 6.1 test vectors (golden check: proves B is the standard
# Ed25519 base point, independently of any external table).
RFC_VECTORS = [
    (bytes.fromhex('9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60'),
     'd75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a'),
    (bytes.fromhex('4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb'),
     '3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c'),
]
for idx, (seed, expected_pub) in enumerate(RFC_VECTORS, 1):
    got = point_to_pub_bytes(scalar_mult_plain(clamp_seed(seed), B)).hex()
    assert got == expected_pub, \
        f"RFC 8032 vec{idx} FAIL: got {got}, want {expected_pub}"
print("RFC 8032 6.1 test vectors 1-2: OK")

# ---------------- generate all 256 entries ----------------------------------
IDENT = (0, 1)
table = []          # index (pos*8 + (w-1)) -> (ysubx, xaddy, t2d)
P_base = B          # 2^(8*pos) * B
for pos in range(32):
    acc = IDENT
    for w in range(1, 9):
        acc = add_pt(acc, P_base)
        x, y = acc
        assert on_curve(acc), f"entry ({pos},{w}) off curve"
        table.append(((y - x) % P, (y + x) % P, (2 * D * x * y) % P))
    for _ in range(8):
        P_base = dbl_pt(P_base)

# ---------------- check 4: closure under addition (w1+w2 <= 8) --------------
for pos in range(32):
    for w1 in range(1, 8):
        for w2 in range(w1 + 1, 9 - w1):
            s = add_pt(decode_niels(*table[pos * 8 + w1 - 1][:2]),
                       decode_niels(*table[pos * 8 + w2 - 1][:2]))
            assert s == decode_niels(*table[pos * 8 + w1 + w2 - 1][:2]), \
                f"add-check FAIL pos={pos} w1={w1} w2={w2}"
print("CHECK (entry(w1)+entry(w2)==entry(w1+w2)): OK")

# ---------------- check 5: entry(pos+1, w) == 256 * entry(pos, w) -----------
for pos in range(31):
    for w in range(1, 9):
        pt_ = decode_niels(*table[pos * 8 + w - 1][:2])
        for _ in range(8):
            pt_ = dbl_pt(pt_)
        assert pt_ == decode_niels(*table[(pos + 1) * 8 + w - 1][:2]), \
            f"scale-check FAIL pos={pos} w={w}"
print("CHECK (row pos+1 == 256 * row pos): OK")

# ---------------- check 6: t2d * d^-1 == 2*x*y (decode convention) ---------
for e, (ys, xa, t2d) in enumerate(table):
    x, y = decode_niels(ys, xa)
    assert t2d * INV_D % P == 2 * x * y % P, f"t2d convention FAIL entry {e}"
print("CHECK (t2d * INV_D == 2*x*y): OK")

# ---------------- emit the table file ---------------------------------------
out = []
out.append("// Auto-generated by precompute_points.py -- do not edit by hand.")
out.append("// 4-bit window fixed-base niels table for scalar_mult (donna method).")
out.append("// Entry (pos, w), pos = 0..31, w = 1..8: point w*2^(8*pos)*B in niels form:")
out.append("//   ysubx = y - x, xaddy = y + x, t2d = 2*d*x*y   (all mod p, 4x ulong LE)")
out.append("// Layout: ED_NIELS_TABLE[((pos*8 + (w-1)) * 12 + 0..3]  = ysubx")
out.append("//         ED_NIELS_TABLE[((pos*8 + (w-1)) * 12 + 4..7]  = xaddy")
out.append("//         ED_NIELS_TABLE[((pos*8 + (w-1)) * 12 + 8..11] = t2d")
out.append("// Size: 256 entries * 96 bytes = 24576 bytes = 24 KB")
out.append("")
out.append("// Curve constant d: d = -121665/121666 mod p")
out.append("// d^-1 (used by the niels decode: T = t2d * d^-1):")
inv_d_l = to_limbs(INV_D)
out.append("__constant ulong ED_INV_D[4] = {")
out.append("    0x%016XULL, 0x%016XULL," % (inv_d_l[0], inv_d_l[1]))
out.append("    0x%016XULL, 0x%016XULL};\n" % (inv_d_l[2], inv_d_l[3]))
out.append("__constant ulong ED_NIELS_TABLE[3072] = {")
first = True
for ys, xa, t2d in table:
    line = []
    for v in (ys, xa, t2d):
        line += ["0x%016XULL" % vv for vv in to_limbs(v)]
    prefix = "" if first else ",\n"
    first = False
    out.append(prefix + "    " + ", ".join(line))
out.append("};")

text = "\n".join(out) + "\n"
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'ed25519_static_tables.cl')
open(dest, 'w').write(text)
print(f"wrote {dest} ({len(text)} bytes, 256 entries)")
