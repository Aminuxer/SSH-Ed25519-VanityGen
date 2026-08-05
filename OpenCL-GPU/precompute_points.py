#!/usr/bin/env python3

"""Precompute Ed25519 scalar multiples of base point B (k=0..255).

Uses the SAME projective addition formula from ed25519.cl:
  A=Z1*Z2; B=A^2; C=X1*Y1; Dd=X2*Y2; E=C*Dd*ED_D
  S=B+E; T=B-E
  F=(X1*Y2+X2*Y1)*A; X3=F*T
  F2=(Y1*Y2+X1*X2)*A; Y3=F2*S
  Z3=S*T
All arithmetic mod p = 2^255 - 19.

Outputs __constant ulong ED_TABLE[3072] formatted as hex ULL literals.
"""

import sys
import struct

# ---------------------------------------------------------------------------
# Curve constants (p = 2^255 - 19)
# ---------------------------------------------------------------------------
P = (1 << 255) - 19

# ED_D from ed25519.cl (4 x 64-bit LE limbs)
D = (0x75EB4DCA135978A3
   | 0x00700A4D4141D8AB << 64
   | 0x8CC740797779E898 << 128
   | 0x52036CEE2B6FFE73 << 192)

# Base point B from ed25519.cl
ED_BASE_X = (0xC9562D608F25D51A
           | 0x692CC7609525A7B2 << 64
           | 0xC0A4E231FDD6DC5C << 128
           | 0x216936D3CD6E53FE << 192)

ED_BASE_Y = (0x6666666666666658
           | 0x6666666666666666 << 64
           | 0x6666666666666666 << 128
           | 0x6666666666666666 << 192)

# ---------------------------------------------------------------------------
# Helper: limb conversion
# ---------------------------------------------------------------------------

def to_limbs(val):
    """Python int -> 4 x 64-bit limbs (little-endian)."""
    return [
        val & 0xFFFFFFFFFFFFFFFF,
        (val >> 64) & 0xFFFFFFFFFFFFFFFF,
        (val >> 128) & 0xFFFFFFFFFFFFFFFF,
        (val >> 192) & 0xFFFFFFFFFFFFFFFF,
    ]


def from_limbs(limbs):
    """4 x 64-bit limbs (LE) -> Python int."""
    return (limbs[0]
          | (limbs[1] << 64)
          | (limbs[2] << 128)
          | (limbs[3] << 192))


# ---------------------------------------------------------------------------
# Modular arithmetic (Python bigints handle the reduction)
# ---------------------------------------------------------------------------

def mul_mod(a, b):
    return (a * b) % P


def add_mod(a, b):
    return (a + b) % P


def sub_mod(a, b):
    return (a - b) % P


# ---------------------------------------------------------------------------
# Projective addition — EXACTLY matches ed25519.cl point_add_proj
#
# Input/Output: (X, Y, Z) as Python ints
# Identity: (0, 1, 1)
# ---------------------------------------------------------------------------

def point_add_proj(X1, Y1, Z1, X2, Y2, Z2):
    """Complete projective addition for Ed25519 (a = -1)."""
    # A = Z1 * Z2
    A = mul_mod(Z1, Z2)
    # B = A^2
    B = mul_mod(A, A)
    # C = X1 * Y1
    C = mul_mod(X1, Y1)
    # Dd = X2 * Y2
    Dd = mul_mod(X2, Y2)
    # E = C * Dd * ED_D
    E = mul_mod(C, Dd)
    E = mul_mod(D, E)

    # S = B + E
    S = add_mod(B, E)
    # T = B - E
    T = sub_mod(B, E)

    # F = (X1*Y2 + X2*Y1) * A
    F = add_mod(mul_mod(X1, Y2), mul_mod(X2, Y1))
    F = mul_mod(F, A)
    # X3 = F * T
    X3 = mul_mod(F, T)

    # F2 = (Y1*Y2 + X1*X2) * A  (a = -1 => y1*y2 + x1*x2)
    F2 = add_mod(mul_mod(Y1, Y2), mul_mod(X1, X2))
    F2 = mul_mod(F2, A)
    # Y3 = F2 * S
    Y3 = mul_mod(F2, S)

    # Z3 = S * T
    Z3 = mul_mod(S, T)

    return X3, Y3, Z3


def point_add_projective(X1, Y1, Z1, X2, Y2, Z2):
    """Wrapper with identity shortcuts (matches point_add_projective in CL)."""
    # Identity = (0, 1, 1)
    if X1 == 0 and Y1 == 1 and Z1 == 1:
        return X2, Y2, Z2
    if X2 == 0 and Y2 == 1 and Z2 == 1:
        return X1, Y1, Z1
    return point_add_proj(X1, Y1, Z1, X2, Y2, Z2)


# ---------------------------------------------------------------------------
# Naive double-and-add (for verification)
# ---------------------------------------------------------------------------

def scalar_mult_naive(k, BX, BY, BZ=1):
    """Compute k*B via double-and-add MSB->LSB. For verification only."""
    RX, RY, RZ = 0, 1, 1  # identity
    for i in range(255, -1, -1):
        bit = (k >> i) & 1
        # Double
        XD, YD, ZD = point_add_projective(RX, RY, RZ, RX, RY, RZ)
        # Conditional add
        XA, YA, ZA = point_add_projective(XD, YD, ZD, BX, BY, BZ)
        if bit:
            RX, RY, RZ = XA, YA, ZA
        else:
            RX, RY, RZ = XD, YD, ZD
    return RX, RY, RZ


# ---------------------------------------------------------------------------
# Precomputation: 0*B .. 255*B
# ---------------------------------------------------------------------------

def precompute_table():
    """Build table[k] = (X, Y, Z) for k*B, k=0..255."""
    table = [(0, 1, 1)]  # k=0: identity
    table.append((ED_BASE_X, ED_BASE_Y, 1))  # k=1: base
    for k in range(2, 256):
        Xp, Yp, Zp = table[k - 1]
        Xn, Yn, Zn = point_add_projective(Xp, Yp, Zp, ED_BASE_X, ED_BASE_Y, 1)
        table.append((Xn, Yn, Zn))
    return table


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def to_affine(X, Y, Z):
    """Convert projective (X,Y,Z) to affine (x,y) via modular inverse of Z."""
    if Z == 0:
        return None, None
    Zi = pow(Z, P - 2, P)
    return (X * Zi) % P, (Y * Zi) % P


def verify(table, n=5):
    """Compare first n entries against naive double-and-add (via affine coords).

    Projective coords (X,Y,Z) are only defined up to a common scalar, so we
    convert both to affine (x = X/Z, y = Y/Z) before comparing.
    """
    print("Verifying first {} points against naive double-and-add...".format(n))
    ok = True
    for k in range(n):
        Xv, Yv, Zv = scalar_mult_naive(k, ED_BASE_X, ED_BASE_Y)
        X, Y, Z = table[k]
        xa_t, ya_t = to_affine(X, Y, Z)
        xa_v, ya_v = to_affine(Xv, Yv, Zv)
        if xa_t == xa_v and ya_t == ya_v:
            print("  k={:3d}  OK  (affine x={:08x}... y={:08x}...)".format(
                k, xa_t >> 200 if xa_t else 0, ya_t >> 200 if ya_t else 0))
        else:
            print("  k={:3d}  MISMATCH!".format(k))
            print("    table affine: x={:064x} y={:064x}".format(xa_t, ya_t))
            print("    naive affine: x={:064x} y={:064x}".format(xa_v, ya_v))
            ok = False
    if not ok:
        sys.exit("Verification FAILED!")
    print("All {} checks passed.\n".format(n))


# ---------------------------------------------------------------------------
# Additional cross-checks
# ---------------------------------------------------------------------------

def check_curve_equation(X, Y, Z, label=""):
    """Verify -x^2 + y^2 = 1 + d*x^2*y^2 in affine coords."""
    Zi = pow(Z, P - 2, P)
    xa = (X * Zi) % P
    ya = (Y * Zi) % P
    lhs = (P - xa * xa % P + ya * ya % P) % P
    rhs = (1 + D * xa % P * xa % P * ya % P * ya % P) % P
    status = "OK" if lhs == rhs else "FAIL"
    if label:
        print("  {} curve check: {}".format(label, status))
    return status == "OK"


def cross_check(table):
    """Additional sanity checks on the table."""
    print("Additional cross-checks:")
    # Verify each point satisfies the curve equation
    for k in [0, 1, 10, 255]:
        check_curve_equation(*table[k], "k={}".format(k))

    # Verify additive chain: table[k] + B == table[k+1]
    print("  Chain check (table[k]+B==table[k+1] for k=0..10):")
    for k in range(11):
        Xa, Ya, Za = point_add_projective(*table[k], ED_BASE_X, ED_BASE_Y, 1)
        Xb, Yb, Zb = table[k + 1]
        if Xa == Xb and Ya == Yb and Za == Zb:
            print("    k={}: OK".format(k))
        else:
            print("    k={}: FAIL (coords differ)".format(k))
            sys.exit("Cross-check failed!")


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def hex_limbs(val):
    """Format a 64-bit limb as 0x...ULL."""
    return "0x{:016X}ULL".format(val & 0xFFFFFFFFFFFFFFFF)


def output_constant(table):
    """Print the __constant ulong ED_TABLE[3072] declaration."""
    print("// Ed25519 precomputed scalar multiples of base point B")
    print("//  k*B stored as (X, Y, Z) in projective coords, mod p = 2^255 - 19")
    print("//  Each point: 4 X limbs + 4 Y limbs + 4 Z limbs = 12 ulong entries")
    print("//  ED_TABLE[k*12 +  0.. 3] = X(k*B)")
    print("//  ED_TABLE[k*12 +  4.. 7] = Y(k*B)")
    print("//  ED_TABLE[k*12 +  8..11] = Z(k*B)")
    print("//  256 points x 12 limbs = 3072 total entries")
    print("// Curve constants (for reference):")
    print("//  P = 2^255 - 19")
    print("//  D = {}".format(hex_limbs(D) + " | " + hex_limbs(D >> 64) + " | " + hex_limbs(D >> 128) + " | " + hex_limbs(D >> 192)))
    print("//  Bx= {}".format(hex_limbs(ED_BASE_X) + " | " + hex_limbs(ED_BASE_X >> 64) + " | " + hex_limbs(ED_BASE_X >> 128) + " | " + hex_limbs(ED_BASE_X >> 192)))
    print("//  By= {}".format(hex_limbs(ED_BASE_Y) + " | " + hex_limbs(ED_BASE_Y >> 64) + " | " + hex_limbs(ED_BASE_Y >> 128) + " | " + hex_limbs(ED_BASE_Y >> 192)))
    print()
    print("__constant ulong ED_TABLE[3072] = {")

    total = 0
    for k in range(256):
        X, Y, Z = table[k]
        limbs = to_limbs(X) + to_limbs(Y) + to_limbs(Z)
        line = "    " + ", ".join(hex_limbs(l) for l in limbs)
        if k < 255:
            print(line + ",")
        else:
            print(line)
        total += 12

    print("};")
    print("// Total entries: {}".format(total))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Ed25519 Precomputation: 0*B .. 255*B")
    print("=" * 60)
    print()
    print("Curve parameters:")
    print("  p  = {}".format(P))
    print("  p  = 0x{:x}".format(P))
    print("  D  = 0x{:x}".format(D))
    print("  Bx = 0x{:x}".format(ED_BASE_X))
    print("  By = 0x{:x}".format(ED_BASE_Y))
    print()

    # Precompute
    table = precompute_table()

    # Verify
    verify(table, 5)
    cross_check(table)

    # Output
    output_constant(table)


if __name__ == "__main__":
    main()
