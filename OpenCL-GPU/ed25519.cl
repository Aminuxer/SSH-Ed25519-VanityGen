// Ed25519: elliptic curve operations for OpenSSH ed25519-keys

// Caller MUST include big_math.cl BEFORE this file.
// 64-bit limbs: 4 x ulong per coordinate
// Optimized: fixed-base 4-bit signed-window ladder over 256 precomputed niels points.

__constant ulong ED_D[4] = {
    0x75EB4DCA135978A3ULL, 0x00700A4D4141D8ABULL,
    0x8CC740797779E898ULL, 0x52036CEE2B6FFE73ULL
};

__constant ulong ED_BASE_X[4] = {
    0xC9562D608F25D51AULL, 0x692CC7609525A7B2ULL,
    0xC0A4E231FDD6DC5CULL, 0x216936D3CD6E53FEULL
};

__constant ulong ED_BASE_Y[4] = {
    0x6666666666666658ULL, 0x6666666666666666ULL,
    0x6666666666666666ULL, 0x6666666666666666ULL
};

// Read/write 32 bytes as 4 x ulong (LE)
/* Read 32 bytes from input array at given offset into 4x64-bit limbs (little-endian). */
__inline void read_32bytes(__generic const uchar* inp, int offset, __generic ulong* limb) {
    for(int i=0;i<4;i++)
        limb[i] = ((ulong)inp[offset+i*8]   ) |
                  (((ulong)inp[offset+i*8+1]) <<  8) |
                  (((ulong)inp[offset+i*8+2]) << 16) |
                  (((ulong)inp[offset+i*8+3]) << 24) |
                  (((ulong)inp[offset+i*8+4]) << 32) |
                  (((ulong)inp[offset+i*8+5]) << 40) |
                  (((ulong)inp[offset+i*8+6]) << 48) |
                  (((ulong)inp[offset+i*8+7]) << 56);
}

/* Write 4x64-bit limbs to output array at given offset as 32 bytes (little-endian). */
__inline void write_32bytes(__generic uchar* out, int offset, __generic const ulong* limb) {
    for(int i=0;i<4;i++){
        out[offset+i*8+0]=(uchar)(limb[i]&0xFF);
        out[offset+i*8+1]=(uchar)((limb[i]>>8)&0xFF);
        out[offset+i*8+2]=(uchar)((limb[i]>>16)&0xFF);
        out[offset+i*8+3]=(uchar)((limb[i]>>24)&0xFF);
        out[offset+i*8+4]=(uchar)((limb[i]>>32)&0xFF);
        out[offset+i*8+5]=(uchar)((limb[i]>>40)&0xFF);
        out[offset+i*8+6]=(uchar)((limb[i]>>48)&0xFF);
        out[offset+i*8+7]=(uchar)((limb[i]>>56)&0xFF);
    }
}

// Subtraction mod p: r = (a - b) mod p (branchless)
/* Subtract two 256-bit values modulo p. Branchless: always computes tmp = a-b and tmp+p,
   then selects based on underflow bit. */
__inline void sub_mod_p(__generic const ulong* a, __generic const ulong* b, __generic ulong* r) {
    ulong tmp[4];
    sub_256(a, b, tmp);
    ulong p[4] = {0xFFFFFFFFFFFFFFEDULL, 0xFFFFFFFFFFFFFFFFULL,
                  0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL};
    ulong r_with_p[4];
    add_256(tmp, p, r_with_p);
    ulong underflow = (tmp[3] >> 63) & 1;
    ulong mask = 0 - underflow;
    r[0] = (mask & r_with_p[0]) | (~mask & tmp[0]);
    r[1] = (mask & r_with_p[1]) | (~mask & tmp[1]);
    r[2] = (mask & r_with_p[2]) | (~mask & tmp[2]);
    r[3] = (mask & r_with_p[3]) | (~mask & tmp[3]);
}

// Projective addition (all parameters are 4 x ulong)
/* Add two projective points P1=(X1,Y1,Z1) and P2=(X2,Y2,Z2) using the complete addition formula
   for Ed25519 with a=-1. Output in (X3,Y3,Z3). All arithmetic mod p. */
__inline void point_add_proj(ulong* X1, ulong* Y1, ulong* Z1, ulong* X2, ulong* Y2, ulong* Z2, ulong* X3, ulong* Y3, ulong* Z3) {
    ulong A[4], B[4], C[4], Dd[4], E[4], F[4], G[4], S[4], T[4];

    mul_mod_p(Z1, Z2, A);
    mul_mod_p(A, A, B);
    mul_mod_p(X1, Y1, C);
    mul_mod_p(X2, Y2, Dd);
    mul_mod_p(C, Dd, E);

    ulong DL[4];
    for(int k=0;k<4;k++) DL[k]=ED_D[k];
    mul_mod_p(DL, E, E);

    add_256(B, E, S); mod_p_reduce(S);
    sub_mod_p(B, E, T);

    mul_mod_p(X1, Y2, F);
    mul_mod_p(X2, Y1, G);
    add_256(F, G, F); mod_p_reduce(F);
    mul_mod_p(F, A, F);
    mul_mod_p(F, T, X3);

    mul_mod_p(Y1, Y2, F);
    mul_mod_p(X1, X2, G);
    add_256(F, G, F); mod_p_reduce(F);
    mul_mod_p(F, A, F);
    mul_mod_p(F, S, Y3);
    mul_mod_p(S, T, Z3);
}

/* Wrapper for point_add_proj: handles identity point shortcuts, copies inputs to local.
   P3 = P1 + P2 in projective coordinates. */
__inline void point_add_projective(__generic const ulong* X1, __generic const ulong* Y1, __generic const ulong* Z1,
                                   __generic const ulong* X2, __generic const ulong* Y2, __generic const ulong* Z2,
                                   __generic ulong* X3, __generic ulong* Y3, __generic ulong* Z3) {
    ulong lx1[4], ly1[4], lz1[4], lx2[4], ly2[4], lz2[4];
    copy_256(X1, lx1); copy_256(Y1, ly1); copy_256(Z1, lz1);
    copy_256(X2, lx2); copy_256(Y2, ly2); copy_256(Z2, lz2);

    int p1id = 1;
    if (lx1[0] != 0 || lx1[1] || lx1[2] || lx1[3]) p1id = 0;
    if (ly1[0] != 1 || ly1[1] || ly1[2] || ly1[3]) p1id = 0;
    if (lz1[0] != 1 || lz1[1] || lz1[2] || lz1[3]) p1id = 0;

    int p2id = 1;
    if (lx2[0] != 0 || lx2[1] || lx2[2] || lx2[3]) p2id = 0;
    if (ly2[0] != 1 || ly2[1] || ly2[2] || ly2[3]) p2id = 0;
    if (lz2[0] != 1 || lz2[1] || lz2[2] || lz2[3]) p2id = 0;

    if (p1id) { copy_256(lx2, X3); copy_256(ly2, Y3); copy_256(lz2, Z3); return; }
    if (p2id) { copy_256(lx1, X3); copy_256(ly1, Y3); copy_256(lz1, Z3); return; }

    point_add_proj(lx1, ly1, lz1, lx2, ly2, lz2, X3, Y3, Z3);
}

/* Initialize a point to the Ed25519 base point B. (X=ED_BASE_X, Y=ED_BASE_Y, Z=1). */
__inline void point_init_base(__generic ulong* X, __generic ulong* Y, __generic ulong* Z) {
    for(int k=0;k<4;k++){ X[k]=ED_BASE_X[k]; Y[k]=ED_BASE_Y[k]; }
    one_256(Z);
}

/* Convert projective point (X, Y, Z) to affine x coordinate as 32-byte LE.
   Uses precomputed squaring chain T for modular inverse of Z. */
__inline void point_to_affine_x(__generic const ulong* T, __generic const ulong* X, __generic const ulong* Y, __generic const ulong* Z, __generic uchar* x) {
    ulong Zi[4], Xa[4];
    mod_p_inverse(T, Zi);
    mul_mod_p(X, Zi, Xa);
    scalar_to_bytes(Xa, x);
}

/* Convert projective point (X, Y, Z) to affine y coordinate as 32-byte LE.
   Uses precomputed squaring chain T for modular inverse of Z. */
__inline void point_to_affine_y(__generic const ulong* T, __generic const ulong* X, __generic const ulong* Y, __generic const ulong* Z, __generic uchar* y) {
    ulong Zi[4], Ya[4];
    mod_p_inverse(T, Zi);
    mul_mod_p(Y, Zi, Ya);
    scalar_to_bytes(Ya, y);
}

// ================================================================
// 4-bit signed-window ladder (donna-style) with 256 niels points.
//
// ED_NIELS_TABLE: 256 entries, row = pos*8 + (w-1), pos 0..31, w 1..8.
// Entry = w * 2^(8*pos) * B in niels form (ysubx = y-x, xaddy = y+x, t2d = 2*d*x*y).
// Full (projective) coordinates: (X, Y, Z, T) with T = X*Y/Z (= x*y*Z).
//
// Cost per scalar_mult: 63 niels adds (7 mul) + 4 full doubles (8 mul)
// + 1 INV_D mul = 449 mul, vs ~2400 for the old 8-bit extended ladder.
// ================================================================
#include "ed25519_static_tables.cl"

/* Load table entry for signed window w at position pos into (ysubx, xaddy, t2d).
   Sign: negated point has (ysubx, xaddy) swapped and t2d negated.
   Returns 1 if the point exists (w != 0), 0 for the identity (w == 0).
   Table read is direct __constant indexing (no generic-pointer helpers). */
__inline int load_niels(int pos, int w, __generic ulong* ysubx, __generic ulong* xaddy, __generic ulong* t2d) {
    int neg = (w < 0) ? 1 : 0;
    int u = neg ? -w : w;
    if (u == 0) return 0;
    int base = (pos * 8 + (u - 1)) * 12;
    ysubx[0] = ED_NIELS_TABLE[base+0]; ysubx[1] = ED_NIELS_TABLE[base+1];
    ysubx[2] = ED_NIELS_TABLE[base+2]; ysubx[3] = ED_NIELS_TABLE[base+3];
    xaddy[0] = ED_NIELS_TABLE[base+4]; xaddy[1] = ED_NIELS_TABLE[base+5];
    xaddy[2] = ED_NIELS_TABLE[base+6]; xaddy[3] = ED_NIELS_TABLE[base+7];
    t2d[0] = ED_NIELS_TABLE[base+8];   t2d[1] = ED_NIELS_TABLE[base+9];
    t2d[2] = ED_NIELS_TABLE[base+10];  t2d[3] = ED_NIELS_TABLE[base+11];
    if (neg) {
        ulong tmp[4];
        for (int i = 0; i < 4; i++) { tmp[i] = ysubx[i]; ysubx[i] = xaddy[i]; xaddy[i] = tmp[i]; }
        ulong zero[4];
        zero_256(zero);
        sub_mod_p(zero, t2d, t2d);
    }
    return 1;
}

/* Decode a niels entry to full coordinates:
   X = xaddy - ysubx, Y = xaddy + ysubx, Z = 2, T = t2d * INV_D. */
__inline void decode_niels(__generic const ulong* ysubx, __generic const ulong* xaddy,
                           __generic const ulong* t2d, __generic const ulong* inv_d,
                           __generic ulong* rX, __generic ulong* rY, __generic ulong* rZ, __generic ulong* rT) {
    sub_mod_p(xaddy, ysubx, rX);
    add_256(xaddy, ysubx, rY); mod_p_reduce(rY);
    rZ[0] = 2; rZ[1] = 0; rZ[2] = 0; rZ[3] = 0;
    mul_mod_p(t2d, inv_d, rT);
}

/* R = R + Q where Q is a fixed-base niels point. 7 muls (donna ge25519_nielsadd2).
   Inputs and outputs may alias (all inputs consumed before first output write). */
__inline void point_nielsadd_full(__generic const ulong* ysubx, __generic const ulong* xaddy, __generic const ulong* t2d,
                                  __generic const ulong* RX, __generic const ulong* RY, __generic const ulong* RZ, __generic const ulong* RT,
                                  __generic ulong* rX, __generic ulong* rY, __generic ulong* rZ, __generic ulong* rT) {
    ulong a[4], e[4], h[4], c[4], f[4], g[4];
    sub_mod_p(RY, RX, a);
    mul_mod_p(a, ysubx, a);
    add_256(RY, RX, e); mod_p_reduce(e);
    mul_mod_p(e, xaddy, e);
    add_256(e, a, h); mod_p_reduce(h);
    sub_mod_p(e, a, e);
    mul_mod_p(RT, t2d, c);
    add_256(RZ, RZ, f);
    mod_p_reduce(f);
    add_256(f, c, g); mod_p_reduce(g);
    sub_mod_p(f, c, f);
    mul_mod_p(e, f, rX);
    mul_mod_p(h, g, rY);
    mul_mod_p(g, f, rZ);
    mul_mod_p(e, h, rT);
}

/* R = 2*P using p1p1 (4 muls) + p1p1_to_full (4 muls). 8 muls total.
   Inputs and outputs may alias (all inputs consumed before first output write). */
__inline void point_double_full(__generic const ulong* X, __generic const ulong* Y, __generic const ulong* Z,
                                __generic ulong* rX, __generic ulong* rY, __generic ulong* rZ, __generic ulong* rT) {
    ulong A[4], B[4], C[4], XY[4], px[4], py[4], pz[4], pt[4];
    mul_mod_p(X, X, A);
    mul_mod_p(Y, Y, B);
    add_256(Z, Z, C);
    mod_p_reduce(C);
    mul_mod_p(C, Z, C);
    add_256(X, Y, XY); mod_p_reduce(XY);
    mul_mod_p(XY, XY, px);
    add_256(B, A, py); mod_p_reduce(py);
    sub_mod_p(B, A, pz);
    sub_mod_p(px, py, px);
    sub_mod_p(C, pz, pt);
    mul_mod_p(px, pt, rX);
    mul_mod_p(py, pz, rY);
    mul_mod_p(pz, pt, rZ);
    mul_mod_p(px, py, rT);
}

/* Scalar multiplication: compute scalar * B (base point) on Ed25519.
   4-bit signed-window ladder over the 256-entry niels table.
   Phase 1: r = entry(0, w[1]); odd windows w[3]..w[63]; then x16 (4 doubles).
   Phase 2: even windows w[0]..w[62].
   Input scalar bytes must be clamped per RFC 8032.
   Output in projective coordinates (X, Y, Z). */
__inline void scalar_mult(__generic const uchar* scalar_bytes,
                          __generic ulong* result_X, __generic ulong* result_Y, __generic ulong* result_Z) {
    // Read clamped scalar as 4x64-bit limbs
    ulong sl[4];
    zero_256(sl);
    seed_to_scalar(scalar_bytes, sl);

    // 64 signed 4-bit windows (donna contract256_window4_modm, branchless)
    int win[64];
    for (int i = 0; i < 64; i++)
        win[i] = (int)((sl[i >> 4] >> ((i & 15) << 2)) & 0xF);
    int carry = 0;
    for (int i = 0; i < 63; i++) {
        int v = win[i] + carry;
        int c1 = v >> 4;
        v &= 15;
        int c2 = v >> 3;
        carry = c1 + c2;
        win[i] = v - (c2 << 4);
    }
    win[63] += carry;

    ulong inv_d[4];
    for (int i = 0; i < 4; i++)
        inv_d[i] = ED_INV_D[i];

    ulong ysubx[4], xaddy[4], t2d[4];
    ulong rX[4], rY[4], rZ[4], rT[4];

    // r = decode(entry(0, w[1])); identity (0, 2, 2, 0) if w[1] == 0
    if (load_niels(0, win[1], ysubx, xaddy, t2d)) {
        decode_niels(ysubx, xaddy, t2d, inv_d, rX, rY, rZ, rT);
    } else {
        zero_256(rX);
        rY[0] = 2; rY[1] = 0; rY[2] = 0; rY[3] = 0;
        rZ[0] = 2; rZ[1] = 0; rZ[2] = 0; rZ[3] = 0;
        zero_256(rT);
    }

    // Phase 1: odd windows 1..63
    for (int pos = 1; pos < 32; pos++) {
        if (load_niels(pos, win[2*pos+1], ysubx, xaddy, t2d))
            point_nielsadd_full(ysubx, xaddy, t2d, rX, rY, rZ, rT, rX, rY, rZ, rT);
    }

    // x16
    for (int d = 0; d < 4; d++)
        point_double_full(rX, rY, rZ, rX, rY, rZ, rT);

    // Phase 2: even windows 0..62
    if (load_niels(0, win[0], ysubx, xaddy, t2d))
        point_nielsadd_full(ysubx, xaddy, t2d, rX, rY, rZ, rT, rX, rY, rZ, rT);
    for (int pos = 1; pos < 32; pos++) {
        if (load_niels(pos, win[2*pos], ysubx, xaddy, t2d))
            point_nielsadd_full(ysubx, xaddy, t2d, rX, rY, rZ, rT, rX, rY, rZ, rT);
    }

    // Output projective (X, Y, Z) — T not needed for final result
    copy_256(rX, result_X);
    copy_256(rY, result_Y);
    copy_256(rZ, result_Z);
}
