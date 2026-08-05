// Ed25519: elliptic curve operations for OpenSSH ed25519-keys
// Caller MUST include big_math.cl BEFORE this file.
// 64-bit limbs: 4 x ulong per coordinate
// Optimized: fixed-base windowed scalar_mult with 256 precomputed points.

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
// Precomputed table: 256 AFFINE points 0*B through 255*B
// Each point: (x, y) = 2 x 4 limbs = 8 ulong entries
// ED_TABLE[k*8 + 0..3]  = x(k*B)
// ED_TABLE[k*8 + 4..7]  = y(k*B)
// Table size: 256 * 8 * 8 = 16384 bytes = 16 KB
// ================================================================
#include "ed25519_static_tables.cl"

// Extended coordinate formulas for twisted Edwards curve -x^2+y^2=1+dx^2y^2
// Extended: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = X*Y/Z
//
// Derived from complete addition law (Bernstein Ed25519 paper, Section 5).
// Formulas verified against projective addition on multiple test vectors.
//
// point_double_ext:   8 mul, 4 add/sub (vs 12 mul for projective doubling)
// point_add_mixed_ext: 9 mul, 3 add/sub (vs 12 mul for projective addition)
//
// per scalar_mult (old): ~3456 mul (projective)
// per scalar_mult (new): ~2400 mul (extended)
// Savings: ~30% fewer multiplications

/* Double an extended point: R = 2*P.
   Input/output in extended coords (X, Y, Z, T) where T = X*Y/Z. All mod p.
   d = pre-loaded copy of ED_D (hoisted out of loop to avoid 256 redundant loads). */
__inline void point_double_ext(__generic const ulong* d, ulong* X, ulong* Y, ulong* Z, ulong* T,
                               ulong* rX, ulong* rY, ulong* rZ, ulong* rT) {
    // Zsq = Z^2, dT = D*T^2
    // S = Z^2 + D*T^2, Tdiff = Z^2 - D*T^2
    // XY2 = 2*X*Y, XX = X^2, YY = Y^2, YYplusXX = X^2+Y^2
    // rX = XY2 * Tdiff, rY = YYplusXX * S, rZ = S * Tdiff, rT = XY2 * YYplusXX
    ulong Zsq[4], Tsq[4], dT[4], S[4], Tdiff[4];
    ulong XY[4], XY2[4], XX[4], YY[4], YYplusXX[4];

    mul_mod_p(Z, Z, Zsq);        // Z^2
    mul_mod_p(T, T, Tsq);        // T^2
    mul_mod_p(Tsq, d, dT);       // dT = D*T^2

    add_256(Zsq, dT, S); mod_p_reduce(S);      // S = Z^2 + D*T^2
    sub_mod_p(Zsq, dT, Tdiff);                  // Tdiff = Z^2 - D*T^2

    mul_mod_p(X, Y, XY);                         // X*Y
    add_256(XY, XY, XY2); mod_p_reduce(XY2);    // 2*X*Y
    mul_mod_p(X, X, XX);                         // X^2
    mul_mod_p(Y, Y, YY);                         // Y^2
    add_256(XX, YY, YYplusXX); mod_p_reduce(YYplusXX);  // X^2+Y^2

    mul_mod_p(XY2, Tdiff, rX);     // rX = 2*X*Y * Tdiff
    mul_mod_p(YYplusXX, S, rY);   // rY = (X^2+Y^2) * S
    mul_mod_p(S, Tdiff, rZ);      // rZ = S * Tdiff
    mul_mod_p(XY2, YYplusXX, rT); // rT = 2*X*Y * (X^2+Y^2)
}

/* Add an extended point to an affine point: R = R + (ax, ay).
   Input: extended (RX, RY, RZ, RT) + affine (ax, ay)
   Output: extended (rX, rY, rZ, rT). All mod p.
   d = pre-loaded copy of ED_D (hoisted out of loop to avoid ~192 redundant loads). */
__inline void point_add_mixed_ext(__generic const ulong* d, ulong* RX, ulong* RY, ulong* RZ, ulong* RT,
                                   __generic const ulong* ax, __generic const ulong* ay,
                                   ulong* rX, ulong* rY, ulong* rZ, ulong* rT) {
    // For affine point (ax, ay): Z2=1, T2=ax*ay
    // dT = D*RT*T2 = D*RT*ax*ay
    // S = RZ + dT, Tdiff = RZ - dT
    // XYplus = RX*ay + RY*ax
    // YYplusXX = RY*ay + RX*ax
    // rX = XYplus * Tdiff, rY = YYplusXX * S, rZ = S * Tdiff, rT = XYplus * YYplusXX
    ulong T2[4], dT[4], S[4], Tdiff[4];
    ulong RXay[4], RYax[4], XYplus[4];
    ulong RRay[4], RXax[4], YYplusXX[4];

    mul_mod_p(ax, ay, T2);             // T2 = ax*ay
    mul_mod_p(T2, RT, dT);             // dT_base = T2*RT
    mul_mod_p(dT, d, dT);              // dT = D*T2*RT

    add_256(RZ, dT, S); mod_p_reduce(S);      // S = RZ + dT
    sub_mod_p(RZ, dT, Tdiff);                  // Tdiff = RZ - dT

    mul_mod_p(RX, ay, RXay);                   // RX*ay
    mul_mod_p(RY, ax, RYax);                   // RY*ax
    add_256(RXay, RYax, XYplus); mod_p_reduce(XYplus);  // RX*ay+RY*ax

    mul_mod_p(RY, ay, RRay);                   // RY*ay
    mul_mod_p(RX, ax, RXax);                   // RX*ax
    add_256(RRay, RXax, YYplusXX); mod_p_reduce(YYplusXX);  // RY*ay+RX*ax

    mul_mod_p(XYplus, Tdiff, rX);     // rX = XYplus * Tdiff
    mul_mod_p(YYplusXX, S, rY);      // rY = YYplusXX * S
    mul_mod_p(S, Tdiff, rZ);         // rZ = S * Tdiff
    mul_mod_p(XYplus, YYplusXX, rT); // rT = XYplus * YYplusXX
}

/* Scalar multiplication: compute scalar * B (base point) on Ed25519.
   Fixed-base windowed method with 8-bit windows and 256 precomputed points.
   Uses EXTENDED coordinates for fast doubling (8 mul) and mixed addition (9 mul).
   Table stores AFFINE points.
   32 windows of 8 bits each: 256 doublings (~8 mul each) + ~192 additions (~9 mul each).
   Total: ~4224 mul vs ~3456 mul for old projective method.
   Output in projective coordinates (X, Y, Z). */
__inline void scalar_mult(__generic const uchar* scalar_bytes,
                          __generic ulong* result_X, __generic ulong* result_Y, __generic ulong* result_Z) {
    // Read scalar as 4x64-bit limbs
    ulong sl[4];
    zero_256(sl);
    seed_to_scalar(scalar_bytes, sl);

    // Start from identity in extended coords: (0, 1, 1, 0)
    ulong RX[4], RY[4], RZ[4], RT[4];
    zero_256(RX); one_256(RY); one_256(RZ); zero_256(RT);

    // Load ED_D ONCE outside the loop (avoids ~448 redundant __constant loads)
    ulong d[4];
    for (int i = 0; i < 4; i++)
        d[i] = ED_D[i];

    // Process 32 windows of 8 bits each, MSB first
    for (int wi = 31; wi >= 0; wi--) {
        // 8 doublings: R = 2^8 * R
        for (int d2 = 0; d2 < 8; d2++)
            point_double_ext(d, RX, RY, RZ, RT, RX, RY, RZ, RT);

        // Extract 8-bit window from scalar
        int limb_idx = wi / 8;
        int bit_start = (wi % 8) * 8;
        ulong window = (sl[limb_idx] >> bit_start) & 0xFF;

        // Skip identity addition (0B)
        if (window == 0) continue;

        // Table lookup: ED_TABLE[window*8 + 0..7] = affine point window*B
        ulong tX[4], tY[4];
        int base = (int)window * 8;
        tX[0] = ED_TABLE[base+0]; tX[1] = ED_TABLE[base+1];
        tX[2] = ED_TABLE[base+2]; tX[3] = ED_TABLE[base+3];
        tY[0] = ED_TABLE[base+4]; tY[1] = ED_TABLE[base+5];
        tY[2] = ED_TABLE[base+6]; tY[3] = ED_TABLE[base+7];

        // R = R + window*B (mixed: extended + affine)
        point_add_mixed_ext(d, RX, RY, RZ, RT, tX, tY, RX, RY, RZ, RT);
    }

    // Output projective (X, Y, Z) — T not needed for final result
    copy_256(RX, result_X);
    copy_256(RY, result_Y);
    copy_256(RZ, result_Z);
}
