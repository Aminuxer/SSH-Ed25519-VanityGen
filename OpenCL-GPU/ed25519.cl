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
// Precomputed table: 256 projective points 0*B through 255*B
// Each point: (X, Y, Z) = 3 x 4 limbs = 12 ulong entries
// ED_TABLE[k*12 + 0..3]  = X coordinate of k*B
// ED_TABLE[k*12 + 4..7]  = Y coordinate of k*B
// ED_TABLE[k*12 + 8..11] = Z coordinate of k*B
// Table size: 256 * 12 * 8 = 24576 bytes = 24 KB
// ================================================================
#include "ed25519_static_tables.cl"

/* Scalar multiplication: compute scalar * B (base point) on Ed25519.
   Fixed-base windowed method with 8-bit windows and 256 precomputed points.
   32 windows of 8 bits each: 256 doublings + 32 table-lookups = 288 proj_add calls.
   Compared to naive 256+128=384 proj_add calls: ~25% fewer operations.
   Output in projective coordinates (X, Y, Z). */
__inline void scalar_mult(__generic const uchar* scalar_bytes,
                          __generic ulong* result_X, __generic ulong* result_Y, __generic ulong* result_Z) {
    // Read scalar as 4x64-bit limbs
    ulong sl[4];
    zero_256(sl);
    seed_to_scalar(scalar_bytes, sl);

    // Start from identity (0B)
    ulong RX[4], RY[4], RZ[4];
    zero_256(RX); one_256(RY); one_256(RZ);

    // Process 32 windows of 8 bits each, MSB first
    for (int wi = 31; wi >= 0; wi--) {
        // 8 doublings: R = 2^8 * R
        // Use point_add_projective (copies inputs to locals) since R==R
        for (int d = 0; d < 8; d++)
            point_add_projective(RX, RY, RZ, RX, RY, RZ, RX, RY, RZ);

        // Extract 8-bit window from scalar
        // sl[0] = bits 0..63, sl[1] = bits 64..127, sl[2] = bits 128..191, sl[3] = bits 192..255
        // Window wi covers bits [wi*8 .. wi*8+7]
        int limb_idx = wi / 8;          // which 64-bit limb (0..3)
        int bit_start = (wi % 8) * 8;   // bit offset within that limb (0, 8, 16, ..., 56)
        ulong window = (sl[limb_idx] >> bit_start) & 0xFF;

        // Skip identity addition (0B = (0,1,1))
        if (window == 0) continue;

        // Table lookup: ED_TABLE[window*12 + 0..11] = point window*B
        ulong tX[4], tY[4], tZ[4];
        int base = (int)window * 12;
        tX[0] = ED_TABLE[base+0]; tX[1] = ED_TABLE[base+1];
        tX[2] = ED_TABLE[base+2]; tX[3] = ED_TABLE[base+3];
        tY[0] = ED_TABLE[base+4]; tY[1] = ED_TABLE[base+5];
        tY[2] = ED_TABLE[base+6]; tY[3] = ED_TABLE[base+7];
        tZ[0] = ED_TABLE[base+8]; tZ[1] = ED_TABLE[base+9];
        tZ[2] = ED_TABLE[base+10]; tZ[3] = ED_TABLE[base+11];

        // R = R + window*B
        point_add_projective(RX, RY, RZ, tX, tY, tZ, RX, RY, RZ);
    }

    copy_256(RX, result_X);
    copy_256(RY, result_Y);
    copy_256(RZ, result_Z);
}
