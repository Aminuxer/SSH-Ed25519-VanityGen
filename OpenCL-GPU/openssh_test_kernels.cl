// Test kernels for openssh.cl
//
// Openssh.cl contains __kernel functions: base64_encode, seed_to_ssh_ed25519_pubkey.
// Those are available directly after #include "./openssh.cl".
//
// All inline helpers from big_math.cl and ed25519.cl are renamed to kl_* prefix
// to allow kernel wrappers in this file to call them safely.
// The defines are active BEFORE the openssh.cl include, so the transitive
// includes (big_math.cl, ed25519.cl, sha512.cl) get renamed too.

// Rename big_math inline functions
#define add_256              kl_add_256
#define sub_256              kl_sub_256
#define copy_256             kl_copy_256
#define one_256              kl_one_256
#define zero_256             kl_zero_256
#define mod_p_reduce         kl_mod_p_reduce
#define mul_mod_p            kl_mul_mod_p
#define mod_p_inverse        kl_mod_p_inverse
#define seed_to_scalar       kl_seed_to_scalar
#define scalar_to_bytes      kl_scalar_to_bytes

// Rename ed25519 inline functions
#define point_add_projective kl_point_add_projective
#define point_init_base      kl_point_init_base
#define point_to_affine_x    kl_point_to_affine_x
#define point_to_affine_y    kl_point_to_affine_y
#define scalar_mult          kl_scalar_mult

// Rename openssh inline helpers (NOT kernels: base64_encode and seed_to_ssh_ed25519_pubkey are __kernel)
#define build_ssh_public_blob kl_build_ssh_public_blob

// ─── Include dependencies BEFORE openssh.cl ─────────────────
// sha512.cl must come first (SHA512_H, sha512_transform, little_s0/s1)
#include "./sha512.cl"
// big_math.cl: add_256, mul_mod_p, scalar_to_bytes, etc.
#include "./big_math.cl"
// ed25519.cl: scalar_mult, point_to_affine_y, etc.
#include "./ed25519.cl"
// openssh.cl: no includes — uses everything from above
#include "./openssh.cl"

#undef add_256
#undef sub_256
#undef copy_256
#undef one_256
#undef zero_256
#undef mod_p_reduce
#undef mul_mod_p
#undef mod_p_inverse
#undef seed_to_scalar
#undef scalar_to_bytes
#undef point_add_projective
#undef point_init_base
#undef point_to_affine_x
#undef point_to_affine_y
#undef scalar_mult
#undef build_ssh_public_blob
// NOTE: base64_encode and seed_to_ssh_ed25519_pubkey are __kernel in openssh.cl — not renamed, not undef'd

// =====================================================================
// Additional kernel wrappers for inline helpers from openssh.cl
// =====================================================================

// build_ssh_public_blob: 32-byte pubkey -> 51-byte SSH blob
__kernel void build_ssh_public_blob(
    __global const uchar* pubKey,   // 32 bytes
    __global uchar* blob            // 51 bytes output
) {
    kl_build_ssh_public_blob(pubKey, blob);
}

// =====================================================================
// Pipeline debug kernels for step-by-step testing
// =====================================================================

// sha512_32: compute SHA512 of 32-byte input (single block with padding)
__kernel void sha512_32(
    __global const uchar* seed,   // 32 bytes
    __global uchar* hash_out      // 64 bytes output
) {
    unsigned long state[8];
    for (int i = 0; i < 8; i++) state[i] = SHA512_H[i];

    unsigned long W[80];
    // Copy seed (32 bytes) as 4 x 64-bit big-endian words
    for (int i = 0; i < 4; i++) {
        W[i] = ((unsigned long)seed[i*8] << 56) |
               ((unsigned long)seed[i*8+1] << 48) |
               ((unsigned long)seed[i*8+2] << 40) |
               ((unsigned long)seed[i*8+3] << 32) |
               ((unsigned long)seed[i*8+4] << 24) |
               ((unsigned long)seed[i*8+5] << 16) |
               ((unsigned long)seed[i*8+6] << 8) |
               ((unsigned long)seed[i*8+7]);
    }
    // Pad: 0x80 after 32 bytes, then zeros, then bit-length (256)
    W[4] = 0x8000000000000000UL;
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 256UL;

    for (int i = 16; i < 80; i++) {
        W[i] = W[i-16] + little_s0(W[i-15]) + W[i-7] + little_s1(W[i-2]);
    }
    sha512_transform(state, W);

    // Write hash (big-endian)
    for (int i = 0; i < 8; i++) {
        unsigned long val = state[i];
        hash_out[i*8+0] = (uchar)(val >> 56);
        hash_out[i*8+1] = (uchar)(val >> 48);
        hash_out[i*8+2] = (uchar)(val >> 40);
        hash_out[i*8+3] = (uchar)(val >> 32);
        hash_out[i*8+4] = (uchar)(val >> 24);
        hash_out[i*8+5] = (uchar)(val >> 16);
        hash_out[i*8+6] = (uchar)(val >> 8);
        hash_out[i*8+7] = (uchar)val;
    }
}

// clamp_and_encode: take 32-byte LE scalar (already clamped),
// do scalar_mult, convert to affine, encode as pubkey
__kernel void clamp_and_encode(
    __global const uchar* scalar_in,  // 32 bytes LE scalar (already clamped)
    __global uchar* pub_out           // 32 bytes (public key)
) {
    // Scalar multiplication
    uint rX[8], rY[8], rZ[8];
    kl_scalar_mult(scalar_in, rX, rY, rZ);

    // Convert to affine Y
    uchar y_affine[32];
    kl_point_to_affine_y(rX, rY, rZ, y_affine);

    // Set sign bit based on X parity
    if ((rX[0] & 1u) != 0) {
        y_affine[31] |= 0x80;
    }

    // Write output
    for (int i = 0; i < 32; i++) {
        pub_out[i] = y_affine[i];
    }
}

// scalar_mult_wrapper: run scalar_mult with 32-byte LE input
__kernel void scalar_mult_wrapper(
    __global const uchar* scalar_in,  // 32 bytes LE scalar
    __global uchar* point_out         // 96 bytes (X:Y:Z projective)
) {
    uint rX[8], rY[8], rZ[8];
    kl_scalar_mult(scalar_in, rX, rY, rZ);

    // Write X (32 bytes LE)
    for (int i = 0; i < 8; i++) {
        point_out[i*4+0] = (uchar)(rX[i] & 0xFF);
        point_out[i*4+1] = (uchar)((rX[i] >> 8) & 0xFF);
        point_out[i*4+2] = (uchar)((rX[i] >> 16) & 0xFF);
        point_out[i*4+3] = (uchar)((rX[i] >> 24) & 0xFF);
    }
    // Write Y (32 bytes LE)
    for (int i = 0; i < 8; i++) {
        point_out[32 + i*4+0] = (uchar)(rY[i] & 0xFF);
        point_out[32 + i*4+1] = (uchar)((rY[i] >> 8) & 0xFF);
        point_out[32 + i*4+2] = (uchar)((rY[i] >> 16) & 0xFF);
        point_out[32 + i*4+3] = (uchar)((rY[i] >> 24) & 0xFF);
    }
    // Write Z (32 bytes LE)
    for (int i = 0; i < 8; i++) {
        point_out[64 + i*4+0] = (uchar)(rZ[i] & 0xFF);
        point_out[64 + i*4+1] = (uchar)((rZ[i] >> 8) & 0xFF);
        point_out[64 + i*4+2] = (uchar)((rZ[i] >> 16) & 0xFF);
        point_out[64 + i*4+3] = (uchar)((rZ[i] >> 24) & 0xFF);
    }
}

// point_to_affine_y_wrapper: convert projective point to affine Y
__kernel void point_to_affine_y_wrapper(
    __global const uchar* point_in,  // 96 bytes (X:Y:Z)
    __global uchar* y_out            // 32 bytes LE
) {
    uint X[8], Y[8], Z[8];
    for (int i = 0; i < 8; i++) {
        X[i] = point_in[i*4+0] | (point_in[i*4+1] << 8) | (point_in[i*4+2] << 16) | (point_in[i*4+3] << 24);
        Y[i] = point_in[32 + i*4+0] | (point_in[32 + i*4+1] << 8) | (point_in[32 + i*4+2] << 16) | (point_in[32 + i*4+3] << 24);
        Z[i] = point_in[64 + i*4+0] | (point_in[64 + i*4+1] << 8) | (point_in[64 + i*4+2] << 16) | (point_in[64 + i*4+3] << 24);
    }
    kl_point_to_affine_y(X, Y, Z, y_out);
}

// debug_pipeline: same as seed_to_ssh_ed25519_pubkey but outputs intermediate values
__kernel void debug_pipeline(
    __global const uchar* seed,       // 32 bytes
    __global uchar* hash_out,         // 64 bytes
    __global uchar* scalar_out,       // 32 bytes LE
    __global uchar* pub_out           // 32 bytes
) {
    // 1. SHA512(seed)
    unsigned long state[8];
    for (int i = 0; i < 8; i++) state[i] = SHA512_H[i];
    unsigned long W[80];
    for (int i = 0; i < 4; i++) {
        W[i] = ((unsigned long)seed[i*8] << 56) |
               ((unsigned long)seed[i*8+1] << 48) |
               ((unsigned long)seed[i*8+2] << 40) |
               ((unsigned long)seed[i*8+3] << 32) |
               ((unsigned long)seed[i*8+4] << 24) |
               ((unsigned long)seed[i*8+5] << 16) |
               ((unsigned long)seed[i*8+6] << 8) |
               ((unsigned long)seed[i*8+7]);
    }
    W[4] = 0x8000000000000000UL;
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 256UL;
    for (int i = 16; i < 80; i++) {
        W[i] = W[i-16] + little_s0(W[i-15]) + W[i-7] + little_s1(W[i-2]);
    }
    sha512_transform(state, W);
    for (int i = 0; i < 8; i++) {
        unsigned long val = state[i];
        hash_out[i*8+0] = (uchar)(val >> 56);
        hash_out[i*8+1] = (uchar)(val >> 48);
        hash_out[i*8+2] = (uchar)(val >> 40);
        hash_out[i*8+3] = (uchar)(val >> 32);
        hash_out[i*8+4] = (uchar)(val >> 24);
        hash_out[i*8+5] = (uchar)(val >> 16);
        hash_out[i*8+6] = (uchar)(val >> 8);
        hash_out[i*8+7] = (uchar)val;
    }

    // 2. Scalar derivation
    uchar scalar_bytes[32];
    for (int i = 0; i < 32; i++) {
        scalar_bytes[i] = hash_out[31 - i];
    }
    scalar_bytes[0] &= 0x7F;   // clear top bit
    scalar_bytes[0] |= 0x40;   // set top-1 bit
    scalar_bytes[31] &= 0xF8;  // clear bottom 3 bits (NaCl clamping)
    for (int i = 0; i < 32; i++) {
        scalar_out[i] = scalar_bytes[i];
    }

    // 3. Scalar multiplication
    uint rX[8], rY[8], rZ[8];
    kl_scalar_mult(scalar_bytes, rX, rY, rZ);

    // 4. Point to affine Y
    kl_point_to_affine_y(rX, rY, rZ, pub_out);
    if ((rX[0] & 1u) != 0) {
        pub_out[31] |= 0x80;
    }
}
