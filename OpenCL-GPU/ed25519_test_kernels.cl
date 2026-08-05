// Test kernels for ed25519.cl (64-bit limbs)
// ONLY kernel wrappers - NO computation logic.

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

#include "big_math.cl"
#include "ed25519.cl"

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

// read_32bytes and write_32bytes are available from ed25519.cl (not renamed)

// =====================================================================
// Kernel wrappers - pure I/O, no computation
// =====================================================================

// scalar_mult: input 32 bytes -> output 96 bytes (X:Y:Z)
__kernel void scalar_mult(__global const uchar* scalar_bytes,
                          __global uchar* result_point) {
    ulong rX[4], rY[4], rZ[4];
    kl_scalar_mult(scalar_bytes, rX, rY, rZ);
    write_32bytes(result_point, 0, rX);
    write_32bytes(result_point, 32, rY);
    write_32bytes(result_point, 64, rZ);
}

// point_add_projective: input 96+96 bytes -> output 96 bytes
__kernel void point_add_projective(__global const uchar* p1_bytes,
                                   __global const uchar* p2_bytes,
                                   __global uchar* result_bytes) {
    ulong X1[4], Y1[4], Z1[4];
    ulong X2[4], Y2[4], Z2[4];
    ulong X3[4], Y3[4], Z3[4];

    read_32bytes(p1_bytes, 0, X1);
    read_32bytes(p1_bytes, 32, Y1);
    read_32bytes(p1_bytes, 64, Z1);
    read_32bytes(p2_bytes, 0, X2);
    read_32bytes(p2_bytes, 32, Y2);
    read_32bytes(p2_bytes, 64, Z2);

    kl_point_add_projective(X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3);

    write_32bytes(result_bytes, 0, X3);
    write_32bytes(result_bytes, 32, Y3);
    write_32bytes(result_bytes, 64, Z3);
}

// point_init_base: input dummy -> output 96 bytes
__kernel void point_init_base(__global uchar* dummy,
                              __global uchar* result_bytes) {
    ulong X[4], Y[4], Z[4];
    kl_point_init_base(X, Y, Z);
    write_32bytes(result_bytes, 0, X);
    write_32bytes(result_bytes, 32, Y);
    write_32bytes(result_bytes, 64, Z);
}

// point_to_affine_x: input 96 bytes -> output 32 bytes
__kernel void point_to_affine_x(__global const uchar* point_bytes,
                                __global uchar* x_bytes) {
    ulong X[4], Y[4], Z[4];
    read_32bytes(point_bytes, 0, X);
    read_32bytes(point_bytes, 32, Y);
    read_32bytes(point_bytes, 64, Z);
    // Precompute T table from Z
    ulong T[1020];
    kl_copy_256(Z, T);
    kl_mod_p_reduce(T);
    for (int k = 1; k < 255; k++)
        kl_mul_mod_p(T+(k-1)*4, T+(k-1)*4, T+k*4);
    kl_point_to_affine_x(T, X, Y, Z, x_bytes);
}

// point_to_affine_y: input 96 bytes -> output 32 bytes
__kernel void point_to_affine_y(__global const uchar* point_bytes,
                                __global uchar* y_bytes) {
    ulong X[4], Y[4], Z[4];
    read_32bytes(point_bytes, 0, X);
    read_32bytes(point_bytes, 32, Y);
    read_32bytes(point_bytes, 64, Z);
    // Precompute T table from Z
    ulong T[1020];
    kl_copy_256(Z, T);
    kl_mod_p_reduce(T);
    for (int k = 1; k < 255; k++)
        kl_mul_mod_p(T+(k-1)*4, T+(k-1)*4, T+k*4);
    kl_point_to_affine_y(T, X, Y, Z, y_bytes);
}
