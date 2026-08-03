// Test kernels for big_math2.cl
// These kernels call inline functions for testing

// Rename inline functions to avoid conflicts
#define add_256 inline_add_256
#define sub_256 inline_sub_256
#define copy_256 inline_copy_256
#define one_256 inline_one_256
#define zero_256 inline_zero_256
#define mod_p_reduce inline_mod_p_reduce
#define mul_mod_p inline_mul_mod_p
#define mod_p_inverse inline_mod_p_inverse
#define seed_to_scalar inline_seed_to_scalar
#define scalar_to_bytes inline_scalar_to_bytes

#include "big_math.cl"

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

// Now define kernels with original names
__kernel void add_256(__global const uint* a, __global const uint* b, __global uint* r) {
    inline_add_256(a, b, r);
}
__kernel void sub_256(__global const uint* a, __global const uint* b, __global uint* r) {
    inline_sub_256(a, b, r);
}
__kernel void copy_256(__global const uint* src, __global uint* dst) {
    inline_copy_256(src, dst);
}
__kernel void one_256(__global uint* x) {
    inline_one_256(x);
}
__kernel void zero_256(__global uint* x) {
    inline_zero_256(x);
}
__kernel void mod_p_reduce(__global uint* x) {
    inline_mod_p_reduce(x);
}
__kernel void mul_mod_p(__global const uint* a, __global const uint* b, __global uint* r) {
    inline_mul_mod_p(a, b, r);
}
__kernel void mod_p_inverse(__global const uint* a, __global uint* result) {
    inline_mod_p_inverse(a, result);
}
__kernel void seed_to_scalar(__global const uchar* seed, __global uint* scalar) {
    inline_seed_to_scalar(seed, scalar);
}
__kernel void scalar_to_bytes(__global const uint* scalar, __global uchar* bytes) {
    inline_scalar_to_bytes(scalar, bytes);
}
