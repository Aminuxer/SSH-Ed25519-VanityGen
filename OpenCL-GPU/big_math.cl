// Fixed implementation for 256-bit arithmetic for Curve25519/Ed25519
// This file provides optimized 256-bit integer arithmetic operations
//
// BYTE ORDER CONVENTION:
// All byte<->limb conversion functions use LITTLE-ENDIAN order.
//   limb[0] = least significant 32 bits -> bytes[0..3]
//   limb[7] = most significant 32 bits   -> bytes[28..31]
// This matches OpenCL host device memory layout on x86/x64 platforms.
// Functions operating solely on uint32 limbs (add_256, sub_256, mul_mod_p,
// mod_p_reduce, etc.) are byte-order agnostic.


// Add two 256-bit numbers: r = a + b
__inline void add_256(__generic const uint* a, __generic const uint* b, __generic uint* r) {
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)(sum & 0xFFFFFFFF);
        carry = sum >> 32;
    }
}

// Subtract two 256-bit numbers: r = a - b
__inline void sub_256(__generic const uint* a, __generic const uint* b, __generic uint* r) {
    long borrow = 0;
    for (int i = 0; i < 8; i++) {
        long diff = (long)a[i] - (long)b[i] - borrow;
        r[i] = (uint)(diff & 0xFFFFFFFF);
        borrow = (diff < 0) ? 1 : 0;
    }
}

// Copy a 256-bit value: dst = src
__inline void copy_256(__generic const uint* src, __generic uint* dst) {
    for (int i = 0; i < 8; i++) dst[i] = src[i];
}

// Set a 256-bit value to 1
__inline void one_256(uint* x) {
    x[0] = 1;
    for (int i = 1; i < 8; i++) x[i] = 0;
}

// Set a 256-bit value to 0
__inline void zero_256(uint* x) {
    for (int i = 0; i < 8; i++) x[i] = 0;
}

// Modular reduction: x = x mod (2^255 - 19)
// Input: x in [0, 2^256), output: x in [0, p)
__inline void mod_p_reduce(uint* x) {
    ulong v[8];
    for (int i = 0; i < 8; i++) v[i] = (ulong)x[i];

    for (int iter = 0; iter < 2; iter++) {
        ulong top = v[7] >> 31;
        if (top == 0) break;
        v[7] &= 0x7FFFFFFFULL;
        ulong carry = 0;
        v[0] += top * 19;
        carry = v[0] >> 32;
        v[0] &= 0xFFFFFFFFULL;
        for (int i = 1; i < 8; i++) {
            v[i] += carry;
            carry = v[i] >> 32;
            v[i] &= 0xFFFFFFFFULL;
        }
    }

    int do_sub = 0;
    if (v[7] == 0x7FFFFFFFULL) {
        int all_max = 1;
        for (int i = 1; i < 7; i++) {
            if (v[i] != 0xFFFFFFFF) { all_max = 0; break; }
        }
        if (all_max && v[0] >= 0xFFFFFFEDULL) {
            do_sub = 1;
        }
    }

    if (do_sub) {
        long borrow = 0;
        long d0 = (long)v[0] - 0xFFFFFFED - borrow;
        v[0] = (ulong)(d0 & 0xFFFFFFFF);
        borrow = (d0 < 0) ? 1 : 0;
        for (int i = 1; i < 7; i++) {
            long di = (long)v[i] - 0xFFFFFFFF - borrow;
            v[i] = (ulong)(di & 0xFFFFFFFF);
            borrow = (di < 0) ? 1 : 0;
        }
        long d7 = (long)v[7] - 0x7FFFFFFF - borrow;
        v[7] = (ulong)(d7 & 0xFFFFFFFF);
    }

    for (int i = 0; i < 8; i++) x[i] = (uint)v[i];
}


// Modular multiplication: r = (a * b) mod p where p = 2^255 - 19
// Uses 32-bit limbs with immediate carry normalization to avoid
// 64-bit overflow in accumulation.
__inline void mul_mod_p(__generic const uint* a, __generic const uint* b, __generic uint* r) {
    // Accumulate into 16 x 32-bit limbs with carry normalization
    // Each addition is done limb-by-limb with carry propagation
    ulong t[16] = {0};

    for (int i = 0; i < 8; i++) {
        ulong ci = 0;  // carry within row i
        for (int j = 0; j < 8; j++) {
            ulong prod = (ulong)a[i] * (ulong)b[j];
            ulong k = (ulong)(i + j);
            // Add prod to t[k] with carry ci
            ulong lo = prod & 0xFFFFFFFFULL;
            ulong hi = prod >> 32;
            ulong s = t[k] + lo + ci;
            t[k] = s & 0xFFFFFFFFULL;
            ci = (s >> 32) + hi;
        }
        // Propagate remaining carry
        ulong k = (ulong)(i + 8);
        while (ci > 0 && k < 16) {
            ulong s = t[k] + ci;
            t[k] = s & 0xFFFFFFFFULL;
            ci = s >> 32;
            k++;
        }
    }

    // Step 2: hi * 19 where hi = product >> 255
    // Extract hi limbs (8 x 32-bit)
    ulong hi[8] = {0};
    hi[0] = ((ulong)(t[7] >> 31)) | ((t[8] & 0x7FFFFFFFULL) << 1);
    for (int k = 1; k < 7; k++) {
        hi[k] = (t[8 + k - 1] >> 31) | ((t[8 + k] & 0x7FFFFFFFULL) << 1);
    }
    hi[7] = (t[14] >> 31) | ((t[15] & 0x7FFFFFFFULL) << 1);

    // lo = bits 0..254: t[0..6] + (t[7] & 0x7FFFFFFF)
    // hi * 19 -> up to 9 limbs, add to lo

    // Multiply hi by 19 limb by limb (no overflow possible)
    ulong h19[9] = {0};
    for (int k = 0; k < 8; k++) {
        ulong prod = hi[k] * 19ULL;
        ulong lo_part = prod & 0xFFFFFFFFULL;
        ulong hi_part = prod >> 32;
        ulong s = h19[k] + lo_part;
        h19[k] = s & 0xFFFFFFFFULL;
        ulong c = (s >> 32) + hi_part;
        ulong s2 = h19[k + 1] + c;
        h19[k + 1] = s2 & 0xFFFFFFFFULL;
        h19[k + 2] += s2 >> 32;
    }

    // Add lo + h19 -> res (9 limbs)
    ulong res[9] = {0};
    ulong carry = 0;
    for (int k = 0; k < 9; k++) {
        ulong lo_val = 0;
        if (k < 7) lo_val = t[k];
        else if (k == 7) lo_val = t[7] & 0x7FFFFFFFULL;

        ulong s = lo_val + h19[k] + carry;
        res[k] = s & 0xFFFFFFFFULL;
        carry = s >> 32;
    }

    // Step 4: Fold bits 255+ of res down
    ulong fold_bits = (res[7] >> 31) | ((res[8] & 0x7FFFFFFFULL) << 1);
    res[7] &= 0x7FFFFFFFULL;

    if (fold_bits > 0) {
        ulong add = fold_bits * 19ULL;
        carry = 0;
        for (int k = 0; k < 3; k++) {
            ulong s = res[k] + (add & 0xFFFFFFFFULL) + carry;
            res[k] = s & 0xFFFFFFFFULL;
            carry = s >> 32;
            if (k == 0) add >>= 32;
        }
    }

    // Step 5: Copy and final reduction
    for (int k = 0; k < 8; k++) r[k] = (uint)res[k];
    mod_p_reduce(r);
}


// Modular inverse: result = a^(-1) mod p where p = 2^255 - 19
__inline void mod_p_inverse(__generic const uint* a, __generic uint* result) {
    uint base[8];
    copy_256(a, base);
    mod_p_reduce(base);

    uint exp[8] = {0xFFFFFFEB, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF};

    uint res[8];
    one_256(res);

    for (int i = 0; i < 255; i++) {
        int bit_idx = i / 32;
        int bit_shift = i % 32;

        if (exp[bit_idx] & (1u << bit_shift)) {
            uint temp[8];
            mul_mod_p(res, base, temp);
            copy_256(temp, res);
        }

        uint temp[8];
        mul_mod_p(base, base, temp);
        copy_256(temp, base);
    }

    copy_256(res, result);
}

// Convert 32-byte seed to 256-bit scalar (LITTLE-ENDIAN)
// Byte 0 (least significant) -> limb 0 bits 0..7
// Consistent with scalar_to_bytes — both use LE byte order.
__inline void seed_to_scalar(__generic const uchar* seed, __generic uint* scalar) {
    for (int i = 0; i < 32; i++) {
        scalar[i/4] |= ((uint)seed[i]) << ((i % 4) * 8);
    }
}

// Convert 256-bit scalar to 32-byte array (LITTLE-ENDIAN)
// Limb i=0 (least significant) -> bytes[0..3] (least significant first)
// Consistent with seed_to_scalar which reads bytes in LE order.
__inline void scalar_to_bytes(__generic const uint* scalar, __generic uchar* bytes) {
    for (int i = 0; i < 8; i++) {
        bytes[i*4 + 0] = (uchar)((scalar[i] >>  0) & 0xFF);
        bytes[i*4 + 1] = (uchar)((scalar[i] >>  8) & 0xFF);
        bytes[i*4 + 2] = (uchar)((scalar[i] >> 16) & 0xFF);
        bytes[i*4 + 3] = (uchar)((scalar[i] >> 24) & 0xFF);
    }
}
