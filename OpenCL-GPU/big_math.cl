// 256-bit arithmetic for Curve25519/Ed25519 - 64-bit limbs (4 x ulong)
//
// BYTE ORDER: LITTLE-ENDIAN
//   limb[0] = bits 0..63 (bytes 0..7)
//   limb[3] = bits 192..255 (bytes 24..31)

// -- Basic operations -----------------------------------------

/* Add two 256-bit values limb by limb with carry propagation. r = a + b (mod 2^256). */
__inline void add_256(__generic const ulong* a, __generic const ulong* b, __generic ulong* r) {
    ulong carry = 0;
    for (int i = 0; i < 4; i++) {
        ulong t = a[i] + carry;
        carry = (t < a[i]) ? 1 : 0;
        ulong s = t + b[i];
        carry += (s < t) ? 1 : 0;
        r[i] = s;
    }
}

/* Subtract two 256-bit values limb by limb with borrow propagation. r = a - b (mod 2^256). */
__inline void sub_256(__generic const ulong* a, __generic const ulong* b, __generic ulong* r) {
    ulong borrow = 0;
    for (int i = 0; i < 4; i++) {
        ulong t = a[i] - borrow;
        borrow = (t > a[i]) ? 1 : 0;
        ulong d = t - b[i];
        borrow += (d > t) ? 1 : 0;
        r[i] = d;
    }
}

/* Copy 256-bit value (4 ulong limbs) from src to dst. */
__inline void copy_256(__generic const ulong* src, __generic ulong* dst) {
    for (int i = 0; i < 4; i++) dst[i] = src[i];
}

/* Set 256-bit value to 1. */
__inline void one_256(ulong* x) {
    x[0] = 1; x[1] = 0; x[2] = 0; x[3] = 0;
}

/* Set 256-bit value to 0. */
__inline void zero_256(ulong* x) {
    x[0] = 0; x[1] = 0; x[2] = 0; x[3] = 0;
}

// -- Modular reduction ----------------------------------------
// p = 2^255 - 19
// p as 4x64-bit LE limbs: [0xFFFFFFFFFFFFFFED, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF]

/* Reduce a 256-bit value x modulo p = 2^255 - 19 (branchless).
   Input x in [0, 2^256). After fold: x in [0, 2^255 + 380). After sub: x in [0, p-1].
   Uses 2 fold passes (bit 255 -> +19) and branchless conditional subtraction. */
__inline void mod_p_reduce(ulong* x) {
    // p = 2^255 - 19
    // Branchless: unroll 2 fold passes, conditional subtraction via mask select.
    // Input x in [0, 2^256). After fold: x in [0, 2^255 + 380). After sub: x in [0, p-1].

    // Fold pass 1: fold bit 255
    {
        ulong top = x[3] >> 63;
        x[3] &= 0x7FFFFFFFFFFFFFFFULL;
        ulong add19 = top * 19ULL;
        ulong s = x[0] + add19;
        ulong c = (s < x[0]) ? 1 : 0;
        x[0] = s;
        s = x[1] + c; c = (s < x[1]) ? 1 : 0; x[1] = s;
        s = x[2] + c; c = (s < x[2]) ? 1 : 0; x[2] = s;
        s = x[3] + c; x[3] = s;
    }

    // Fold pass 2: fold any remaining bit 255 (from carry)
    {
        ulong top = x[3] >> 63;
        x[3] &= 0x7FFFFFFFFFFFFFFFULL;
        ulong add19 = top * 19ULL;
        ulong s = x[0] + add19;
        ulong c = (s < x[0]) ? 1 : 0;
        x[0] = s;
        s = x[1] + c; c = (s < x[1]) ? 1 : 0; x[1] = s;
        s = x[2] + c; c = (s < x[2]) ? 1 : 0; x[2] = s;
        s = x[3] + c; x[3] = s;
    }

    // Branchless conditional subtraction: if x >= p, x -= p
    // Unrolled borrow chain (matching sub_256 logic) to avoid __generic overhead.
    // p limbs: [0xFFFFFFFFFFFFFFED, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF]
    {
        ulong borrow = 0;
        // limb 0
        ulong t0 = x[0] - borrow;
        borrow = (t0 > x[0]) ? 1 : 0;
        ulong d0 = t0 - 0xFFFFFFFFFFFFFFEDULL;
        borrow += (d0 > t0) ? 1 : 0;
        // limb 1
        ulong t1 = x[1] - borrow;
        borrow = (t1 > x[1]) ? 1 : 0;
        ulong d1 = t1 - 0xFFFFFFFFFFFFFFFFULL;
        borrow += (d1 > t1) ? 1 : 0;
        // limb 2
        ulong t2 = x[2] - borrow;
        borrow = (t2 > x[2]) ? 1 : 0;
        ulong d2 = t2 - 0xFFFFFFFFFFFFFFFFULL;
        borrow += (d2 > t2) ? 1 : 0;
        // limb 3
        ulong t3 = x[3] - borrow;
        borrow = (t3 > x[3]) ? 1 : 0;
        ulong d3 = t3 - 0x7FFFFFFFFFFFFFFFULL;
        borrow += (d3 > t3) ? 1 : 0;
        // borrow = 1 means x < p, borrow = 0 means x >= p
        ulong mask = 0 - borrow;
        x[0] = (mask & x[0]) | (~mask & d0);
        x[1] = (mask & x[1]) | (~mask & d1);
        x[2] = (mask & x[2]) | (~mask & d2);
        x[3] = (mask & x[3]) | (~mask & d3);
    }
}

// -- Modular multiplication: r = (a * b) mod p ----------------
// Strategy: expand to 8x32-bit, multiply with 32-bit internal accumulation,
// pack back to 4x64-bit, fold, reduce. Reuses the proven 32-bit algorithm.

/* Multiply two 256-bit values modulo p = 2^255 - 19. r = (a * b) mod p.
   Expands to 8x32-bit limbs, does 8x8 multiplication with carry, folds high bits * 19,
   packs back to 4x64-bit, then calls mod_p_reduce for final reduction. */
__inline void mul_mod_p(__generic const ulong* a, __generic const ulong* b, __generic ulong* r) {
    // Expand 4x64-bit -> 8x32-bit
    uint a32[8], b32[8];
    for (int i = 0; i < 4; i++) {
        a32[2*i]   = (uint)(a[i] & 0xFFFFFFFF);
        a32[2*i+1] = (uint)(a[i] >> 32);
        b32[2*i]   = (uint)(b[i] & 0xFFFFFFFF);
        b32[2*i+1] = (uint)(b[i] >> 32);
    }

    // 8x8 -> 16-limb accumulator (32-bit each)
    ulong t32[16] = {0};
    for (int i = 0; i < 8; i++) {
        ulong ci = 0;
        for (int j = 0; j < 8; j++) {
            ulong prod = (ulong)a32[i] * (ulong)b32[j];
            int k = i + j;
            ulong s = t32[k] + (prod & 0xFFFFFFFF) + ci;
            t32[k] = s & 0xFFFFFFFF;
            ci = (s >> 32) + (prod >> 32);
        }
        int k = i + 8;
        while (ci > 0 && k < 16) {
            ulong s = t32[k] + ci;
            t32[k] = s & 0xFFFFFFFF;
            ci = s >> 32;
            k++;
        }
    }

    // Fold hi*19 where hi = product >> 255
    ulong hi[8] = {0};
    hi[0] = (t32[7] >> 31) | ((t32[8] & 0x7FFFFFFF) << 1);
    for (int k = 1; k < 7; k++)
        hi[k] = (t32[8+k-1] >> 31) | ((t32[8+k] & 0x7FFFFFFF) << 1);
    hi[7] = (t32[14] >> 31) | ((t32[15] & 0x7FFFFFFF) << 1);

    // hi * 19
    ulong h19[9] = {0};
    for (int k = 0; k < 8; k++) {
        ulong prod = hi[k] * 19ULL;
        ulong lo = prod & 0xFFFFFFFF;
        ulong s = h19[k] + lo;
        ulong c = s >> 32;
        h19[k] = s & 0xFFFFFFFF;
        s = h19[k+1] + (prod >> 32) + c;
        h19[k+1] = s & 0xFFFFFFFF;
        h19[k+2] += s >> 32;
    }

    // lo + h19
    ulong res[9] = {0};
    ulong carry = 0;
    for (int k = 0; k < 9; k++) {
        ulong lo_val = (k < 7) ? t32[k] : ((k == 7) ? (t32[7] & 0x7FFFFFFF) : 0);
        ulong s = lo_val + h19[k] + carry;
        res[k] = s & 0xFFFFFFFF;
        carry = s >> 32;
    }

    // Fold bits 255+ (branchless)
    ulong fold = (res[7] >> 31) | ((res[8] & 0x7FFFFFFF) << 1);
    res[7] &= 0x7FFFFFFF;
    ulong add19 = fold * 19ULL;
    ulong s0 = res[0] + add19;
    res[0] = s0 & 0xFFFFFFFF;
    ulong c = s0 >> 32;
    ulong s1 = res[1] + (add19 >> 32) + c;
    res[1] = s1 & 0xFFFFFFFF;
    c = s1 >> 32;
    ulong s2 = res[2] + c;
    res[2] = s2 & 0xFFFFFFFF;
    c = s2 >> 32;
    res[3] = (res[3] + c) & 0xFFFFFFFF;

    // Pack 8x32-bit -> 4x64-bit
    for (int k = 0; k < 4; k++)
        r[k] = res[2*k] | (res[2*k+1] << 32);

    mod_p_reduce(r);
}

// -- Modular inverse (Montgomery Ladder with precomputed table) ---
// a^(-1) mod p = a^(p-2) mod p, p = 2^255 - 19
//
// Precomputed table: T[k*4..k*4+3] = a^(2^k) for k=0..254 (flat array, 1020 bytes)
// a^(p-2) = Prod_{k in bits(p-2)} T[k]
// p-2 = 2^255 - 21 -> bits 254..5=1, bit4=0, bit3=1, bit2=0, bit1=1, bit0=1
// Result = T[254] * T[253] * ... * T[5] * T[3] * T[1] * T[0]  (skip T[4], T[2])
//
// Precomputation (done ONCE per key in vanity_sshgen.cl):
//   T[0] = a (input value)
//   T[k] = T[k-1]^2 for k=1..254 (255 squarings total)
//   Then mod_p_inverse multiplies T[k] only where bit k of (p-2) is 1.
//
// All inverses in the same invocation share the same table.

/* Compute modular inverse: result = a^(-1) mod p = a^(p-2) mod p.
   Takes precomputed squaring chain T[1020] (T[k*4..k*4+3] = a^(2^k)).
   Multiplies T[k] where bit k of (p-2) is set. Branchless, constant-time. */
__inline void mod_p_inverse(__generic const ulong* T, __generic ulong* result) {
    // a^(p-2) where p-2 = 2^255 - 21
    // Bit pattern of p-2: bits 254..5=1, bit4=0, bit3=1, bit2=0, bit1=1, bit0=1
    // a^(p-2) = T[254]*T[253]*...*T[5]*T[3]*T[1]*T[0]
    one_256(result);
    ulong tmp[4];
    // bits 254..5 (all 1s)
    for (int k = 254; k >= 5; k--) {
        mul_mod_p(result, T + k*4, tmp);
        copy_256(tmp, result);
    }
    // bit 3
    mul_mod_p(result, T + 3*4, tmp);
    copy_256(tmp, result);
    // bit 1
    mul_mod_p(result, T + 1*4, tmp);
    copy_256(tmp, result);
    // bit 0
    mul_mod_p(result, T, tmp);
    copy_256(tmp, result);
    mod_p_reduce(result);
}

// -- Byte conversion ------------------------------------------

/* Convert 32-byte LE seed to 4x64-bit scalar (little-endian). */
__inline void seed_to_scalar(__generic const uchar* seed, __generic ulong* scalar) {
    for (int i = 0; i < 4; i++) {
        scalar[i] = ((ulong)seed[i*8]   ) |
                    (((ulong)seed[i*8+1]) <<  8) |
                    (((ulong)seed[i*8+2]) << 16) |
                    (((ulong)seed[i*8+3]) << 24) |
                    (((ulong)seed[i*8+4]) << 32) |
                    (((ulong)seed[i*8+5]) << 40) |
                    (((ulong)seed[i*8+6]) << 48) |
                    (((ulong)seed[i*8+7]) << 56);
    }
}

/* Convert 4x64-bit scalar to 32-byte LE array (little-endian). */
__inline void scalar_to_bytes(__generic const ulong* scalar, __generic uchar* bytes) {
    for (int i = 0; i < 4; i++) {
        bytes[i*8+0] = (uchar)((scalar[i] >>  0) & 0xFF);
        bytes[i*8+1] = (uchar)((scalar[i] >>  8) & 0xFF);
        bytes[i*8+2] = (uchar)((scalar[i] >> 16) & 0xFF);
        bytes[i*8+3] = (uchar)((scalar[i] >> 24) & 0xFF);
        bytes[i*8+4] = (uchar)((scalar[i] >> 32) & 0xFF);
        bytes[i*8+5] = (uchar)((scalar[i] >> 40) & 0xFF);
        bytes[i*8+6] = (uchar)((scalar[i] >> 48) & 0xFF);
        bytes[i*8+7] = (uchar)((scalar[i] >> 56) & 0xFF);
    }
}
