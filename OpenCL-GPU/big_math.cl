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

/* Multiply two 256-bit values modulo p = 2^255 - 19. r = (a * b) mod p.
   Fully unrolled: 8x32-bit limbs in scalar registers (no indexable arrays),
   8x8 schoolbook multiplication with carry, hi*19 fold, mod_p_reduce.
   Bit-exact equivalent of the former array-based implementation; the
   scalar form avoids __private-array spills on CUDA-OpenCL (Pascal). */
__inline void mul_mod_p(__generic const ulong* a, __generic const ulong* b, __generic ulong* r) {
    uint a0 = (uint)a[0];  uint a1 = (uint)(a[0] >> 32);
    uint a2 = (uint)a[1];  uint a3 = (uint)(a[1] >> 32);
    uint a4 = (uint)a[2];  uint a5 = (uint)(a[2] >> 32);
    uint a6 = (uint)a[3];  uint a7 = (uint)(a[3] >> 32);
    uint b0 = (uint)b[0];  uint b1 = (uint)(b[0] >> 32);
    uint b2 = (uint)b[1];  uint b3 = (uint)(b[1] >> 32);
    uint b4 = (uint)b[2];  uint b5 = (uint)(b[2] >> 32);
    uint b6 = (uint)b[3];  uint b7 = (uint)(b[3] >> 32);

    ulong t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
    ulong t8 = 0, t9 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0, t14 = 0, t15 = 0;
    ulong s;

    {
        ulong ci = 0;
        ulong p;
        p = (ulong)a0 * (ulong)b0;
        s = t0 + (uint)p + ci; t0 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a0 * (ulong)b1;
        s = t1 + (uint)p + ci; t1 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a0 * (ulong)b2;
        s = t2 + (uint)p + ci; t2 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a0 * (ulong)b3;
        s = t3 + (uint)p + ci; t3 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a0 * (ulong)b4;
        s = t4 + (uint)p + ci; t4 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a0 * (ulong)b5;
        s = t5 + (uint)p + ci; t5 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a0 * (ulong)b6;
        s = t6 + (uint)p + ci; t6 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a0 * (ulong)b7;
        s = t7 + (uint)p + ci; t7 = (uint)s; ci = (s >> 32) + (p >> 32);
    if (ci != 0) {
        s = t8 + ci; t8 = (uint)s; ci = s >> 32;
        if (ci != 0) {
            s = t9 + ci; t9 = (uint)s; ci = s >> 32;
            if (ci != 0) {
                s = t10 + ci; t10 = (uint)s; ci = s >> 32;
            }
        }
    }
    }
    {
        ulong ci = 0;
        ulong p;
        p = (ulong)a1 * (ulong)b0;
        s = t1 + (uint)p + ci; t1 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a1 * (ulong)b1;
        s = t2 + (uint)p + ci; t2 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a1 * (ulong)b2;
        s = t3 + (uint)p + ci; t3 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a1 * (ulong)b3;
        s = t4 + (uint)p + ci; t4 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a1 * (ulong)b4;
        s = t5 + (uint)p + ci; t5 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a1 * (ulong)b5;
        s = t6 + (uint)p + ci; t6 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a1 * (ulong)b6;
        s = t7 + (uint)p + ci; t7 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a1 * (ulong)b7;
        s = t8 + (uint)p + ci; t8 = (uint)s; ci = (s >> 32) + (p >> 32);
    if (ci != 0) {
        s = t9 + ci; t9 = (uint)s; ci = s >> 32;
        if (ci != 0) {
            s = t10 + ci; t10 = (uint)s; ci = s >> 32;
            if (ci != 0) {
                s = t11 + ci; t11 = (uint)s; ci = s >> 32;
            }
        }
    }
    }
    {
        ulong ci = 0;
        ulong p;
        p = (ulong)a2 * (ulong)b0;
        s = t2 + (uint)p + ci; t2 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a2 * (ulong)b1;
        s = t3 + (uint)p + ci; t3 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a2 * (ulong)b2;
        s = t4 + (uint)p + ci; t4 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a2 * (ulong)b3;
        s = t5 + (uint)p + ci; t5 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a2 * (ulong)b4;
        s = t6 + (uint)p + ci; t6 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a2 * (ulong)b5;
        s = t7 + (uint)p + ci; t7 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a2 * (ulong)b6;
        s = t8 + (uint)p + ci; t8 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a2 * (ulong)b7;
        s = t9 + (uint)p + ci; t9 = (uint)s; ci = (s >> 32) + (p >> 32);
    if (ci != 0) {
        s = t10 + ci; t10 = (uint)s; ci = s >> 32;
        if (ci != 0) {
            s = t11 + ci; t11 = (uint)s; ci = s >> 32;
            if (ci != 0) {
                s = t12 + ci; t12 = (uint)s; ci = s >> 32;
            }
        }
    }
    }
    {
        ulong ci = 0;
        ulong p;
        p = (ulong)a3 * (ulong)b0;
        s = t3 + (uint)p + ci; t3 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a3 * (ulong)b1;
        s = t4 + (uint)p + ci; t4 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a3 * (ulong)b2;
        s = t5 + (uint)p + ci; t5 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a3 * (ulong)b3;
        s = t6 + (uint)p + ci; t6 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a3 * (ulong)b4;
        s = t7 + (uint)p + ci; t7 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a3 * (ulong)b5;
        s = t8 + (uint)p + ci; t8 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a3 * (ulong)b6;
        s = t9 + (uint)p + ci; t9 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a3 * (ulong)b7;
        s = t10 + (uint)p + ci; t10 = (uint)s; ci = (s >> 32) + (p >> 32);
    if (ci != 0) {
        s = t11 + ci; t11 = (uint)s; ci = s >> 32;
        if (ci != 0) {
            s = t12 + ci; t12 = (uint)s; ci = s >> 32;
            if (ci != 0) {
                s = t13 + ci; t13 = (uint)s; ci = s >> 32;
            }
        }
    }
    }
    {
        ulong ci = 0;
        ulong p;
        p = (ulong)a4 * (ulong)b0;
        s = t4 + (uint)p + ci; t4 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a4 * (ulong)b1;
        s = t5 + (uint)p + ci; t5 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a4 * (ulong)b2;
        s = t6 + (uint)p + ci; t6 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a4 * (ulong)b3;
        s = t7 + (uint)p + ci; t7 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a4 * (ulong)b4;
        s = t8 + (uint)p + ci; t8 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a4 * (ulong)b5;
        s = t9 + (uint)p + ci; t9 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a4 * (ulong)b6;
        s = t10 + (uint)p + ci; t10 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a4 * (ulong)b7;
        s = t11 + (uint)p + ci; t11 = (uint)s; ci = (s >> 32) + (p >> 32);
    if (ci != 0) {
        s = t12 + ci; t12 = (uint)s; ci = s >> 32;
        if (ci != 0) {
            s = t13 + ci; t13 = (uint)s; ci = s >> 32;
            if (ci != 0) {
                s = t14 + ci; t14 = (uint)s; ci = s >> 32;
            }
        }
    }
    }
    {
        ulong ci = 0;
        ulong p;
        p = (ulong)a5 * (ulong)b0;
        s = t5 + (uint)p + ci; t5 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a5 * (ulong)b1;
        s = t6 + (uint)p + ci; t6 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a5 * (ulong)b2;
        s = t7 + (uint)p + ci; t7 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a5 * (ulong)b3;
        s = t8 + (uint)p + ci; t8 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a5 * (ulong)b4;
        s = t9 + (uint)p + ci; t9 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a5 * (ulong)b5;
        s = t10 + (uint)p + ci; t10 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a5 * (ulong)b6;
        s = t11 + (uint)p + ci; t11 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a5 * (ulong)b7;
        s = t12 + (uint)p + ci; t12 = (uint)s; ci = (s >> 32) + (p >> 32);
    if (ci != 0) {
        s = t13 + ci; t13 = (uint)s; ci = s >> 32;
        if (ci != 0) {
            s = t14 + ci; t14 = (uint)s; ci = s >> 32;
            if (ci != 0) {
                s = t15 + ci; t15 = (uint)s; ci = s >> 32;
            }
        }
    }
    }
    {
        ulong ci = 0;
        ulong p;
        p = (ulong)a6 * (ulong)b0;
        s = t6 + (uint)p + ci; t6 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a6 * (ulong)b1;
        s = t7 + (uint)p + ci; t7 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a6 * (ulong)b2;
        s = t8 + (uint)p + ci; t8 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a6 * (ulong)b3;
        s = t9 + (uint)p + ci; t9 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a6 * (ulong)b4;
        s = t10 + (uint)p + ci; t10 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a6 * (ulong)b5;
        s = t11 + (uint)p + ci; t11 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a6 * (ulong)b6;
        s = t12 + (uint)p + ci; t12 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a6 * (ulong)b7;
        s = t13 + (uint)p + ci; t13 = (uint)s; ci = (s >> 32) + (p >> 32);
    if (ci != 0) {
        s = t14 + ci; t14 = (uint)s; ci = s >> 32;
        if (ci != 0) {
            s = t15 + ci; t15 = (uint)s; ci = s >> 32;
        }
    }
    }
    {
        ulong ci = 0;
        ulong p;
        p = (ulong)a7 * (ulong)b0;
        s = t7 + (uint)p + ci; t7 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a7 * (ulong)b1;
        s = t8 + (uint)p + ci; t8 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a7 * (ulong)b2;
        s = t9 + (uint)p + ci; t9 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a7 * (ulong)b3;
        s = t10 + (uint)p + ci; t10 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a7 * (ulong)b4;
        s = t11 + (uint)p + ci; t11 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a7 * (ulong)b5;
        s = t12 + (uint)p + ci; t12 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a7 * (ulong)b6;
        s = t13 + (uint)p + ci; t13 = (uint)s; ci = (s >> 32) + (p >> 32);
        p = (ulong)a7 * (ulong)b7;
        s = t14 + (uint)p + ci; t14 = (uint)s; ci = (s >> 32) + (p >> 32);
    if (ci != 0) {
        s = t15 + ci; t15 = (uint)s; ci = s >> 32;
    }
    }

    // hi window = bits 255..316 of the 512-bit product (62-bit windows of 19-fold)
    ulong hi0 = (t7  >> 31) | ((t8  & 0x7FFFFFFF) << 1);
    ulong hi1 = (t8  >> 31) | ((t9  & 0x7FFFFFFF) << 1);
    ulong hi2 = (t9  >> 31) | ((t10 & 0x7FFFFFFF) << 1);
    ulong hi3 = (t10 >> 31) | ((t11 & 0x7FFFFFFF) << 1);
    ulong hi4 = (t11 >> 31) | ((t12 & 0x7FFFFFFF) << 1);
    ulong hi5 = (t12 >> 31) | ((t13 & 0x7FFFFFFF) << 1);
    ulong hi6 = (t13 >> 31) | ((t14 & 0x7FFFFFFF) << 1);
    ulong hi7 = (t14 >> 31) | ((t15 & 0x7FFFFFFF) << 1);

    // hi * 19 accumulated over acc0..acc8 (32-bit limbs, top may carry +2)
    ulong acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0, acc8 = 0;
    ulong prod, lo, c;
    prod = hi0 * 19ULL; lo = prod & 0xFFFFFFFF;
    s = acc0 + lo; c = s >> 32; acc0 = s & 0xFFFFFFFF;
    s = acc1 + (prod >> 32) + c; acc1 = s & 0xFFFFFFFF; acc2 += s >> 32;
    prod = hi1 * 19ULL; lo = prod & 0xFFFFFFFF;
    s = acc1 + lo; c = s >> 32; acc1 = s & 0xFFFFFFFF;
    s = acc2 + (prod >> 32) + c; acc2 = s & 0xFFFFFFFF; acc3 += s >> 32;
    prod = hi2 * 19ULL; lo = prod & 0xFFFFFFFF;
    s = acc2 + lo; c = s >> 32; acc2 = s & 0xFFFFFFFF;
    s = acc3 + (prod >> 32) + c; acc3 = s & 0xFFFFFFFF; acc4 += s >> 32;
    prod = hi3 * 19ULL; lo = prod & 0xFFFFFFFF;
    s = acc3 + lo; c = s >> 32; acc3 = s & 0xFFFFFFFF;
    s = acc4 + (prod >> 32) + c; acc4 = s & 0xFFFFFFFF; acc5 += s >> 32;
    prod = hi4 * 19ULL; lo = prod & 0xFFFFFFFF;
    s = acc4 + lo; c = s >> 32; acc4 = s & 0xFFFFFFFF;
    s = acc5 + (prod >> 32) + c; acc5 = s & 0xFFFFFFFF; acc6 += s >> 32;
    prod = hi5 * 19ULL; lo = prod & 0xFFFFFFFF;
    s = acc5 + lo; c = s >> 32; acc5 = s & 0xFFFFFFFF;
    s = acc6 + (prod >> 32) + c; acc6 = s & 0xFFFFFFFF; acc7 += s >> 32;
    prod = hi6 * 19ULL; lo = prod & 0xFFFFFFFF;
    s = acc6 + lo; c = s >> 32; acc6 = s & 0xFFFFFFFF;
    s = acc7 + (prod >> 32) + c; acc7 = s & 0xFFFFFFFF; acc8 += s >> 32;
    prod = hi7 * 19ULL; lo = prod & 0xFFFFFFFF;
    s = acc7 + lo; c = s >> 32; acc7 = s & 0xFFFFFFFF;
    s = acc8 + (prod >> 32) + c; acc8 = s & 0xFFFFFFFF;

    // low 256 bits (top bit of t7 folded with the hi window) + acc
    ulong carry = 0;
    s = t0 + acc0 + carry;        ulong res0 = s & 0xFFFFFFFF; carry = s >> 32;
    s = t1 + acc1 + carry;        ulong res1 = s & 0xFFFFFFFF; carry = s >> 32;
    s = t2 + acc2 + carry;        ulong res2 = s & 0xFFFFFFFF; carry = s >> 32;
    s = t3 + acc3 + carry;        ulong res3 = s & 0xFFFFFFFF; carry = s >> 32;
    s = t4 + acc4 + carry;        ulong res4 = s & 0xFFFFFFFF; carry = s >> 32;
    s = t5 + acc5 + carry;        ulong res5 = s & 0xFFFFFFFF; carry = s >> 32;
    s = t6 + acc6 + carry;        ulong res6 = s & 0xFFFFFFFF; carry = s >> 32;
    s = (t7 & 0x7FFFFFFF) + acc7 + carry; ulong res7 = s & 0xFFFFFFFF; carry = s >> 32;
    s = acc8 + carry;             ulong res8 = s & 0xFFFFFFFF;

    // Fold bits 255+ (branchless)
    ulong fold = (res7 >> 31) | ((res8 & 0x7FFFFFFF) << 1);
    res7 &= 0x7FFFFFFF;
    ulong add19 = fold * 19ULL;
    s = res0 + add19; res0 = s & 0xFFFFFFFF; c = s >> 32;
    s = res1 + (add19 >> 32) + c; res1 = s & 0xFFFFFFFF; c = s >> 32;
    s = res2 + c; res2 = s & 0xFFFFFFFF; c = s >> 32;
    res3 = (res3 + c) & 0xFFFFFFFF;

    // Pack 8x32-bit -> 4x64-bit
    r[0] = res0 | (res1 << 32);
    r[1] = res2 | (res3 << 32);
    r[2] = res4 | (res5 << 32);
    r[3] = res6 | (res7 << 32);

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
