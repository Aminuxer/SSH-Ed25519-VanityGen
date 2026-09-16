// vanity_sshgen.cl - GPU kernel for vanity SSH key search (v. 2026-09-11)
// Includes the tested pipeline files:
//   sha512.cl   - SHA512_H, sha512_transform, little_s0, little_s1
//   big_math.cl - mod_p_reduce, mul_mod_p, copy_256, scalar_to_bytes
//   ed25519.cl  - scalar_mult
//   openssh.cl  - BASE64_TABLE
//
// SEED LIFECYCLE:
//   - CPU generates random seeds ONCE at startup, uploads to GPU (READ_WRITE)
//   - Each work-item reads its seed, uses it, then increments by 1 (256-bit LE)
//   - On NEXT kernel launch, the same work-item gets the already-incremented seed
//   - NO H2D seed transfers in the main loop - seeds live and mutate on GPU
//
// THE KERNEL (per work-item):
//   1. Load seed from seeds[idx*32..(idx+1)*32]
//   2. SHA512(seed) -> expanded[64]
//   3. Clamp expanded[0..31] -> scalar (LE interpretation, RFC 8032)
//   4. scalar_mult -> projective point (Ed25519)
//   5. BATCHED affine inverse: the 32 work-items of this workgroup invert
//      their rZ values together in shared memory (see below)
//   6. Affine Y + sign bit -> 32-byte public key
//   7. Build SSH blob + Base64 encode (51 -> 68 chars)
//   8. Match pattern in variable part (base64[pos 25+])
//   9. If matched: write [seed 32][pub 32] to foundSeeds, pattern idx to results
//  10. Increment seed by 1 (256-bit LE with carry) back to seeds buffer
//
// 2026-09-11 CHANGES:
//   The per-candidate modular inverse of rZ no longer uses the 8KB
//   per-work-item squaring chain T[1020] (507 field muls + large private
//   memory spill). All 32 work-items of a workgroup batch-invert their
//   Z values in __local memory:
//     a) publish rZ to __local Zlg[lid]
//     b) inclusive prefix scan  P[i] = Z[0]*...*Z[i]         (5 mul-steps)
//     c) Fermat T = P[31]^(p-2) with the exponent chain SPLIT across
//        the 32 lanes: lane j computes V_j = (a^(2^(8j)))^(E_j)
//        where a = P[31] and E = p-2 = sum_j E_j * 2^(8j):
//        E_0 = 0xEB, E_1..E_30 = 0xFF, E_31 = 0x7F
//        (lane j does 8j squarings; the warp cost is the max = 248)
//     d) inclusive product scan of V: T = V_0*...*V_31        (5 mul-steps)
//     e) inclusive suffix scan  Q[i] = Z[i]*...*Z[31]         (5 mul-steps)
//     f) Zinv[i] = T * P[i-1] * Q[i+1]   (P[-1] = Q[32] = 1)
//   Inverse cost: ~270 warp-mul steps per 32 candidates (~8.5/candidate)
//   vs 507 per candidate in V1, and no per-item 8KB private spill.
//   Everything else (signature, seed lifecycle, blob, base64, match)
//   is identical to V1. Found entry = [seed 32][pub 32] (64 bytes).
//
// Output (2026-09-16):
//   results[i]     = pattern index (0-based) if matched, -1 otherwise
//                    (GPU-side bookkeeping only; host does NOT read it)
//   matchCount     = atomic counter of matches in this launch (int).
//                    The host resets it to 0 before every launch.
//   foundSeeds     = match entries indexed by atomic slot, stride 68:
//                    [seed 32][pub 32][pat_idx 4]

#include "./sha512.cl"
#include "./big_math.cl"
#include "./ed25519.cl"
#include "./openssh.cl"

__kernel void vanity_search(
    __global uchar* seeds,           // READ_WRITE: 32-byte LE seeds, incremented each launch
    const int numSeeds,
    __global const uchar* patterns,
    __global const uchar* patternLens,
    __global const uchar* caseInsensFlags,
    const int numPatterns,
    __global int* results,
    __global uchar* pubKeyOut,
    __global uchar* foundSeeds,       // write [seed 32][pub 32][pat 4] on match (stride 68)
    __global int* matchCount          // atomic match counter, host resets to 0 per batch
) {
    int idx = (int)get_global_id(0);
    if (idx >= numSeeds) return;
    int lid = (int)get_local_id(0);  // 0..31 (workgroup size is 32)

    // -- Load seed (256-bit LE) -----------------------------------------
    uchar seed[32];
    for (int i = 0; i < 32; i++)
        seed[i] = seeds[idx * 32 + i];

    // -- 1. SHA512(seed) -> expanded -------------------------------------
    unsigned long state[8];
    for (int i = 0; i < 8; i++)
        state[i] = SHA512_H[i];

    unsigned long W[80];
    for (int i = 0; i < 4; i++) {
        W[i] = ((unsigned long)seed[i*8]   << 56) |
               ((unsigned long)seed[i*8+1] << 48) |
               ((unsigned long)seed[i*8+2] << 40) |
               ((unsigned long)seed[i*8+3] << 32) |
               ((unsigned long)seed[i*8+4] << 24) |
               ((unsigned long)seed[i*8+5] << 16) |
               ((unsigned long)seed[i*8+6] <<  8) |
               ((unsigned long)seed[i*8+7]);
    }
    W[4]  = 0x8000000000000000UL;
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 256UL;

    for (int i = 16; i < 80; i++)
        W[i] = W[i-16] + little_s0(W[i-15]) + W[i-7] + little_s1(W[i-2]);

    sha512_transform(state, W);

    uchar expanded[64];
    for (int i = 0; i < 8; i++) {
        unsigned long v = state[i];
        expanded[i*8+0] = (uchar)(v >> 56);
        expanded[i*8+1] = (uchar)(v >> 48);
        expanded[i*8+2] = (uchar)(v >> 40);
        expanded[i*8+3] = (uchar)(v >> 32);
        expanded[i*8+4] = (uchar)(v >> 24);
        expanded[i*8+5] = (uchar)(v >> 16);
        expanded[i*8+6] = (uchar)(v >>  8);
        expanded[i*8+7] = (uchar)v;
    }

    // -- 2. Clamp -------------------------------------------------------
    uchar scalar[32];
    for (int i = 0; i < 32; i++)
        scalar[i] = expanded[i];
    scalar[0]  &= 0xF8;
    scalar[31] &= 0x7F;
    scalar[31] |= 0x40;

    // -- 3. scalar_mult -> projective (X, Y, Z) --------------------------
    ulong rX[4], rY[4], rZ[4];
    scalar_mult(scalar, rX, rY, rZ);

    // -- 4. Batched affine inverse (workgroup of 32) ---------------------
    // CONSTRAINT (Pascal / CUDA-OpenCL driver on this host): passing a __local
    // pointer to a generic-pointer helper (copy_256 / mul_mod_p) crashes the
    // kernel at launch (clWaitForEvents -9999). All __local accesses below use
    // direct indexing; helper calls operate on __private arrays only.
    __local ulong Zlg[32][4];
    __local ulong Plg[32][4];
    __local ulong Vlg[32][4];
    __local ulong Qlg[32][4];

    mod_p_reduce(rZ);
    Zlg[lid][0] = rZ[0];
    Zlg[lid][1] = rZ[1];
    Zlg[lid][2] = rZ[2];
    Zlg[lid][3] = rZ[3];
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4a. Inclusive prefix: P[i] = Z[0]*...*Z[i]
    Plg[lid][0] = Zlg[lid][0];
    Plg[lid][1] = Zlg[lid][1];
    Plg[lid][2] = Zlg[lid][2];
    Plg[lid][3] = Zlg[lid][3];
    for (int s = 1; s < 32; s <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= s) {
            ulong A[4], B[4], R[4];
            A[0] = Plg[lid - s][0]; A[1] = Plg[lid - s][1];
            A[2] = Plg[lid - s][2]; A[3] = Plg[lid - s][3];
            B[0] = Plg[lid][0];     B[1] = Plg[lid][1];
            B[2] = Plg[lid][2];     B[3] = Plg[lid][3];
            mul_mod_p(A, B, R);
            Plg[lid][0] = R[0]; Plg[lid][1] = R[1];
            Plg[lid][2] = R[2]; Plg[lid][3] = R[3];
        }
    }

    // 4b. Fermat T = P[31]^(p-2), exponent chain split across the 32 lanes.
    {
        ulong s4[4], w4[4], tmp4[4];
        s4[0] = Plg[31][0]; s4[1] = Plg[31][1];
        s4[2] = Plg[31][2]; s4[3] = Plg[31][3];
        for (int k = 0; k < 8 * lid; k++) {
            mul_mod_p(s4, s4, tmp4);
            s4[0] = tmp4[0]; s4[1] = tmp4[1];
            s4[2] = tmp4[2]; s4[3] = tmp4[3];
        }
        uint Ej = (lid == 0) ? 0x000000EBu
                 : ((lid < 31) ? 0xFFFFFFFFu : 0x0000007Fu);
        one_256(w4);
        for (int b = 7; b >= 0; b--) {
            mul_mod_p(w4, w4, tmp4);
            w4[0] = tmp4[0]; w4[1] = tmp4[1];
            w4[2] = tmp4[2]; w4[3] = tmp4[3];
            if (((Ej >> b) & 1u) != 0u) {
                mul_mod_p(w4, s4, tmp4);
                w4[0] = tmp4[0]; w4[1] = tmp4[1];
                w4[2] = tmp4[2]; w4[3] = tmp4[3];
            }
        }
        Vlg[lid][0] = w4[0]; Vlg[lid][1] = w4[1];
        Vlg[lid][2] = w4[2]; Vlg[lid][3] = w4[3];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4c. Inclusive product of V: V[31] = T = P[31]^(p-2)
    for (int s = 1; s < 32; s <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= s) {
            ulong A[4], B[4], R[4];
            A[0] = Vlg[lid - s][0]; A[1] = Vlg[lid - s][1];
            A[2] = Vlg[lid - s][2]; A[3] = Vlg[lid - s][3];
            B[0] = Vlg[lid][0];     B[1] = Vlg[lid][1];
            B[2] = Vlg[lid][2];     B[3] = Vlg[lid][3];
            mul_mod_p(A, B, R);
            Vlg[lid][0] = R[0]; Vlg[lid][1] = R[1];
            Vlg[lid][2] = R[2]; Vlg[lid][3] = R[3];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4d. Inclusive suffix: Q[i] = Z[i]*...*Z[31]
    Qlg[lid][0] = Zlg[lid][0];
    Qlg[lid][1] = Zlg[lid][1];
    Qlg[lid][2] = Zlg[lid][2];
    Qlg[lid][3] = Zlg[lid][3];
    for (int s = 1; s < 32; s <<= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid + s <= 31) {
            ulong A[4], B[4], R[4];
            A[0] = Qlg[lid + s][0]; A[1] = Qlg[lid + s][1];
            A[2] = Qlg[lid + s][2]; A[3] = Qlg[lid + s][3];
            B[0] = Qlg[lid][0];     B[1] = Qlg[lid][1];
            B[2] = Qlg[lid][2];     B[3] = Qlg[lid][3];
            mul_mod_p(A, B, R);
            Qlg[lid][0] = R[0]; Qlg[lid][1] = R[1];
            Qlg[lid][2] = R[2]; Qlg[lid][3] = R[3];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4e. Zinv[lid] = T * P[lid-1] * Q[lid+1]   (P[-1] = Q[32] = 1)
    ulong rZi[4], T[4], Qp[4], Pm[4], tmp4b[4];
    T[0] = Vlg[31][0]; T[1] = Vlg[31][1];
    T[2] = Vlg[31][2]; T[3] = Vlg[31][3];
    if (lid == 0) {
        Qp[0] = Qlg[1][0]; Qp[1] = Qlg[1][1];
        Qp[2] = Qlg[1][2]; Qp[3] = Qlg[1][3];
        mul_mod_p(T, Qp, rZi);
    } else if (lid == 31) {
        Pm[0] = Plg[30][0]; Pm[1] = Plg[30][1];
        Pm[2] = Plg[30][2]; Pm[3] = Plg[30][3];
        mul_mod_p(T, Pm, rZi);
    } else {
        Qp[0] = Qlg[lid + 1][0]; Qp[1] = Qlg[lid + 1][1];
        Qp[2] = Qlg[lid + 1][2]; Qp[3] = Qlg[lid + 1][3];
        mul_mod_p(T, Qp, tmp4b);
        Pm[0] = Plg[lid - 1][0]; Pm[1] = Plg[lid - 1][1];
        Pm[2] = Plg[lid - 1][2]; Pm[3] = Plg[lid - 1][3];
        mul_mod_p(tmp4b, Pm, rZi);
    }

    // 4f. Affine Y + sign bit -> public key
    ulong rYa[4], rXi[4];
    mul_mod_p(rY, rZi, rYa);    // affine Y = rY / rZ mod p
    mul_mod_p(rX, rZi, rXi);    // affine X = rX / rZ mod p (for sign bit)

    uchar pubkey[32];
    scalar_to_bytes(rYa, pubkey);
    if ((rXi[0] & 1u) != 0)
        pubkey[31] |= 0x80;

    for (int i = 0; i < 32; i++)
        pubKeyOut[idx * 32 + i] = pubkey[i];

    // -- 5. Build SSH blob (51 bytes) -----------------------------------
    uchar blob[51];
    blob[0] = 0x00; blob[1] = 0x00; blob[2] = 0x00; blob[3] = 0x0B;
    blob[4] = 's';  blob[5] = 's';  blob[6] = 'h';  blob[7] = '-';
    blob[8] = 'e';  blob[9] = 'd'; blob[10] = '2'; blob[11] = '5';
    blob[12] = '5'; blob[13] = '1'; blob[14] = '9';
    blob[15] = 0x00; blob[16] = 0x00; blob[17] = 0x00; blob[18] = 0x20;
    for (int i = 0; i < 32; i++)
        blob[19 + i] = pubkey[i];

    // -- 6. Base64 encode (51 bytes -> 68 chars) -------------------------
    uchar b64[72];
    int bpos = 0;
    for (int i = 0; i < 17; i++) {
        uint b0 = (uint)blob[i*3];
        uint b1 = (uint)blob[i*3+1];
        uint b2 = (uint)blob[i*3+2];
        uint t  = (b0 << 16) | (b1 << 8) | b2;
        b64[bpos++] = BASE64_TABLE[(t >> 18) & 0x3F];
        b64[bpos++] = BASE64_TABLE[(t >> 12) & 0x3F];
        b64[bpos++] = BASE64_TABLE[(t >>  6) & 0x3F];
        b64[bpos++] = BASE64_TABLE[ t        & 0x3F];
    }

    // -- 7. Pattern matching (variable part, pos >= 25) -----------------
    int foundPat = -1;
    for (int p = 0; p < numPatterns && foundPat < 0; p++) {
        uchar plen = patternLens[p];
        uchar ci   = caseInsensFlags[p];
        const uchar* pat = patterns + p * 32;

        for (int pos = 25; pos <= 68 - (int)plen && foundPat < 0; pos++) {
            int match = 1;
            for (int j = 0; j < (int)plen; j++) {
                uchar bc = b64[pos + j];
                uchar pc = pat[j];
                if (ci) {
                    if (bc >= 'A' && bc <= 'Z') bc = bc - 'A' + 'a';
                    if (pc >= 'A' && pc <= 'Z') pc = pc - 'A' + 'a';
                }
                if (bc != pc) { match = 0; break; }
            }
            if (match) foundPat = p;
        }
    }

    // -- 8. Write results ----------------------------------------------
    results[idx] = foundPat;
    if (foundPat >= 0) {
        // 2026-09-16: GPU-side match bookkeeping. The kernel accounts the
        // match in the atomic counter; the host reads ONLY the counter
        // (4 bytes per batch) and, when non-zero, the entries below.
        // Entry layout: [seed 32][pub 32][pat_idx 4], stride 68, indexed
        // by the atomic slot. At most one entry per work-item per launch
        // and the host resets the counter to 0 before every launch, so
        // slot < numSeeds. The seed written is the MATCHING one
        // (BEFORE the increment below), so the CPU can rebuild the key
        // without recomputing anything.
        int slot = atomic_inc(matchCount);
        if (slot < numSeeds) {
            uchar* e = foundSeeds + slot * 68;
            for (int i = 0; i < 32; i++)
                e[i] = seed[i];
            for (int i = 0; i < 32; i++)
                e[32 + i] = pubkey[i];
            e[64] = (uchar)(foundPat & 0xFF);
            e[65] = (uchar)((foundPat >> 8) & 0xFF);
            e[66] = (uchar)((foundPat >> 16) & 0xFF);
            e[67] = (uchar)((foundPat >> 24) & 0xFF);
        }
    }

    // -- 9. Increment seed by 1 (256-bit LE with carry) ----------------
    {
        unsigned carry = 1;
        for (int i = 0; i < 32; i++) {
            unsigned s = (unsigned)seed[i] + carry;
            seeds[idx * 32 + i] = (uchar)(s & 0xFF);
            if (s < 256) { carry = 0; break; }
        }
    }
}
