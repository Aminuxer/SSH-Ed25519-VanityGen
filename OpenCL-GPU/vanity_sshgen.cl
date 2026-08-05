// vanity_sshgen.cl - GPU kernel for vanity SSH key search
// Includes the tested pipeline files:
//   sha512.cl   - SHA512_H, sha512_transform, little_s0, little_s1
//   big_math.cl - mod_p_reduce, mul_mod_p, mod_p_inverse, scalar_to_bytes
//   ed25519.cl  - scalar_mult
//   openssh.cl  - BASE64_TABLE (plus __kernel funcs we don't call here)
//
// SEED LIFECYCLE:
//   - CPU generates random seeds ONCE at startup, uploads to GPU (READ_WRITE)
//   - Each work-item reads its seed, uses it, then increments by 1 (256-bit LE)
//   - On NEXT kernel launch, the same work-item gets the already-incremented seed
//   - NO H2D seed transfers in the main loop - seeds live and mutate on GPU
//
// The kernel (per work-item):
//   1. Load seed from seeds[idx*32..(idx+1)*32]
//   2. SHA512(seed) -> expanded[64]
//   3. Clamp expanded[0..31] -> scalar (LE interpretation, RFC 8032)
//   4. scalar_mult -> projective point (Ed25519)
//   5. Affine Y + sign bit -> 32-byte public key
//   6. Build SSH blob + Base64 encode (51 -> 68 chars)
//   7. Match pattern in variable part (base64[pos 25+])
//   8. If matched: write seed to foundSeeds, pattern idx to results
//   9. Increment seed by 1 (256-bit LE with carry) back to seeds buffer
//
// Output:
//   results[i]     = pattern index (0-based) if matched, -1 otherwise
//   foundSeeds[i]  = the seed bytes that produced the match (only valid if results[i]>=0)

#include "./sha512.cl"
#include "./big_math.cl"
#include "./ed25519.cl"
#include "./openssh.cl"

/* Main kernel: for each work-item, generate a vanity SSH key.
   Pipeline: load seed -> SHA512 -> clamp -> scalar_mult -> inverse -> base64 -> pattern match.
   Seed is incremented on each launch so the same work-item continues from where it left off.
   Precomputed squaring chain T is built from rZ to speed up modular inverse. */
__kernel void vanity_search(
    __global uchar* seeds,           // READ_WRITE: 32-byte LE seeds, incremented each launch
    const int numSeeds,
    __global const uchar* patterns,
    __global const uchar* patternLens,
    __global const uchar* caseInsensFlags,
    const int numPatterns,
    __global int* results,
    __global uchar* pubKeyOut,
    __global uchar* foundSeeds        // write seed on match (only valid when results[idx]>=0)
) {
    int idx = (int)get_global_id(0);
    if (idx >= numSeeds) return;

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
    ulong rX[4], rY[8], rZ[8];
    scalar_mult(scalar, rX, rY, rZ);

    // -- 4. Precompute Z^(2^k) table for Montgomery inverse --------------
    // T[k*4..k*4+3] = rZ^(2^k) for k=0..254 (flat array, 1020 bytes)
    ulong T[1020];
    mod_p_reduce(rZ);
    copy_256(rZ, T);
    for (int k = 1; k < 255; k++)
        mul_mod_p(T+(k-1)*4, T+(k-1)*4, T+k*4);

    // -- 5. Affine Y + sign bit -> public key -----------------------------
    // Use precomputed table for mod_p_inverse
    ulong rZi[4], rYa[8], rXi[8];
    mod_p_inverse(T, rZi);
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
        // Write the MATCHING seed (BEFORE increment) so CPU can reproduce it
        for (int i = 0; i < 32; i++)
            foundSeeds[idx * 32 + i] = seed[i];
    }

    // -- 9. Increment seed by 1 (256-bit LE with carry) ----------------
    // Each work-item owns its seed and increments it independently.
    // No synchronization needed - work-items don't share seed indices.
    {
        unsigned carry = 1;
        for (int i = 0; i < 32; i++) {
            unsigned s = (unsigned)seed[i] + carry;
            seeds[idx * 32 + i] = (uchar)(s & 0xFF);
            if (s < 256) { carry = 0; break; }
        }
    }
}
