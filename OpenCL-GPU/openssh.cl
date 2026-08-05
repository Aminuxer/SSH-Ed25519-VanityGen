// OpenSSH Ed25519 public key generation - OpenCL kernel
// TIMESTAMP 1785623229
//
// Dependencies (provided by test kernel wrapper via #include):
//   sha512.cl  -> sha512_transform, SHA512_H, little_s0, little_s1
//   big_math.cl -> scalar_to_bytes, seed_to_scalar, add_256, etc.
//   ed25519.cl  -> scalar_mult, point_to_affine_y, point_init_base, D, BASE_X/Y
//
// THIS FILE MUST NOT #include other .cl files.
// The test wrapper (openssh_test_kernels.cl) includes dependencies with
// function renaming to avoid conflicts.
//
// Architecture:
//   seed_to_ssh_ed25519_pubkey is a single __kernel that:
//     1. Computes SHA512(seed) inline (sha512_transform from sha512.cl)
//     2. Clamps first 32 bytes ?
//     3. scalar_mult -> projective point (ed25519.cl)
//     4. point_to_affine_y -> 32-byte public key
//     5. build SSH public blob + base64 -> public key line
//
// Output format (OpenSSH public key line):
//   "ssh-ed25519 <base64-blob> <comment>"
//   where base64-blob = base64( uint32(11) || "ssh-ed25519" || uint32(32) || pubkey(32) )

// --- Base64 ---------------------------------------------

__constant uchar BASE64_TABLE[64] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',
    '0','1','2','3','4','5','6','7','8','9','+','/'
};

// Kernel: base64_encode
// Input:  [uint32 LE length (4 bytes)][data bytes]
// Output: [base64 string + '=' padding + null terminator]
// Max output: ceil(128/3)*4 + 1 = 173 bytes (for up to 128 input bytes)
__kernel void base64_encode(
    __global const uchar* inputWithLen,  // first 4 bytes = inLen (LE uint32), then data
    __global uchar* output                // output buffer
) {
    // Read input length
    uint inLen = ((uint)inputWithLen[0] |
                  ((uint)inputWithLen[1] << 8) |
                  ((uint)inputWithLen[2] << 16) |
                  ((uint)inputWithLen[3] << 24));
    const uchar* data = inputWithLen + 4;

    int outPos = 0;

    // Process full 3-byte groups
    int fullGroups = inLen / 3;
    for (int i = 0; i < fullGroups; i++) {
        uint b0 = (uint)data[i * 3];
        uint b1 = (uint)data[i * 3 + 1];
        uint b2 = (uint)data[i * 3 + 2];
        uint triplet = (b0 << 16) | (b1 << 8) | b2;

        output[outPos + 0] = BASE64_TABLE[(triplet >> 18) & 0x3F];
        output[outPos + 1] = BASE64_TABLE[(triplet >> 12) & 0x3F];
        output[outPos + 2] = BASE64_TABLE[(triplet >>  6) & 0x3F];
        output[outPos + 3] = BASE64_TABLE[ triplet        & 0x3F];
        outPos += 4;
    }

    // Handle remaining bytes
    int remainder = inLen - fullGroups * 3;
    if (remainder > 0) {
        uint lastTriplet = 0;
        if (remainder >= 1) lastTriplet |= ((uint)data[fullGroups * 3]) << 16;
        if (remainder >= 2) lastTriplet |= ((uint)data[fullGroups * 3 + 1]) << 8;

        output[outPos + 0] = BASE64_TABLE[(lastTriplet >> 18) & 0x3F];
        output[outPos + 1] = BASE64_TABLE[(lastTriplet >> 12) & 0x3F];

        if (remainder >= 2) {
            output[outPos + 2] = BASE64_TABLE[(lastTriplet >> 6) & 0x3F];
        } else {
            output[outPos + 2] = '=';
        }
        output[outPos + 3] = '=';
        outPos += 4;
    }

    // Null-terminate
    output[outPos] = 0;
}

// --- Inline helpers -------------------------------------

/* Build SSH public key binary blob (SSH wire format = BIG-ENDIAN):
   uint32_BE(11) || "ssh-ed25519" || uint32_BE(32) || pubKey(32). Total: 51 bytes. */
__inline void build_ssh_public_blob(__generic const uchar* pubKey, __generic uchar* blob) {
    // uint32_BE(11) = type string length
    blob[0] = 0x00; blob[1] = 0x00; blob[2] = 0x00; blob[3] = 0x0B;
    // "ssh-ed25519" (11 bytes)
    blob[4] = 's'; blob[5] = 's'; blob[6] = 'h';
    blob[7] = '-'; blob[8] = 'e'; blob[9] = 'd';
    blob[10] = '2'; blob[11] = '5'; blob[12] = '5';
    blob[13] = '1'; blob[14] = '9';
    // uint32_BE(32) = key data length
    blob[15] = 0x00; blob[16] = 0x00; blob[17] = 0x00; blob[18] = 0x20;
    // Copy 32 bytes of public key
    for (int i = 0; i < 32; i++) {
        blob[19 + i] = pubKey[i];
    }
}

// --- Main pubkey kernel ---------------------------------

/* Full pipeline: seed -> SHA512 -> clamp -> scalar_mult -> affine Y -> SSH public key line.
   Precomputes Z^(2^k) squaring chain T for fast modular inverse.
   Outputs: 32-byte publicKey and full "ssh-ed25519 <b64> <comment>" line. */
__kernel void seed_to_ssh_ed25519_pubkey(
    __global const uchar* seed,           // 32-byte seed (LE)
    __global const uchar* commentBuf,     // [4 bytes LE length][comment bytes (max 64)]
    __global uchar* publicKey,            // 32 bytes output
    __global uchar* pubLine               // 256 bytes output: "ssh-ed25519 <b64> <comment>\0"
) {
    // ======================================================================
    // 1. SHA512(seed) -> expanded[64]
    //    seed = 32 bytes = single SHA512 block (needs padding to 128 bytes)
    // ======================================================================
    unsigned long state[8];
    state[0] = SHA512_H[0]; state[1] = SHA512_H[1]; state[2] = SHA512_H[2]; state[3] = SHA512_H[3];
    state[4] = SHA512_H[4]; state[5] = SHA512_H[5]; state[6] = SHA512_H[6]; state[7] = SHA512_H[7];

    unsigned long W[80];
    // Copy seed (32 bytes) as first 4 x 64-bit words (BE)
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
    // Pad: 0x80 after 32 bytes, then zeros, then bit-length (256 bits = 32*8)
    W[4] = 0x8000000000000000UL;
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 256UL;  // bit length = 32 * 8 = 256

    // Extend W[16..79]
    for (int i = 16; i < 80; i++) {
        W[i] = W[i-16] + little_s0(W[i-15]) + W[i-7] + little_s1(W[i-2]);
    }
    sha512_transform(state, W);

    // Write expanded[64] (BE output)
    uchar expanded[64];
    for (int i = 0; i < 8; i++) {
        unsigned long val = state[i];
        expanded[i*8+0] = (uchar)(val >> 56);
        expanded[i*8+1] = (uchar)(val >> 48);
        expanded[i*8+2] = (uchar)(val >> 40);
        expanded[i*8+3] = (uchar)(val >> 32);
        expanded[i*8+4] = (uchar)(val >> 24);
        expanded[i*8+5] = (uchar)(val >> 16);
        expanded[i*8+6] = (uchar)(val >> 8);
        expanded[i*8+7] = (uchar)val;
    }

    // ======================================================================
    // 2. Scalar derivation: SHA512 output -> clamped scalar (LE interpretation)
    //
    //    expanded[0..31] = SHA512 output bytes.
    //    cryptography.hazmat interprets this as LITTLE-ENDIAN integer:
    //      k = expanded[0]*256^0 + ... + expanded[31]*256^31
    //    So scalar_bytes[i] = expanded[i] (no reversal needed)
    //
    //    Clamping (NaCl/cryptography):
    //      expanded[0] &= 0xF8  (clear bottom 3 bits of LSB byte)
    //      expanded[31] &= 0x7F | 0x40  (clear bit 7, set bit 6 of MSB byte)
    uchar scalar_bytes[32];
    for (int i = 0; i < 32; i++) {
        scalar_bytes[i] = expanded[i];
    }
    scalar_bytes[0] &= 0xF8;     // LSB: clear bottom 3 bits
    scalar_bytes[31] &= 0x7F;   // MSB: clear top bit
    scalar_bytes[31] |= 0x40;   // MSB: set top-1 bit

    // ======================================================================
    // 3. scalar_mult -> projective point (X, Y, Z)
    // ======================================================================
    ulong rX[4], rY[8], rZ[8];
    scalar_mult(scalar_bytes, rX, rY, rZ);

    // ======================================================================
    // 4. Precompute Z^(2^k) table for Montgomery inverse
    // ======================================================================
    ulong T[1020];
    mod_p_reduce(rZ);
    copy_256(rZ, T);
    for (int k = 1; k < 255; k++)
        mul_mod_p(T+(k-1)*4, T+(k-1)*4, T+k*4);

    // ======================================================================
    // 5. point_to_affine_y -> publicKey (32 bytes LE)
    //    Set high bit of publicKey[31] based on sign of affine X (bit 0)
    // ======================================================================
    uchar pubkey_affine[32];
    point_to_affine_y(T, rX, rY, rZ, pubkey_affine);
    // Compute affine X = rX / rZ mod p for sign bit
    ulong rZi[4], rXi[8];
    mod_p_inverse(T, rZi);
    mul_mod_p(rX, rZi, rXi);
    // Set sign bit (bit 7 of last byte) based on parity of affine X
    if ((rXi[0] & 1u) != 0) {
        pubkey_affine[31] |= 0x80;
    }
    // Write to output
    for (int i = 0; i < 32; i++) {
        publicKey[i] = pubkey_affine[i];
    }

    // ======================================================================
    // 5. Build SSH public key line: "ssh-ed25519 <base64> <comment>"
    //    BASE64 STRING IS GENERATED INSIDE THIS GPU KERNEL
    // ======================================================================
    // 5a. build_ssh_public_blob
    uchar ssh_blob[51];
    build_ssh_public_blob(pubkey_affine, ssh_blob);

    // 5b. Base64 encode (inline - can't call another __kernel)
    uchar b64buf[88];
    int b64len = 0;
    {
        int inLen = 51;
        int fullGroups = inLen / 3;  // 17
        int remainder = inLen - fullGroups * 3;  // 0
        for (int i = 0; i < fullGroups; i++) {
            uint b0 = (uint)ssh_blob[i * 3];
            uint b1 = (uint)ssh_blob[i * 3 + 1];
            uint b2 = (uint)ssh_blob[i * 3 + 2];
            uint triplet = (b0 << 16) | (b1 << 8) | b2;
            b64buf[b64len++] = BASE64_TABLE[(triplet >> 18) & 0x3F];
            b64buf[b64len++] = BASE64_TABLE[(triplet >> 12) & 0x3F];
            b64buf[b64len++] = BASE64_TABLE[(triplet >>  6) & 0x3F];
            b64buf[b64len++] = BASE64_TABLE[triplet         & 0x3F];
        }
        if (remainder > 0) {
            uint lastTriplet = 0;
            if (remainder >= 1) lastTriplet |= ((uint)ssh_blob[fullGroups*3]) << 16;
            if (remainder >= 2) lastTriplet |= ((uint)ssh_blob[fullGroups*3+1]) << 8;
            b64buf[b64len++] = BASE64_TABLE[(lastTriplet >> 18) & 0x3F];
            b64buf[b64len++] = BASE64_TABLE[(lastTriplet >> 12) & 0x3F];
            if (remainder >= 2) {
                b64buf[b64len++] = BASE64_TABLE[(lastTriplet >> 6) & 0x3F];
            } else {
                b64buf[b64len++] = '=';
            }
            b64buf[b64len++] = '=';
        }
        b64buf[b64len] = 0;
    }

    // 5c. pubLine = "ssh-ed25519 " + b64 + " " + comment
    int pos = 0;
    // Write "ssh-ed25519 " (12 bytes) manually
    pubLine[pos++] = 's'; pubLine[pos++] = 's'; pubLine[pos++] = 'h';
    pubLine[pos++] = '-'; pubLine[pos++] = 'e'; pubLine[pos++] = 'd';
    pubLine[pos++] = '2'; pubLine[pos++] = '5'; pubLine[pos++] = '5';
    pubLine[pos++] = '1'; pubLine[pos++] = '9'; pubLine[pos++] = ' ';
    for (int i = 0; i < b64len; i++) {
        pubLine[pos++] = b64buf[i];
    }

    // Copy comment
    uint commentLen = ((uint)commentBuf[0] |
                       ((uint)commentBuf[1] << 8) |
                       ((uint)commentBuf[2] << 16) |
                       ((uint)commentBuf[3] << 24));
    if (commentLen > 0) {
        pubLine[pos++] = ' ';
        int maxComment = 64;
        if (commentLen > maxComment) commentLen = maxComment;
        for (int i = 0; i < (int)commentLen; i++) {
            pubLine[pos++] = commentBuf[4 + i];
        }
    }
    pubLine[pos] = 0;
}
