/**
 * Full OpenCL SHA-512 Implementation
 * Supports arbitrary-length inputs with proper padding.
 */

#ifndef SHA512_CL
#define SHA512_CL

__constant unsigned long SHA512_K[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};

__constant unsigned long SHA512_H[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
};

#define choose(x, y, z) (bitselect(z, y, x))
#define bit_maj(x, y, z) (bitselect(x, y, ((x) ^ (z))))

__inline unsigned long rotr64(unsigned long x, unsigned int n) {
    return (x >> n) | (x << (64 - n));
}
#define S0(x) (rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39))
#define S1(x) (rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41))
#define little_s0(x) (rotr64(x, 1) ^ rotr64(x, 8) ^ ((x) >> 7))
#define little_s1(x) (rotr64(x, 19) ^ rotr64(x, 61) ^ ((x) >> 6))

#define SHA512_STEP(a, b, c, d, e, f, g, h, x, K) { h += K + S1(e) + choose(e, f, g) + x; d += h; h += S0(a) + bit_maj(a, b, c); }

__inline unsigned long load64_global(const __global uchar* p) {
    return ((unsigned long)p[0] << 56) |
           ((unsigned long)p[1] << 48) |
           ((unsigned long)p[2] << 40) |
           ((unsigned long)p[3] << 32) |
           ((unsigned long)p[4] << 24) |
           ((unsigned long)p[5] << 16) |
           ((unsigned long)p[6] << 8) |
           ((unsigned long)p[7]);
}

__inline unsigned long load64_constant(const __constant uchar* p) {
    return ((unsigned long)p[0] << 56) |
           ((unsigned long)p[1] << 48) |
           ((unsigned long)p[2] << 40) |
           ((unsigned long)p[3] << 32) |
           ((unsigned long)p[4] << 24) |
           ((unsigned long)p[5] << 16) |
           ((unsigned long)p[6] << 8) |
           ((unsigned long)p[7]);
}

__inline void sha512_transform(__private unsigned long* state, __private unsigned long* W) {
    unsigned long a = state[0], b = state[1], c = state[2], d = state[3];
    unsigned long e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 80; i += 16) {
        SHA512_STEP(a, b, c, d, e, f, g, h, W[i+0], SHA512_K[i+0]);
        SHA512_STEP(h, a, b, c, d, e, f, g, W[i+1], SHA512_K[i+1]);
        SHA512_STEP(g, h, a, b, c, d, e, f, W[i+2], SHA512_K[i+2]);
        SHA512_STEP(f, g, h, a, b, c, d, e, W[i+3], SHA512_K[i+3]);
        SHA512_STEP(e, f, g, h, a, b, c, d, W[i+4], SHA512_K[i+4]);
        SHA512_STEP(d, e, f, g, h, a, b, c, W[i+5], SHA512_K[i+5]);
        SHA512_STEP(c, d, e, f, g, h, a, b, W[i+6], SHA512_K[i+6]);
        SHA512_STEP(b, c, d, e, f, g, h, a, W[i+7], SHA512_K[i+7]);
        SHA512_STEP(a, b, c, d, e, f, g, h, W[i+8], SHA512_K[i+8]);
        SHA512_STEP(h, a, b, c, d, e, f, g, W[i+9], SHA512_K[i+9]);
        SHA512_STEP(g, h, a, b, c, d, e, f, W[i+10], SHA512_K[i+10]);
        SHA512_STEP(f, g, h, a, b, c, d, e, W[i+11], SHA512_K[i+11]);
        SHA512_STEP(e, f, g, h, a, b, c, d, W[i+12], SHA512_K[i+12]);
        SHA512_STEP(d, e, f, g, h, a, b, c, W[i+13], SHA512_K[i+13]);
        SHA512_STEP(c, d, e, f, g, h, a, b, W[i+14], SHA512_K[i+14]);
        SHA512_STEP(b, c, d, e, f, g, h, a, W[i+15], SHA512_K[i+15]);
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__kernel void sha512(__constant uchar* input, __global uchar* output, __global ulong* length_buf) {
    ulong msg_len = *length_buf;
    
    // Handle empty string case - SHA-512 of "" is known
    if (msg_len == 0) {
        __private unsigned long empty_hash[8] = {
            0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL,
            0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL, 0x0000000000000000UL
        };
        __private unsigned long state[8];
        state[0] = SHA512_H[0]; state[1] = SHA512_H[1]; state[2] = SHA512_H[2]; state[3] = SHA512_H[3];
        state[4] = SHA512_H[4]; state[5] = SHA512_H[5]; state[6] = SHA512_H[6]; state[7] = SHA512_H[7];
        
        __private unsigned long W[80] = {0};
        W[15] = 0x8000000000000000UL;  // 0x80 followed by 111 zeros, then length=0 in last 16 bytes
        
        sha512_transform(state, W);
        for (int i = 0; i < 8; i++) {
            unsigned long val = state[i];
            output[i*8+0] = (unsigned char)(val >> 56);
            output[i*8+1] = (unsigned char)(val >> 48);
            output[i*8+2] = (unsigned char)(val >> 40);
            output[i*8+3] = (unsigned char)(val >> 32);
            output[i*8+4] = (unsigned char)(val >> 24);
            output[i*8+5] = (unsigned char)(val >> 16);
            output[i*8+6] = (unsigned char)(val >> 8);
            output[i*8+7] = (unsigned char)(val);
        }
        return;
    }
    
    __private unsigned long state[8];
    state[0] = SHA512_H[0]; state[1] = SHA512_H[1]; state[2] = SHA512_H[2]; state[3] = SHA512_H[3];
    state[4] = SHA512_H[4]; state[5] = SHA512_H[5]; state[6] = SHA512_H[6]; state[7] = SHA512_H[7];
    
    ulong num_blocks = (msg_len + 16 + 127) / 128;
    for (ulong block_num = 0; block_num < num_blocks; block_num++) {
        __private unsigned long W[80];
        if (block_num == num_blocks - 1) {
            __private unsigned char block_data[128];
            for (int i = 0; i < 112; i++) block_data[i] = (block_num * 112 + i < msg_len) ? input[block_num * 112 + i] : 0;
            if (msg_len % 112 < 112) block_data[msg_len % 112] = 0x80;
            for (int i = 0; i < 8; i++) block_data[120 + i] = (unsigned char)((msg_len * 8) >> (56 - i * 8));
            for (int i = 0; i < 16; i++)
                W[i] = ((unsigned long)block_data[i*8] << 56) | ((unsigned long)block_data[i*8+1] << 48) |
                       ((unsigned long)block_data[i*8+2] << 40) | ((unsigned long)block_data[i*8+3] << 32) |
                       ((unsigned long)block_data[i*8+4] << 24) | ((unsigned long)block_data[i*8+5] << 16) |
                       ((unsigned long)block_data[i*8+6] << 8) | ((unsigned long)block_data[i*8+7]);
        } else {
            for (int i = 0; i < 16; i++) W[i] = load64_constant(input + block_num * 112 + i * 8);
        }
        for (int i = 16; i < 80; i++) W[i] = W[i-16] + little_s0(W[i-15]) + W[i-7] + little_s1(W[i-2]);
        sha512_transform(state, W);
    }
    for (int i = 0; i < 8; i++) {
        unsigned long val = state[i];
        output[i*8+0] = (unsigned char)(val >> 56);
        output[i*8+1] = (unsigned char)(val >> 48);
        output[i*8+2] = (unsigned char)(val >> 40);
        output[i*8+3] = (unsigned char)(val >> 32);
        output[i*8+4] = (unsigned char)(val >> 24);
        output[i*8+5] = (unsigned char)(val >> 16);
        output[i*8+6] = (unsigned char)(val >> 8);
        output[i*8+7] = (unsigned char)(val);
    }
}

#endif
