// Ed25519: elliptic curve operations for OpenSSH ed25519-keys
// Caller MUST include big_math.cl BEFORE this file.
// This file does NOT include big_math.cl to avoid double-definition with test wrappers.

__constant uint ED_D[8] = {
    324630691, 1978355146, 1094834347, 7342669, 2004478104, 2361868409, 728759923, 1375956206
};

__constant uint ED_BASE_X[8] = {
    2401621274, 3377868128, 2502272946, 1764542304, 4258716764, 3232031281, 3446559742, 560543443
};

__constant uint ED_BASE_Y[8] = {
    0x66666658u, 0x66666666u, 0x66666666u, 0x66666666u,
    0x66666666u, 0x66666666u, 0x66666666u, 0x66666666u
};

__inline void read_32bytes(__generic const uchar* inp, int offset, __generic uint* limb) {
    for(int i=0;i<8;i++) limb[i]=(uint)(inp[offset+i*4]|(inp[offset+i*4+1]<<8)|(inp[offset+i*4+2]<<16)|(inp[offset+i*4+3]<<24));
}
__inline void write_32bytes(__generic uchar* out, int offset, __generic const uint* limb) {
    for(int i=0;i<8;i++){out[offset+i*4]=(uchar)(limb[i]&0xFF);out[offset+i*4+1]=(uchar)((limb[i]>>8)&0xFF);out[offset+i*4+2]=(uchar)((limb[i]>>16)&0xFF);out[offset+i*4+3]=(uchar)((limb[i]>>24)&0xFF);}
}

// Subtraction mod p: r = (a - b) mod p
__inline void sub_mod_p(__generic const uint* a, __generic const uint* b, __generic uint* r) {
    long borrow = 0;
    for (int i = 0; i < 8; i++) {
        long diff = (long)a[i] - (long)b[i] - borrow;
        r[i] = (uint)(diff & 0xFFFFFFFF);
        borrow = (diff < 0) ? 1 : 0;
    }
    if (borrow) {
        uint pl[8]; pl[0]=0xFFFFFFED;pl[1]=0xFFFFFFFF;pl[2]=0xFFFFFFFF;pl[3]=0xFFFFFFFF;
        pl[4]=0xFFFFFFFF;pl[5]=0xFFFFFFFF;pl[6]=0xFFFFFFFF;pl[7]=0x7FFFFFFF;
        add_256(r, pl, r);
    }
}

// Projective addition
__inline void point_add_proj(uint* X1, uint* Y1, uint* Z1, uint* X2, uint* Y2, uint* Z2, uint* X3, uint* Y3, uint* Z3) {
    uint A[8], B[8], C[8], Dd[8], E[8], F[8], G[8], S[8], T[8];
    mul_mod_p(Z1, Z2, A);
    mul_mod_p(A, A, B);
    mul_mod_p(X1, Y1, C);
    mul_mod_p(X2, Y2, Dd);
    mul_mod_p(C, Dd, E);
    uint DL[8]; for(int k=0;k<8;k++) DL[k]=ED_D[k];
    mul_mod_p(DL, E, E);
    add_256(B, E, S); mod_p_reduce(S);
    sub_mod_p(B, E, T);
    mul_mod_p(X1, Y2, F);
    mul_mod_p(X2, Y1, G);
    add_256(F, G, F); mod_p_reduce(F);
    mul_mod_p(F, A, F);
    mul_mod_p(F, T, X3);
    mul_mod_p(Y1, Y2, F);
    mul_mod_p(X1, X2, G);
    // a=-1: y3 = (y1*y2 - a*x1*x2) = (y1*y2 + x1*x2)
    add_256(F, G, F); mod_p_reduce(F);
    mul_mod_p(F, A, F);
    mul_mod_p(F, S, Y3);
    mul_mod_p(S, T, Z3);
}

__inline void point_add_projective(__generic const uint* X1, __generic const uint* Y1, __generic const uint* Z1,
                                   __generic const uint* X2, __generic const uint* Y2, __generic const uint* Z2,
                                   __generic uint* X3, __generic uint* Y3, __generic uint* Z3) {
    uint lx1[8], ly1[8], lz1[8], lx2[8], ly2[8], lz2[8];
    copy_256(X1, lx1); copy_256(Y1, ly1); copy_256(Z1, lz1);
    copy_256(X2, lx2); copy_256(Y2, ly2); copy_256(Z2, lz2);
    int p1id=1, p2id=1;
    for(int k=0;k<8;k++){
        if(k==0){if(lx1[k]!=0||ly1[k]!=1||lz1[k]!=1)p1id=0;if(lx2[k]!=0||ly2[k]!=1||lz2[k]!=1)p2id=0;}
        else    {if(lx1[k]!=0||ly1[k]!=0||lz1[k]!=0)p1id=0;if(lx2[k]!=0||ly2[k]!=0||lz2[k]!=0)p2id=0;}
    }
    if(p1id){copy_256(lx2,X3);copy_256(ly2,Y3);copy_256(lz2,Z3);return;}
    if(p2id){copy_256(lx1,X3);copy_256(ly1,Y3);copy_256(lz1,Z3);return;}
    point_add_proj(lx1, ly1, lz1, lx2, ly2, lz2, X3, Y3, Z3);
}

__inline void point_init_base(__generic uint* X, __generic uint* Y, __generic uint* Z) {
    uint BL[8]; for(int k=0;k<8;k++) BL[k]=ED_BASE_X[k]; copy_256(BL, X);
    uint BYL[8]; for(int k=0;k<8;k++) BYL[k]=ED_BASE_Y[k]; copy_256(BYL, Y);
    one_256(Z);
}

__inline void point_to_affine_x(__generic const uint* X, __generic const uint* Y, __generic const uint* Z, __generic uchar* x) {
    uint Zi[8], Xa[8]; mod_p_inverse(Z, Zi); mul_mod_p(X, Zi, Xa); scalar_to_bytes(Xa, x);
}
__inline void point_to_affine_y(__generic const uint* X, __generic const uint* Y, __generic const uint* Z, __generic uchar* y) {
    uint Zi[8], Ya[8]; mod_p_inverse(Z, Zi); mul_mod_p(Y, Zi, Ya); scalar_to_bytes(Ya, y);
}

__inline void scalar_mult(__generic const uchar* scalar_bytes,
                          __generic uint* result_X, __generic uint* result_Y, __generic uint* result_Z) {
    // Double-and-add: start with identity, process bits MSB to LSB
    uint RX[8], RY[8], RZ[8];
    zero_256(RX); one_256(RY); one_256(RZ);
    uint BX[8], BY[8], BZ[8];
    for(int k=0;k<8;k++){BX[k]=ED_BASE_X[k];BY[k]=ED_BASE_Y[k];}
    one_256(BZ);
    uint sl[8]; zero_256(sl); seed_to_scalar(scalar_bytes, sl);
    for(int i=255;i>=0;i--){
        // Double
        uint Xt[8],Yt[8],Zt[8];
        point_add_proj(RX,RY,RZ,RX,RY,RZ,Xt,Yt,Zt);
        copy_256(Xt,RX);copy_256(Yt,RY);copy_256(Zt,RZ);
        // Add base if bit set
        int bi=i/32,bs=i%32;
        if((sl[bi]>>bs)&1u){
            point_add_proj(RX,RY,RZ,BX,BY,BZ,Xt,Yt,Zt);
            copy_256(Xt,RX);copy_256(Yt,RY);copy_256(Zt,RZ);
        }
    }
    copy_256(RX,result_X);copy_256(RY,result_Y);copy_256(RZ,result_Z);
}