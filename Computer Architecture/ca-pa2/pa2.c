typedef unsigned short SFP16;
#define PNAN ((SFP16) 0x7f01)
#define NNAN ((SFP16) 0xff01)
#define PINF ((SFP16) 0x7f00)
#define NINF ((SFP16) 0xff00)
#define swap(a, b) {SFP16 tmp = a; a = b; b = tmp;}

SFP16 fpadd(SFP16 x, SFP16 y) {
    // Set escapes for special cases
    if ((x == PNAN || x == NNAN) || (y == PNAN || y == NNAN)) {
        return PNAN;
    }
    if ((x == PINF || x == NINF) && (y == PINF || y == NINF)) {
        return x == y ? x : PNAN;
    }
    if ((x == PINF || x == NINF) || (y == PINF || y == NINF)) {
        return x == PINF || x == NINF ? x : y;
    }

    // Store s, e, f values for x and y
    SFP16 sx = x >> 15;
    SFP16 ex = (x >> 8) % (1 << 7);
    SFP16 fx = x % (1 << 8);
    SFP16 sy = y >> 15;
    SFP16 ey = (y >> 8) % (1 << 7);
    SFP16 fy = y % (1 << 8);

    // Swap x and y if |x| < |y|
    if (ex < ey || ex == ey && fx < fy) {
        swap(sx, sy)
        swap(ex, ey)
        swap(fx, fy)
    }

    // Store mx, my in the form of 1ffffffff000 (normalized) or 0ffffffff000 (denormalized)
    SFP16 mx = ex != 0 ? (fx << 3) + (1 << 11) : fx << 3;
    SFP16 my = ey != 0 ? (fy << 3) + (1 << 11) : fy << 3;

    // Set ex, ey = 1 if denormalized
    ex = ex != 0 ? ex : 1;
    ey = ey != 0 ? ey : 1;

    // Calculate d (amount to shift)
    SFP16 d = ex - ey;

    // Shift my considering the sticky bit
    for (SFP16 i = 0; i < d; i++) {
        SFP16 sticky = my % 2;
        my = my >> 1;
        my = (my >> 1 << 1) + (sticky | (my % 2));
    }

    // Add or subtract values
    SFP16 m = sx == sy ? mx + my : mx - my;
    SFP16 e = ex;
    SFP16 sign = sx;

    // Normalize m and modify GRS
    if (m >> 12 == 1) {
        m = (m >> 2 << 1) + (m % (1 << 2) != 0 ? 1 : 0);
        e++;
    }
    while (m >> 11 == 0 && e > 1) {
        m = m << 1;
        e--;
    }

    // Find L, R, S from m and delete GRS bits
    SFP16 l = m % (1 << 4) >> 3;
    SFP16 r = m % (1 << 3) >> 2;
    SFP16 s = m % (1 << 2) != 0 ? 1 : 0;
    m = m >> 3;

    // Round m (round to even)
    if (r == 1 && s == 1 || l == 1 && r == 1 && s == 0) {
        m = m + 1;
    }

    // Re-normalize m
    if (m >> 9 == 1) {
       m = m >> 1;
       e++;
    }

    // Adjust e in the case of adding two denormalized values
    if (m >> 8 == 0) {
        e--;
    }

    // Find output f and return in SFP16 format
    SFP16 f = m % (1 << 8);
    SFP16 res = (sign << 15) + (e << 8) + f;
    return res;
}
