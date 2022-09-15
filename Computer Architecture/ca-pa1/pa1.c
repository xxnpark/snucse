//---------------------------------------------------------------
//
//  4190.308 Computer Architecture (Fall 2022)
//
//  Project #1:
//
//  September 6, 2022
//
//  Seongyeop Jeong (seongyeop.jeong@snu.ac.kr)
//  Jaehoon Shim (mattjs@snu.ac.kr)
//  IlKueon Kang (kangilkueon@snu.ac.kr)
//  Wookje Han (gksdnrwp@snu.ac.kr)
//  Jinsol Park (jinsolpark@snu.ac.kr)
//  Systems Software & Architecture Laboratory
//  Dept. of Computer Science and Engineering
//  Seoul National University
//
//---------------------------------------------------------------

typedef unsigned char u8;

int encode(const u8* src, int width, int height, u8* result) {
    int w, h;
    int ind = 0;
    int shift = 0;
    u8 carry = 0;

    for (h = 0; h < height; h++) {
        u8 min = 255;
        u8 max = 0;

        for (w = 0; w < width; w++) {
            u8 pixel = *(src + h * width + w);
            u8 avg;

            if (h == 0) {
                if (w == 0) {
                    avg = 0;
                } else {
                    avg = *(src + h * width + w-1);
                }
            } else {
                if (w == 0) {
                    avg = *(src + (h-1) * width + w);
                } else {
                    avg = (*(src + h * width + w-1) + *(src + (h-1) * width + w) + *(src + (h-1) * width + w-1)) / 3;
                }
            }

            u8 filter = pixel > avg ? pixel - avg : pixel - avg + 256;

            if (filter < min) {
                min = filter;
            }
            if (filter > max) {
                max = filter;
            }
        }

        u8 maxDelta = max - min;
        u8 n;

        if (maxDelta < 1) {
            n = 0;
        } else if (maxDelta < 2) {
            n = 1;
        } else if (maxDelta < 4) {
            n = 2;
        } else if (maxDelta < 8) {
            n = 3;
        } else if (maxDelta < 16) {
            n = 4;
        } else if (maxDelta < 32) {
            n = 5;
        } else if (maxDelta < 64) {
            n = 6;
        } else if (maxDelta < 128) {
            n = 7;
        } else {
            n = 8;
        }

        u8 shiftedN = n<<4;

        *(result + ind++) = carry + (min>>shift);
        carry = min<<(8-shift);

        if (shift >= 4) {
            *(result + ind++) = carry + (shiftedN>>shift);
            carry = shiftedN<<(8-shift);
            shift = shift - 4;
        } else {
            carry = carry + (shiftedN>>shift);
            shift = shift + 4;
        }

        if (n == 0) {
            continue;
        }

        for (w = 0; w < width; w++) {
            u8 pixel = *(src + h * width + w);
            u8 avg;

            if (h == 0) {
                if (w == 0) {
                    avg = 0;
                } else {
                    avg = *(src + h * width + w-1);
                }
            } else {
                if (w == 0) {
                    avg = *(src + (h-1) * width + w);
                } else {
                    avg = (*(src + h * width + w-1) + *(src + (h-1) * width + w) + *(src + (h-1) * width + w-1)) / 3;
                }
            }

            u8 filter = pixel > avg ? pixel - avg : pixel - avg + 256;
            u8 shiftedDelta = (filter-min)<<(8-n);

            if (shift >= 8 - n) {
                *(result + ind++) = carry + (shiftedDelta>>shift);
                carry = shiftedDelta<<(8-shift);
                shift = shift + n - 8;
            } else {
                carry = carry + (shiftedDelta>>shift);
                shift = shift + n;
            }
        }
    }

    if (shift) {
        *(result + ind++) = carry;
    }

    return ind;
}

