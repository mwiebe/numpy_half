/*
 * IEEE Half-Precision Floating Point Conversions
 * Copyright (c) 2010, Mark Wiebe
 *
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NumPy Developers nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTERS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "half.h"

typedef npy_uint16 UINT16_T;
typedef npy_uint32 UINT32_T;
typedef npy_uint64 UINT64_T;

UINT16_T floatbits_to_halfbits(UINT32_T f)
{
    UINT32_T f_exp, f_man;
    UINT16_T h_sgn, h_exp, h_man;

    h_sgn = (UINT16_T) ((f&0x80000000u) >> 16);
    f_exp = (f&0x7f800000u);
    
    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            f_man = (f&0x007fffffu);
            if (f_man != 0) {
                /* NaN - propagate the flag in the mantissa... */
                UINT16_T ret = (UINT16_T) (0x7c00u + (f_man >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (UINT16_T) (h_sgn + 0x7c00u);
            }
        } else {
            /* signed inf */
            return (UINT16_T) (h_sgn + 0x7c00u);
        }
    }
    
    /* Exponent underflow converts to denormalized half or signed zero */
    if (f_exp <= 0x38000000u) {
        /* 
         * Signed zeros, denormalized floats, and floats with small
         * exponents all convert to signed zero halfs.
         */
        if (f_exp < 0x33000000u) {
            return (UINT16_T) h_sgn;
        }
        /* Make the denormalized mantissa */
        f_exp >>= 23;
        f_man = (0x00800000u + (f&0x007fffffu)) >> (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
        h_man = (UINT16_T) ((f_man + 0x00001000u) >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_man will be zero.
         * This is the correct result.
         */
        return (UINT16_T) (h_sgn + h_man);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (UINT16_T) ((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_man = (f&0x007fffffu);
    h_man = (UINT16_T) ((f_man + 0x00001000u) >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_man will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
    return (UINT16_T) (h_sgn + h_exp + h_man);
}

UINT16_T doublebits_to_halfbits(UINT64_T d)
{
    UINT64_T d_exp, d_man;
    UINT16_T h_sgn, h_exp, h_man;

    h_sgn = (d&0x8000000000000000u) >> 48;
    d_exp = (d&0x7ff0000000000000u);
    
    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (d_exp >= 0x40f0000000000000u) {
        if (d_exp == 0x7ff0000000000000u) {
            d_man = (d&0x000fffffffffffffu);
            if (d_man != 0) {
                /* NaN - propagate the flag in the mantissa... */
                UINT16_T ret = (UINT16_T) (0x7c00u + (d_man >> 42));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (UINT16_T) (h_sgn + 0x7c00u);
            }
        } else {
            /* signed inf */
            return (UINT16_T) (h_sgn + 0x7c00u);
        }
    }
    
    /* Exponent underflow converts to denormalized half or signed zero */
    if (d_exp <= 0x3f00000000000000u) {
        /* 
         * Signed zeros, denormalized floats, and floats with small
         * exponents all convert to signed zero halfs.
         */
        if (d_exp < 0x3e60000000000000u) {
            return (UINT16_T) h_sgn;
        }
        /* Make the denormalized mantissa */
        d_exp >>= 52;
        d_man = (0x0010000000000000u + (d&0x000fffffffffffffu))
                                                    >> (1009 - d_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
        h_man = (UINT16_T) ((d_man + 0x0000020000000000u) >> 42);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_man will be zero.
         * This is the correct result.
         */
        return (UINT16_T) (h_sgn + h_man);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (UINT16_T) ((d_exp - 0x3f00000000000000u) >> 42);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    d_man = (d&0x000fffffffffffffu);
    h_man = (UINT16_T) ((d_man + 0x0000020000000000u) >> 42);

    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_man will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
    return (UINT16_T) (h_sgn + h_exp + h_man);
}

UINT32_T halfbits_to_floatbits(UINT16_T h)
{
    UINT16_T h_exp, h_man;
    UINT32_T f_sgn, f_exp, f_man;

    h_exp = (h&0x7c00u);
    f_sgn = ((UINT32_T)h&0x8000u) << 16;
    switch (h_exp) {
        case 0x0000u: /* 0 or denormalized */
            h_man = (h&0x03ffu);
            /* Signed zero */
            if (h_man == 0) {
                return f_sgn;
            }
            /* Denormalized */
            h_man <<= 1;
            while ((h_man&0x0400u) == 0) {
                h_man <<= 1;
                h_exp++;
            }
            f_exp = ((UINT32_T)(127 - 15 - h_exp)) << 23;
            f_man = ((UINT32_T)(h_man&0x03ffu)) << 13;
            return f_sgn + f_exp + f_man;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the mantissa */
            return f_sgn + 0x7f800000u + (((UINT32_T)(h&0x03ffu)) << 13);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return f_sgn + (((UINT32_T)(h&0x7fffu) + 0x1c000u) << 13);
    }
}

UINT64_T halfbits_to_doublebits(UINT16_T h)
{
    UINT16_T h_exp, h_man;
    UINT64_T d_sgn, d_exp, d_man;

    h_exp = (h&0x7c00u);
    d_sgn = ((UINT64_T)h&0x8000u) << 48;
    switch (h_exp) {
        case 0x0000u: /* 0 or denormalized */
            h_man = (h&0x03ffu);
            /* Signed zero */
            if (h_man == 0) {
                return d_sgn;
            }
            /* Denormalized */
            h_man <<= 1;
            while ((h_man&0x0400u) == 0) {
                h_man <<= 1;
                h_exp++;
            }
            d_exp = ((UINT64_T)(1023 - 15 - h_exp)) << 52;
            d_man = ((UINT64_T)(h_man&0x03ffu)) << 42;
            return d_sgn + d_exp + d_man;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the mantissa */
            return d_sgn + 0x7ff0000000000000u +
                                (((UINT64_T)(h&0x03ffu)) << 42);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return d_sgn + (((UINT64_T)(h&0x7fffu) + 0xfc000u) << 42);
    }
}
 
