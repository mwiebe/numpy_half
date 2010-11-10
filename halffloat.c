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
#include "halffloat.h"
#include "numpy/ufuncobject.h"

/*
 * This chooses between 'ties to even' and 'ties away from zero'.
 */
#define HALF_ROUND_TIES_TO_EVEN 1
/*
 * If these are 1, the conversions try to trigger underflow
 * and overflow in the FP system when needed.
 */
#define HALF_GENERATE_OVERFLOW 1
#define HALF_GENERATE_UNDERFLOW 1
#define HALF_GENERATE_INVALID 1

#if !defined(generate_overflow_error)
static double numeric_over_big = 1e300;
static void generate_overflow_error(void) {
        double dummy;
        dummy = numeric_over_big * 1e300;
        if (dummy)
           return;
        else
           numeric_over_big += 0.1;
        return;
        return;
}
#endif

#if !defined(generate_underflow_error)
static double numeric_under_small = 1e-300;
static void generate_underflow_error(void) {
        double dummy;
        dummy = numeric_under_small * 1e-300;
        if (!dummy)
           return;
        else
           numeric_under_small += 1e-300;
        return;
}
#endif

#if !defined(generate_invalid_error)
static double numeric_inv_inf = 1e1000;
static void generate_invalid_error(void) {
        double dummy;
        dummy = numeric_inv_inf - 1e1000;
        if (!dummy)
           return;
        else
           numeric_inv_inf += 1.0;
        return;
}
#endif


/*
 ********************************************************************
 *                   HALF-PRECISION ROUTINES                        *
 ********************************************************************
 */

float
half_to_float(npy_half h)
{
    float ret;
    *((npy_uint32*)&ret) = halfbits_to_floatbits(h);
    return ret;
}

double
half_to_double(npy_half h)
{
    double ret;
    *((npy_uint64*)&ret) = halfbits_to_doublebits(h);
    return ret;
}

npy_half
float_to_half(float f)
{
    return floatbits_to_halfbits(*((npy_uint32*)&f));
}

npy_half
double_to_half(double d)
{
    return doublebits_to_halfbits(*((npy_uint64*)&d));
}

int
half_isnonzero(npy_half h)
{
    return (h&0x7fff) != 0;
}

int
half_isnan(npy_half h)
{
    return ((h&0x7c00u) == 0x7c00u) && ((h&0x03ffu) != 0x0000u);
}

int
half_isinf(npy_half h)
{
    return ((h&0x7c00u) == 0x7c00u) && ((h&0x03ffu) == 0x0000u);
}

int
half_isfinite(npy_half h)
{
    return ((h&0x7c00u) != 0x7c00u);
}

int
half_signbit(npy_half h)
{
    return (h&0x8000u) != 0;
}

npy_half
half_spacing(npy_half h)
{
    npy_half ret;
    npy_uint16 h_exp = h&0x7c00u;
    npy_uint16 h_man = h&0x03ffu;
    if (h_exp == 0x7c00u || h == 0x7bffu) {
#if HALF_GENERATE_INVALID
        generate_invalid_error();
#endif
        ret = HALF_NAN;
    } else if ((h&0x8000u) && h_man == 0) { /* Negative boundary case */
        if (h_exp > 0x2c00u) { /* If result is normalized */
            ret = h_exp - 0x2c00u;
        } else if(h_exp > 0x0400u) { /* The result is denormalized, but not the smallest */
            ret = 1 << ((h_exp >> 10) - 2);
        } else {
            ret = 0x0001u; /* Smallest denormalized half */
        }
    } else if (h_exp > 0x2800u) { /* If result is still normalized */
        ret = h_exp - 0x2800u;
    } else if (h_exp > 0x0400u) { /* The result is denormalized, but not the smallest */
        ret = 1 << ((h_exp >> 10) - 1);
    } else {
        ret = 0x0001u;
    }

    return ret;
}

npy_half
half_copysign(npy_half x, npy_half y)
{
    return (x&0x7fffu) | (y&0x8000u);
}

npy_half
half_nextafter(npy_half x, npy_half y)
{
    npy_half ret;

    if (!half_isfinite(x) || half_isnan(y)) {
#if HALF_GENERATE_INVALID
        generate_invalid_error();
#endif
        ret = HALF_NAN;
    } else if (half_eq_nonan(x, y)) {
        ret = x;
    } else if (!half_isnonzero(x)) {
        ret = (y&0x8000u) + 1; /* Smallest denormalized half */
    } else if (!(x&0x8000u)) { /* x > 0 */
        if ((npy_int16)x > (npy_int16)y) { /* x > y */
            ret = x-1;
        } else {
            ret = x+1;
        }
    } else {
        if (!(y&0x8000u) || (x&0x7fffu) > (y&0x7fffu)) { /* x < y */
            ret = x-1;
        } else {
            ret = x+1;
        }
    }
#ifdef HALF_GENERATE_OVERFLOW
    if (half_isinf(ret)) {
        generate_overflow_error();
    }
#endif

    return ret;
}
 
int
half_eq_nonan(npy_half h1, npy_half h2)
{
    return (h1 == h2 || ((h1 | h2) & 0x7fff) == 0);
}

int
half_eq(npy_half h1, npy_half h2)
{
    /*
     * The equality cases are as follows:
     *   - If either value is NaN, never equal.
     *   - If the values are equal, equal.
     *   - If the values are both signed zeros, equal.
     */
    return (!half_isnan(h1) && !half_isnan(h2)) &&
           (h1 == h2 || ((h1 | h2) & 0x7fff) == 0);
}

int
half_ne(npy_half h1, npy_half h2)
{
    return !half_eq(h1, h2);
}

int
half_lt_nonan(npy_half h1, npy_half h2)
{
    if (h1&0x8000u) {
        if (h2&0x8000u) {
            return (h1&0x7fffu) > (h2&0x7fffu);
        } else {
            /* Signed zeros are equal, have to check for it */
            return (h1 != 0x8000u) || (h2 != 0x0000u);
        }
    } else {
        if (h2&0x8000u) {
            return 0;
        } else {
            return (h1&0x7fffu) < (h2&0x7fffu);
        }
    }
}

int
half_lt(npy_half h1, npy_half h2)
{
    return (!half_isnan(h1) && !half_isnan(h2)) && half_lt_nonan(h1, h2);
}

int
half_gt(npy_half h1, npy_half h2)
{
    return half_lt(h2, h1);
}

int
half_le_nonan(npy_half h1, npy_half h2)
{
    if (h1&0x8000u) {
        if (h2&0x8000u) {
            return (h1&0x7fffu) >= (h2&0x7fffu);
        } else {
            return 1;
        }
    } else {
        if (h2&0x8000u) {
            /* Signed zeros are equal, have to check for it */
            return (h1 == 0x0000u) && (h2 == 0x8000u);
        } else {
            return (h1&0x7fffu) <= (h2&0x7fffu);
        }
    }
}

int
half_le(npy_half h1, npy_half h2)
{
    return (!half_isnan(h1) && !half_isnan(h2)) && half_le_nonan(h1, h2);
}

int
half_ge(npy_half h1, npy_half h2)
{
    return half_le(h2, h1);
}



/*
 ********************************************************************
 *                     BIT-LEVEL CONVERSIONS                        *
 ********************************************************************
 */

/*TODO
 * Should these routines query the CPU float rounding flags?
 * The routine currently does 'ties to even', or 'ties away
 * from zero', depending on a #define above.
 */

npy_uint16
floatbits_to_halfbits(npy_uint32 f)
{
    npy_uint32 f_exp, f_man;
    npy_uint16 h_sgn, h_exp, h_man;

    h_sgn = (npy_uint16) ((f&0x80000000u) >> 16);
    f_exp = (f&0x7f800000u);
    
    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (f_exp >= 0x47800000u) {
        if (f_exp == 0x7f800000u) {
            /*
             * No need to generate FP_INVALID or FP_OVERFLOW here, as
             * the float/double routine should have done that.
             */
            f_man = (f&0x007fffffu);
            if (f_man != 0) {
                /* NaN - propagate the flag in the mantissa... */
                npy_uint16 ret = (npy_uint16) (0x7c00u + (f_man >> 13));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return (npy_uint16) (h_sgn + 0x7c00u);
            }
        } else {
            /* overflow to signed inf */
#if HALF_GENERATE_OVERFLOW
            generate_overflow_error();
#endif
            return (npy_uint16) (h_sgn + 0x7c00u);
        }
    }
    
    /* Exponent underflow converts to denormalized half or signed zero */
    if (f_exp <= 0x38000000u) {
        /* 
         * Signed zeros, denormalized floats, and floats with small
         * exponents all convert to signed zero halfs.
         */
        if (f_exp < 0x33000000u) {
#if HALF_GENERATE_UNDERFLOW 
            /* If f != 0, we underflowed to 0 */
            if ((f&0x7fffffff) != 0) {
                generate_underflow_error();
            }
#endif
            return h_sgn;
        }
        /* It underflowed to a denormalized value */
#if HALF_GENERATE_UNDERFLOW 
        generate_underflow_error();
#endif
        /* Make the denormalized mantissa */
        f_exp >>= 23;
        f_man = (0x00800000u + (f&0x007fffffu)) >> (113 - f_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
#if HALF_ROUND_TIES_TO_EVEN 
        /*
         * If the last bit in the half mantissa is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half mantissa.  In all other cases, we do.
         */
        if ((f_man&0x00003fffu) != 0x00001000u) {
            f_man += 0x00001000u;
        }
#else
        f_man += 0x00001000u;
#endif
        h_man = (npy_uint16) (f_man >> 13);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_man will be zero.
         * This is the correct result.
         */
        return (npy_uint16) (h_sgn + h_man);
    }

    /* Regular case with no overflow or underflow */
    h_exp = (npy_uint16) ((f_exp - 0x38000000u) >> 13);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    f_man = (f&0x007fffffu);
#if HALF_ROUND_TIES_TO_EVEN 
    /*
     * If the last bit in the half mantissa is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half mantissa.  In all other cases, we do.
     */
    if ((f_man&0x00003fffu) != 0x00001000u) {
        f_man += 0x00001000u;
    }
#else
    f_man += 0x00001000u;
#endif
    h_man = (npy_uint16) (f_man >> 13);
    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_man will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
#if HALF_GENERATE_OVERFLOW
    h_man += h_exp;
    if (h_man == 0x7c00u) {
        generate_overflow_error();
    }
    return h_sgn + h_man;
#else
    return h_sgn + h_exp + h_man;
#endif
}

npy_uint16
doublebits_to_halfbits(npy_uint64 d)
{
    npy_uint64 d_exp, d_man;
    npy_uint16 h_sgn, h_exp, h_man;

    h_sgn = (d&0x8000000000000000u) >> 48;
    d_exp = (d&0x7ff0000000000000u);
    
    /* Exponent overflow/NaN converts to signed inf/NaN */
    if (d_exp >= 0x40f0000000000000u) {
        if (d_exp == 0x7ff0000000000000u) {
            /*
             * No need to generate FP_INVALID or FP_OVERFLOW here, as
             * the float/double routine should have done that.
             */
            d_man = (d&0x000fffffffffffffu);
            if (d_man != 0) {
                /* NaN - propagate the flag in the mantissa... */
                npy_uint16 ret = (npy_uint16) (0x7c00u + (d_man >> 42));
                /* ...but make sure it stays a NaN */
                if (ret == 0x7c00u) {
                    ret++;
                }
                return h_sgn + ret;
            } else {
                /* signed inf */
                return h_sgn + 0x7c00u;
            }
        } else {
            /* overflow to signed inf */
#if HALF_GENERATE_OVERFLOW
            generate_overflow_error();
#endif
            return h_sgn + 0x7c00u;
        }
    }
    
    /* Exponent underflow converts to denormalized half or signed zero */
    if (d_exp <= 0x3f00000000000000u) {
        /* 
         * Signed zeros, denormalized floats, and floats with small
         * exponents all convert to signed zero halfs.
         */
        if (d_exp < 0x3e60000000000000u) {
#if HALF_GENERATE_UNDERFLOW 
            /* If d != 0, we underflowed to 0 */
            if ((d&0x7fffffffffffffff) != 0) {
                generate_underflow_error();
            }
#endif
            return h_sgn;
        }
        /* It underflowed to a denormalized value */
#if HALF_GENERATE_UNDERFLOW 
        generate_underflow_error();
#endif
        /* Make the denormalized mantissa */
        d_exp >>= 52;
        d_man = (0x0010000000000000u + (d&0x000fffffffffffffu))
                                                    >> (1009 - d_exp);
        /* Handle rounding by adding 1 to the bit beyond half precision */
#if HALF_ROUND_TIES_TO_EVEN 
        /*
         * If the last bit in the half mantissa is 0 (already even), and
         * the remaining bit pattern is 1000...0, then we do not add one
         * to the bit after the half mantissa.  In all other cases, we do.
         */
        if ((d_man&0x000007ffffffffffu) != 0x0000020000000000u) {
            d_man += 0x0000020000000000u;
        }
#else
        d_man += 0x0000020000000000u;
#endif
        h_man = (npy_uint16) (d_man >> 42);
        /*
         * If the rounding causes a bit to spill into h_exp, it will
         * increment h_exp from zero to one and h_man will be zero.
         * This is the correct result.
         */
        return h_sgn + h_man;
    }

    /* Regular case with no overflow or underflow */
    h_exp = (npy_uint16) ((d_exp - 0x3f00000000000000u) >> 42);
    /* Handle rounding by adding 1 to the bit beyond half precision */
    d_man = (d&0x000fffffffffffffu);
#if HALF_ROUND_TIES_TO_EVEN 
    /*
     * If the last bit in the half mantissa is 0 (already even), and
     * the remaining bit pattern is 1000...0, then we do not add one
     * to the bit after the half mantissa.  In all other cases, we do.
     */
    if ((d_man&0x000007ffffffffffu) != 0x0000020000000000u) {
        d_man += 0x0000020000000000u;
    }
#else
    d_man += 0x0000020000000000u;
#endif
    h_man = (npy_uint16) (d_man >> 42);

    /*
     * If the rounding causes a bit to spill into h_exp, it will
     * increment h_exp by one and h_man will be zero.  This is the
     * correct result.  h_exp may increment to 15, at greatest, in
     * which case the result overflows to a signed inf.
     */
#if HALF_GENERATE_OVERFLOW
    h_man += h_exp;
    if (h_man == 0x7c00u) {
        generate_overflow_error();
    }
    return h_sgn + h_man;
#else
    return h_sgn + h_exp + h_man;
#endif
}

npy_uint32
halfbits_to_floatbits(npy_uint16 h)
{
    npy_uint16 h_exp, h_man;
    npy_uint32 f_sgn, f_exp, f_man;

    h_exp = (h&0x7c00u);
    f_sgn = ((npy_uint32)h&0x8000u) << 16;
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
            f_exp = ((npy_uint32)(127 - 15 - h_exp)) << 23;
            f_man = ((npy_uint32)(h_man&0x03ffu)) << 13;
            return f_sgn + f_exp + f_man;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the mantissa */
            return f_sgn + 0x7f800000u + (((npy_uint32)(h&0x03ffu)) << 13);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return f_sgn + (((npy_uint32)(h&0x7fffu) + 0x1c000u) << 13);
    }
}

npy_uint64
halfbits_to_doublebits(npy_uint16 h)
{
    npy_uint16 h_exp, h_man;
    npy_uint64 d_sgn, d_exp, d_man;

    h_exp = (h&0x7c00u);
    d_sgn = ((npy_uint64)h&0x8000u) << 48;
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
            d_exp = ((npy_uint64)(1023 - 15 - h_exp)) << 52;
            d_man = ((npy_uint64)(h_man&0x03ffu)) << 42;
            return d_sgn + d_exp + d_man;
        case 0x7c00u: /* inf or NaN */
            /* All-ones exponent and a copy of the mantissa */
            return d_sgn + 0x7ff0000000000000u +
                                (((npy_uint64)(h&0x03ffu)) << 42);
        default: /* normalized */
            /* Just need to adjust the exponent and shift */
            return d_sgn + (((npy_uint64)(h&0x7fffu) + 0xfc000u) << 42);
    }
}
 
