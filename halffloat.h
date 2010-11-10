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
#ifndef __HALF_H__
#define __HALF_H__

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef npy_uint16 npy_half;

/*
 * Half-precision routines
 */

/* Conversions */
float half_to_float(npy_half h);
double half_to_double(npy_half h);
npy_half float_to_half(float f);
npy_half double_to_half(double d);
/* Comparisons */
int half_eq(npy_half h1, npy_half h2);
int half_ne(npy_half h1, npy_half h2);
int half_le(npy_half h1, npy_half h2);
int half_lt(npy_half h1, npy_half h2);
int half_ge(npy_half h1, npy_half h2);
int half_gt(npy_half h1, npy_half h2);
/* faster *_nonan variants for when you know h1 and h2 are not NaN */
int half_eq_nonan(npy_half h1, npy_half h2);
int half_lt_nonan(npy_half h1, npy_half h2);
int half_le_nonan(npy_half h1, npy_half h2);
/* Miscellaneous functions */
int half_isnonzero(npy_half h);
int half_isnan(npy_half h);
int half_isinf(npy_half h);
int half_isfinite(npy_half h);
int half_signbit(npy_half h);
npy_half half_spacing(npy_half h);
npy_half half_copysign(npy_half x, npy_half y);
npy_half half_nextafter(npy_half x, npy_half y);

/*
 * Half-precision constants
 */

#define HALF_ZERO   (0x0000u)
#define HALF_PZERO  (0x0000u)
#define HALF_NZERO  (0x8000u)
#define HALF_ONE    (0x3c00u)
#define HALF_NEGONE (0xbc00u)
#define HALF_PINF   (0x7c00u)
#define HALF_NINF   (0xfc00u)
#define HALF_NAN    (0x7e00u)

/*
 * Bit-level conversions
 */

npy_uint16 floatbits_to_halfbits(npy_uint32 f);
npy_uint16 doublebits_to_halfbits(npy_uint64 d);
npy_uint32 halfbits_to_floatbits(npy_uint16 h);
npy_uint64 halfbits_to_doublebits(npy_uint16 h);

#ifdef __cplusplus
}
#endif

#endif
