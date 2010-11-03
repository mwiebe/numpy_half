import numpy as np
import half as h
from numpy.testing import *

import warnings

def test_half_consistency():
    """Checks that all 16-bit values survive conversion
       to/from 32-bit and 64-bit float"""
    # Create an array of all possible 16-bit values
    a = np.arange(0x10000, dtype=np.uint16)
    a_f16 = a.copy()
    a_f16.dtype = h.float16

    # Convert float16 to float32 and back
    a_f32 = np.array(a_f16, dtype=np.float32)
    b = np.array(a_f32, dtype=h.float16)
    b.dtype = np.uint16
    assert_equal(a,b)

    # Convert float16 to float64 and back
    a_f64 = np.array(a_f16, dtype=np.float64)
    b = np.array(a_f64, dtype=h.float16)
    b.dtype = np.uint16
    assert_equal(a,b)

    # Convert float16 to longdouble and back
    a_ld = np.array(a_f16, dtype=np.longdouble)
    b = np.array(a_f64, dtype=h.float16)
    b.dtype = np.uint16
    assert_equal(a,b)

    # Check the range for which all integers can be represented
    a = np.arange(-2048,2049)
    a_f16 = np.array(a, dtype=h.float16)
    b = np.array(a_f16, dtype=np.int)
    assert_equal(a,b)

def test_half_values():
    """Confirms a small number of known half values"""
    a = np.array([1.0, -1.0,
                  2.0, -2.0,
                  0.0999755859375, 0.333251953125, # 1/10, 1/3
                  65504, -65504,           # Maximum magnitude
                  2.0**(-14), -2.0**(-14), # Minimum normalized
                  2.0**(-24), -2.0**(-24), # Minimum denormalized
                  0, -1/1e1000,            # Signed zeros
                  np.inf, -np.inf])
    b = np.array([0x3c00, 0xbc00,
                  0x4000, 0xc000,
                  0x2e66, 0x3555,
                  0x7bff, 0xfbff,
                  0x0400, 0x8400,
                  0x0001, 0x8001,
                  0x0000, 0x8000,
                  0x7c00, 0xfc00], dtype=np.uint16)
    b.dtype = h.float16
    assert_equal(a, b)

def test_half_rounding():
    """Checks that rounding when converting to half is correct"""
    a = np.array([2.0**(-25),  # Rounds to minimum denormalized
                  2.0**(-26),  # Underflows to zero
                  1.0+2.0**(-11), # rounds to 1.0+2**(-10)
                  1.0+2.0**(-12), # rounds to 1.0
                  65519,          # rounds to 65504
                  65520],         # rounds to inf
                  dtype=np.float64)
    rounded = [2.0**(-24),
                     0.0,
                     1.0+2.0**(-10),
                     1.0,
                     65504,
                     np.inf]

    # Check float64->float16 rounding
    b = np.array(a, dtype=h.float16)
    assert_equal(b, rounded)

    # Check float32->float16 rounding
    a = np.array(a, dtype=np.float32)
    b = np.array(a, dtype=h.float16)
    assert_equal(b, rounded)

@dec.slow
def test_half_correctness():
    """Builds every finite half-float value in python
       code and compares it to the dtype conversion"""
    # Create an array of all possible 16-bit values
    a = np.arange(0x10000, dtype=np.uint16)
    a_f16 = a.copy()
    a_f16.dtype = h.float16
    # Convert to 32-bit and 64-bit float with the numpy machinery
    a_f32 = np.array(a_f16, dtype=np.float32)
    a_f64 = np.array(a_f16, dtype=np.float64)
    # Compare to hand-built values for each finite value
    for i in a:
        sgn = (i&0x8000) >> 15
        exp = (i&0x7c00) >> 10
        man = (i&0x03ff)
        if exp != 31: # Skip inf and NaN
            if exp == 0:
                man = man * 2**(-10)
                exp = -14
            else:
                man = (0x0400+man) * 2**(-10)
                exp = exp - 15

            val = (-1.0)**sgn * man * 2.0**exp
            assert_equal(a_f16[i], val, "Half value 0x%x isn't correct" % i)
            assert_equal(a_f32[i], val, "Half value 0x%x isn't correct" % i)
            assert_equal(a_f64[i], val, "Half value 0x%x isn't correct" % i)
