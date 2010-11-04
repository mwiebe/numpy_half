from info import __doc__

__all__ = ['float16']

from numpy_half import float16

def add_to_typeDict():
    # Add it to the numpy type dictionary
    import sys, numpy
    f16 = numpy.dtype(float16)
    numpy.typeDict['float16'] = f16
    numpy.typeDict['f2'] = f16
    numpy.typeDict['=f2'] = f16
    if sys.byteorder == 'little':
        numpy.typeDict['<f2'] = f16
        numpy.typeDict['>f2'] = f16.newbyteorder('>')
    else:
        typeDict['>f2'] = f16
        numpy.typeDict['<f2'] = f16.newbyteorder('<')


add_to_typeDict()

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
