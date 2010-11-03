from info import __doc__

__all__ = ['float16']

from numpy_half import float16

def add_to_typeDict():
    # Add it to the numpy type dictionary
    from numpy import typeDict
    typeDict['float16'] = float16
    typeDict['f2'] = float16

add_to_typeDict()

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
