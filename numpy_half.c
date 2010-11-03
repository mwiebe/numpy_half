/*
 * IEEE Half-Precision Floating Point Type for NumPy
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

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "half.h"

typedef npy_uint16 npy_half;

typedef struct {
        PyObject_HEAD
        npy_half obval;
} PyHalfScalarObject;

PyTypeObject PyHalfArrType_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "half.float16",                             /* tp_name*/
    sizeof(PyHalfScalarObject),                 /* tp_basicsize*/
    0,                                          /* tp_itemsize */
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
#if defined(NPY_PY3K)
    0,                                          /* tp_reserved */
#else
    0,                                          /* tp_compare */
#endif
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    0,                                          /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    0,                                          /* tp_methods */
    0,                                          /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    0,                                          /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          /* tp_free */
    0,                                          /* tp_is_gc */
    0,                                          /* tp_bases */
    0,                                          /* tp_mro */
    0,                                          /* tp_cache */
    0,                                          /* tp_subclasses */
    0,                                          /* tp_weaklist */
    0,                                          /* tp_del */
#if PY_VERSION_HEX >= 0x02060000
    0,                                          /* tp_version_tag */
#endif
};

static PyArray_ArrFuncs _PyHalf_ArrFuncs;
PyArray_Descr *half_descr;

static npy_half
MyPyFloat_AsHalf(PyObject *obj)
{
    double d;
    npy_uint64 dbits;
    PyObject *num;

    if (obj == Py_None) {
        d = NPY_NAN;
    } else {
        num = PyNumber_Float(obj);
        if (num == NULL) {
            d = NPY_NAN;
        } else {
            d = PyFloat_AsDouble(num);
            Py_DECREF(num);
        }
    }
    /*
     * Was having some trouble getting the bits of 'd',
     * possibly a gcc compiler optimization bug?
     * This memcpy makes it work.
     */
    memcpy(&dbits, &d, 8);
    return doublebits_to_halfbits(dbits);
}

static PyObject *
HALF_getitem(char *ip, PyArrayObject *ap)
{
    npy_half t1;
    npy_uint64 t2;

    if ((ap == NULL) || PyArray_ISBEHAVED_RO(ap)) {
        t1 = *((npy_half *)ip);
        t2 = halfbits_to_doublebits(t1);
    }
    else {
        ap->descr->f->copyswap(&t1, ip, !PyArray_ISNOTSWAPPED(ap), ap);
        t2 = halfbits_to_doublebits(t1);
    }
    return PyFloat_FromDouble(*((double*)&t2));
}

static int HALF_setitem(PyObject *op, char *ov, PyArrayObject *ap)
{
    npy_half temp; /* ensures alignment */

    if (PyArray_IsScalar(op, Half)) {
        temp = ((PyHalfScalarObject *)op)->obval;
    }
    else {
        temp = MyPyFloat_AsHalf(op);
    }
    if (PyErr_Occurred()) {
        if (PySequence_Check(op)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError,
                    "setting an array element with a sequence.");
        }
        return -1;
    }
    if (ap == NULL || PyArray_ISBEHAVED(ap))
        *((npy_half *)ov)=temp;
    else {
        ap->descr->f->copyswap(ov, &temp, !PyArray_ISNOTSWAPPED(ap), ap);
    }
    return 0;

}

static void
HALF_copyswap (void *dst, void *src, int swap, void *NPY_UNUSED(arr))
{

    if (src != NULL) {
        /* copy first if needed */
        memcpy(dst, src, 2);
    }
    if (swap) {
        char *a, *b, c;

        a = (char *)dst;
        b = a + 1;
        c = *a; *a++ = *b; *b = c;
    }
}

static void
HALF_to_FLOAT(npy_half *ip, npy_uint32 *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = halfbits_to_floatbits(*ip++);
    }
}

static void
HALF_to_DOUBLE(npy_half *ip, npy_uint64 *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = halfbits_to_doublebits(*ip++);
    }
}

static void
HALF_to_LONGDOUBLE(npy_half *ip, npy_longdouble *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    npy_uint32 temp;

    while (n--) {
        temp = halfbits_to_floatbits(*ip++);
        *op++ = *((double*)&temp);
    }
}

static void
HALF_to_CFLOAT(npy_half *ip, npy_uint32 *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = halfbits_to_floatbits(*ip++);
        *op++ = 0;
    }
}

static void
HALF_to_CDOUBLE(npy_half *ip, npy_uint64 *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = halfbits_to_doublebits(*ip++);
        *op++ = 0;
    }
}

static void
HALF_to_CLONGDOUBLE(npy_half *ip, npy_longdouble *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    npy_uint32 temp;

    while (n--) {
        temp = halfbits_to_floatbits(*ip++);
        *op++ = *((float*)&temp);
        *op++ = 0.0;
    }
}

#define MAKE_HALF_TO_T(TYPE, type)                                             \
static void                                                                    \
HALF_to_ ## TYPE(npy_half *ip, type *op, npy_intp n,                           \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    npy_uint32 temp;                                                           \
                                                                               \
    while (n--) {                                                              \
        temp = halfbits_to_floatbits(*ip++);                                   \
        *op++ = (type)*((float*)&temp);                                        \
    }                                                                          \
}

MAKE_HALF_TO_T(BOOL, npy_bool);
MAKE_HALF_TO_T(BYTE, npy_byte);
MAKE_HALF_TO_T(UBYTE, npy_ubyte);
MAKE_HALF_TO_T(SHORT, npy_short);
MAKE_HALF_TO_T(USHORT, npy_ushort);
MAKE_HALF_TO_T(INT, npy_int);
MAKE_HALF_TO_T(UINT, npy_uint);
MAKE_HALF_TO_T(LONG, npy_long);
MAKE_HALF_TO_T(ULONG, npy_ulong);
MAKE_HALF_TO_T(LONGLONG, npy_longlong);
MAKE_HALF_TO_T(ULONGLONG, npy_ulonglong);
 
static void
FLOAT_to_HALF(npy_uint32 *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = floatbits_to_halfbits(*ip++);
    }
}
 
static void
DOUBLE_to_HALF(npy_uint64 *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = doublebits_to_halfbits(*ip++);
    }
}

static void
LONGDOUBLE_to_HALF(npy_longdouble *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    npy_uint64 temp;

    while (n--) {
        *((double*)&temp) = (double)(*ip++);
        *op++ = doublebits_to_halfbits(temp);
    }
}

static void
CFLOAT_to_HALF(npy_uint32 *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = floatbits_to_halfbits(*ip);
        ip += 2;
    }
}

static void
CDOUBLE_to_HALF(npy_uint64 *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    while (n--) {
        *op++ = doublebits_to_halfbits(*ip);
        ip += 2;
    }
}

static void
CLONGDOUBLE_to_HALF(npy_longdouble *ip, npy_half *op, npy_intp n,
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop))
{
    npy_uint64 temp;

    while (n--) {
        *((double*)&temp) = (double)(*ip);
        *op++ = doublebits_to_halfbits(temp);
        ip += 2;
    }
}


#define MAKE_T_TO_HALF(TYPE, type)                                             \
static void                                                                    \
TYPE ## _to_HALF(type *ip, npy_half *op, npy_intp n,                           \
               PyArrayObject *NPY_UNUSED(aip), PyArrayObject *NPY_UNUSED(aop)) \
{                                                                              \
    npy_uint32 temp;                                                           \
                                                                               \
    while (n--) {                                                              \
        *((float*)&temp) = (float)(*ip++);                                     \
        *op++ = floatbits_to_halfbits(temp);                                   \
    }                                                                          \
}

MAKE_T_TO_HALF(BOOL, npy_bool);
MAKE_T_TO_HALF(BYTE, npy_byte);
MAKE_T_TO_HALF(UBYTE, npy_ubyte);
MAKE_T_TO_HALF(SHORT, npy_short);
MAKE_T_TO_HALF(USHORT, npy_ushort);
MAKE_T_TO_HALF(INT, npy_int);
MAKE_T_TO_HALF(UINT, npy_uint);
MAKE_T_TO_HALF(LONG, npy_long);
MAKE_T_TO_HALF(ULONG, npy_ulong);
MAKE_T_TO_HALF(LONGLONG, npy_longlong);
MAKE_T_TO_HALF(ULONGLONG, npy_ulonglong);


static void register_cast_function(int sourceType, int destType, PyArray_VectorUnaryFunc *castfunc)
{
    PyArray_Descr *descr = PyArray_DescrFromType(sourceType);
    PyArray_RegisterCastFunc(descr, destType, castfunc);
    Py_DECREF(descr);
}

static PyObject *
half_arrtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *obj = NULL;
    npy_half value = 0;

    if (!PyArg_ParseTuple(args, "|O", &obj)) {
        return NULL;
    }
    if (obj != NULL) {
        value = MyPyFloat_AsHalf(obj);
    }
    return PyArray_Scalar(&value, half_descr, NULL);
}

static PyObject *
gentype_richcompare(PyObject *self, PyObject *other, int cmp_op)
{
    PyObject *arr, *ret;

    arr = PyArray_FromScalar(self, NULL);
    if (arr == NULL) {
        return NULL;
    }
    ret = Py_TYPE(arr)->tp_richcompare(arr, other, cmp_op);
    Py_DECREF(arr);
    return ret;
}

static long
halftype_hash(PyObject *obj)
{
    npy_uint64 temp;
    temp = halfbits_to_doublebits(((PyHalfScalarObject *)obj)->obval);
    return _Py_HashDouble(*((double*)&temp));
}

PyObject* halftype_repr(PyObject *o)
{
    npy_uint64 temp;
    char str[48];

    temp = halfbits_to_doublebits(((PyHalfScalarObject *)o)->obval);
    sprintf(str, "float16(%g)", *((double*)&temp));
    return PyString_FromString(str);
}

PyObject* halftype_str(PyObject *o)
{
    npy_uint64 temp;
    char str[48];

    temp = halfbits_to_doublebits(((PyHalfScalarObject *)o)->obval);
    sprintf(str, "%g", *((double*)&temp));
    return PyString_FromString(str);
}

static PyMethodDef HalfMethods[] = {
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initnumpy_half(void)
{
    PyObject *m;
    int halfNum;

    m = Py_InitModule("numpy_half", HalfMethods);
    if (m == NULL) {
        return;
    }

    /* Make sure NumPy is initialized */
    import_array();

    /* Register the half array scalar type */
#if defined(NPY_PY3K)
    PyHalfArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
#else
    PyHalfArrType_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES;
#endif
    PyHalfArrType_Type.tp_new = half_arrtype_new;
    PyHalfArrType_Type.tp_richcompare = gentype_richcompare;
    PyHalfArrType_Type.tp_hash = halftype_hash;
    PyHalfArrType_Type.tp_repr = halftype_repr;
    PyHalfArrType_Type.tp_str = halftype_str;
    PyHalfArrType_Type.tp_base = &PyFloatingArrType_Type;
    if (PyType_Ready(&PyHalfArrType_Type) < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_SystemError, "could not initialize PyHalfArrType_Type");
        return;
    }

    /* The array functions */
    PyArray_InitArrFuncs(&_PyHalf_ArrFuncs);
    _PyHalf_ArrFuncs.getitem = (PyArray_GetItemFunc*)HALF_getitem;
    _PyHalf_ArrFuncs.setitem = (PyArray_SetItemFunc*)HALF_setitem;
    _PyHalf_ArrFuncs.copyswap = (PyArray_CopySwapFunc*)HALF_copyswap;
    _PyHalf_ArrFuncs.cast[NPY_BOOL] = (PyArray_VectorUnaryFunc*)HALF_to_BOOL;
    _PyHalf_ArrFuncs.cast[NPY_BYTE] = (PyArray_VectorUnaryFunc*)HALF_to_BYTE;
    _PyHalf_ArrFuncs.cast[NPY_UBYTE] = (PyArray_VectorUnaryFunc*)HALF_to_UBYTE;
    _PyHalf_ArrFuncs.cast[NPY_SHORT] = (PyArray_VectorUnaryFunc*)HALF_to_SHORT;
    _PyHalf_ArrFuncs.cast[NPY_USHORT] = (PyArray_VectorUnaryFunc*)HALF_to_USHORT;
    _PyHalf_ArrFuncs.cast[NPY_INT] = (PyArray_VectorUnaryFunc*)HALF_to_INT;
    _PyHalf_ArrFuncs.cast[NPY_UINT] = (PyArray_VectorUnaryFunc*)HALF_to_UINT;
    _PyHalf_ArrFuncs.cast[NPY_LONG] = (PyArray_VectorUnaryFunc*)HALF_to_LONG;
    _PyHalf_ArrFuncs.cast[NPY_ULONG] = (PyArray_VectorUnaryFunc*)HALF_to_ULONG;
    _PyHalf_ArrFuncs.cast[NPY_LONGLONG] = (PyArray_VectorUnaryFunc*)HALF_to_LONGLONG;
    _PyHalf_ArrFuncs.cast[NPY_ULONGLONG] = (PyArray_VectorUnaryFunc*)HALF_to_ULONGLONG;
    _PyHalf_ArrFuncs.cast[NPY_FLOAT] = (PyArray_VectorUnaryFunc*)HALF_to_FLOAT;
    _PyHalf_ArrFuncs.cast[NPY_DOUBLE] = (PyArray_VectorUnaryFunc*)HALF_to_DOUBLE;
    _PyHalf_ArrFuncs.cast[NPY_LONGDOUBLE] = (PyArray_VectorUnaryFunc*)HALF_to_LONGDOUBLE;
    _PyHalf_ArrFuncs.cast[NPY_CFLOAT] = (PyArray_VectorUnaryFunc*)HALF_to_CFLOAT;
    _PyHalf_ArrFuncs.cast[NPY_CDOUBLE] = (PyArray_VectorUnaryFunc*)HALF_to_CDOUBLE;
    _PyHalf_ArrFuncs.cast[NPY_CLONGDOUBLE] = (PyArray_VectorUnaryFunc*)HALF_to_CLONGDOUBLE;

    /* The half array descr */
    half_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
    half_descr->typeobj = &PyHalfArrType_Type;
    half_descr->kind = 'f';
    half_descr->type = 'j';
    half_descr->byteorder = '=';
    half_descr->type_num = 0; /* assigned at registration */
    half_descr->elsize = 2;
    half_descr->alignment = 2;
    half_descr->subarray = NULL;
    half_descr->fields = NULL;
    half_descr->names = NULL;
    half_descr->f = &_PyHalf_ArrFuncs;


    Py_INCREF(&PyHalfArrType_Type);
    halfNum = PyArray_RegisterDataType(half_descr);

    if (halfNum < 0)
        return;

    register_cast_function(NPY_BOOL, halfNum, (PyArray_VectorUnaryFunc*)BOOL_to_HALF);
    register_cast_function(NPY_BYTE, halfNum, (PyArray_VectorUnaryFunc*)BYTE_to_HALF);
    register_cast_function(NPY_UBYTE, halfNum, (PyArray_VectorUnaryFunc*)UBYTE_to_HALF);
    register_cast_function(NPY_SHORT, halfNum, (PyArray_VectorUnaryFunc*)SHORT_to_HALF);
    register_cast_function(NPY_USHORT, halfNum, (PyArray_VectorUnaryFunc*)USHORT_to_HALF);
    register_cast_function(NPY_INT, halfNum, (PyArray_VectorUnaryFunc*)INT_to_HALF);
    register_cast_function(NPY_UINT, halfNum, (PyArray_VectorUnaryFunc*)UINT_to_HALF);
    register_cast_function(NPY_LONG, halfNum, (PyArray_VectorUnaryFunc*)LONG_to_HALF);
    register_cast_function(NPY_ULONG, halfNum, (PyArray_VectorUnaryFunc*)ULONG_to_HALF);
    register_cast_function(NPY_LONGLONG, halfNum, (PyArray_VectorUnaryFunc*)LONGLONG_to_HALF);
    register_cast_function(NPY_ULONGLONG, halfNum, (PyArray_VectorUnaryFunc*)ULONGLONG_to_HALF);
    register_cast_function(NPY_FLOAT, halfNum, (PyArray_VectorUnaryFunc*)FLOAT_to_HALF);
    register_cast_function(NPY_DOUBLE, halfNum, (PyArray_VectorUnaryFunc*)DOUBLE_to_HALF);
    register_cast_function(NPY_LONGDOUBLE, halfNum, (PyArray_VectorUnaryFunc*)LONGDOUBLE_to_HALF);
    register_cast_function(NPY_CFLOAT, halfNum, (PyArray_VectorUnaryFunc*)CFLOAT_to_HALF);
    register_cast_function(NPY_CDOUBLE, halfNum, (PyArray_VectorUnaryFunc*)CDOUBLE_to_HALF);
    register_cast_function(NPY_CLONGDOUBLE, halfNum, (PyArray_VectorUnaryFunc*)CLONGDOUBLE_to_HALF);

    PyArray_RegisterCanCast(half_descr, NPY_FLOAT, NPY_NOSCALAR);
    PyArray_RegisterCanCast(half_descr, NPY_DOUBLE, NPY_NOSCALAR);
    PyArray_RegisterCanCast(half_descr, NPY_LONGDOUBLE, NPY_NOSCALAR);
    PyArray_RegisterCanCast(half_descr, NPY_CFLOAT, NPY_NOSCALAR);
    PyArray_RegisterCanCast(half_descr, NPY_CDOUBLE, NPY_NOSCALAR);
    PyArray_RegisterCanCast(half_descr, NPY_CLONGDOUBLE, NPY_NOSCALAR);

    PyModule_AddObject(m, "float16", (PyObject *)&PyHalfArrType_Type);
}