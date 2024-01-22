from numbers import Integral, Real
import numpy

def check_is_ndarray(v: object, vname: str) -> None:
    """
    Check whether a Python object is a ``numpy.ndarray``.

    Parameters
    ----------
    v : object
        Python object to test.
    vname : str
        Name assigned to the Python object for an error message.
    """
    if not isinstance(v, numpy.ndarray):
        raise TypeError("variable '{0}' must be of type '{1}' but received type '{2}'".format(vname,numpy.ndarray.__name__,type(v).__name__))

def check_is_Real(v: object, vname: str) -> None:
    """
    Check whether a Python object is a ``Real``.

    Parameters
    ----------
    v : object
        Python object to test.
    vname : str
        Name assigned to the Python object for an error message.
    """
    if not isinstance(v, Real):
        raise TypeError("variable '{0}' must be of type '{1}' but received type '{2}'".format(vname,Real.__name__,type(v).__name__))

def check_is_Integral(v: object, vname: str) -> None:
    """
    Check whether a Python object is a ``Integral``.

    Parameters
    ----------
    v : object
        Python object to test.
    vname : str
        Name assigned to the Python object for an error message.
    """
    if not isinstance(v, Integral):
        raise TypeError("variable '{0}' must be of type '{1}' but received type '{2}'".format(vname,Integral.__name__,type(v).__name__))

def check_is_gteq(v: object, vname: str, value: object) -> None:
    """
    Check if a Python object is greater than or equal to another Python object.

    Parameters
    ----------
    v : object
        A Python object.
    vname : str
        Name of the Python object for use in the error message.
    value : object
        Lower value of the input Python object.
    """
    if v < value:
        raise ValueError("variable '{0}' is not greater than or equal to {1}".format(vname, value))

def check_ndarray_ndim(v: numpy.ndarray, vname: str, vndim: int) -> None:
    """
    Check if a ``numpy.ndarray`` has a specific number of dimensions.
    
    Parameters
    ----------
    v : numpy.ndarray
        Input array.
    vname : str
        Name assigned to input array.
    vndim : int
        Expected number of dimensions for the array.
    """
    if v.ndim != vndim:
        raise ValueError("numpy.ndarray '{0}' must have dimension equal to {1}".format(vname, vndim))

def check_ndarray_axis_len_eq(v: numpy.ndarray, vname: str, vaxis: int, vaxislen: int) -> None:
    """
    Check if a ``numpy.ndarray`` has a specific length along a specific axis.
    
    Parameters
    ----------
    v : numpy.ndarray
        Input array.
    vname : str
        Name assigned to input array.
    vaxis : int
        Axis along which to measure the length.
    vaxislen : int
        Expected length of the input array along the provided axis.
    """
    if v.shape[vaxis] != vaxislen:
        raise ValueError("numpy.ndarray '{0}' must have axis {1} equal to {2}".format(vname,vaxis,vaxislen))

