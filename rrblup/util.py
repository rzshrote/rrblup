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

