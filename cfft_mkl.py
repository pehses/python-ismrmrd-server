import numpy as np
from numpy.fft import fftshift, ifftshift
try:
    import mkl_fft
    def fft(data, axis, norm="ortho"):
        fft_elem = data.shape[axis]
        out = mkl_fft.fft(data, axis=axis)
        if norm == "ortho":
            return out/np.sqrt(fft_elem)
        elif norm == "forward":
            return out/fft_elem
        elif norm == "backward":
            return out
        else:
            raise ValueError

    def ifft(data, axis, norm="ortho"):
        fft_elem = data.shape[axis]
        out = mkl_fft.ifft(data, axis=axis)
        if norm == "ortho":
            return out*np.sqrt(fft_elem)
        elif norm == "forward":
            return out*fft_elem
        elif norm == "backward":
            return out
        else:
            raise ValueError

    def fftn(data, axes, norm="ortho"):
        fft_elem = np.prod(data.shape[axes])
        out = mkl_fft.fftn(data, axes=axes)
        if norm == "ortho":
            return out/np.sqrt(fft_elem)
        elif norm == "forward":
            return out/fft_elem
        elif norm == "backward":
            return out
        else:
            raise ValueError

    def ifftn(data, axes, norm="ortho"):
        fft_elem = np.prod(data.shape[axes])
        out = mkl_fft.ifftn(data, axes=axes)
        if norm == "ortho":
            return out*np.sqrt(fft_elem)
        elif norm == "forward":
            return out*fft_elem
        elif norm == "backward":
            return out
        else:
            raise ValueError
except ImportError:
    from numpy.fft import fft, ifft, fftn, ifftn


def cfftn(data, axes, norm="ortho"):
    """ Centered fast fourier transform, n-dimensional.

    :param data: Complex input data.
    :param axes: Axes along which to shift and transform.
    :return: Fourier transformed data.
    """
    return fftshift(fftn(ifftshift(data, axes=axes), axes=axes, norm=norm), axes=axes)


def cifftn(data, axes, norm="ortho"):
    """ Centered inverse fast fourier transform, n-dimensional.

    :param data: Complex input data.
    :param axes: Axes along which to shift.
    :return: Inverse fourier transformed data.
    """
    return ifftshift(ifftn(fftshift(data, axes=axes), axes=axes, norm=norm), axes=axes)


def cfft(data, axis=-1, norm="ortho"):
    """ Centered fast fourier transform, 1-dimensional.

    :param data: Complex input data.
    :param axis: Axis along which to shift and transform.
    :return: Fourier transformed data.
    """
    return fftshift(fft(ifftshift(data, axes=[axis]), axis=axis, norm=norm), axes=[axis])


def cifft(data, axis=-1, norm="ortho"):
    """ Centered inverse fast fourier transform, 1-dimensional.

    :param data: Complex input data.
    :param axis: Axis along which to shift.
    :return: Inverse fourier transformed data.
    """
    return ifftshift(ifft(fftshift(data, axes=[axis]), axis=axis, norm=norm), axes=[axis])
