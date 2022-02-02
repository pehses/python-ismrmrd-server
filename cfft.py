
try:
    import pyfftw.interfaces.numpy_fft as fft
except ImportError:
    import numpy.fft as fft


def cfftn(data, axes, norm="ortho"):
    """ Centered fast fourier transform, n-dimensional.

    :param data: Complex input data.
    :param axes: Axes along which to shift and transform.
    :return: Fourier transformed data.
    """
    return fft.fftshift(fft.fftn(fft.ifftshift(data, axes=axes), axes=axes, norm=norm), axes=axes)


def cifftn(data, axes, norm="ortho"):
    """ Centered inverse fast fourier transform, n-dimensional.

    :param data: Complex input data.
    :param axes: Axes along which to shift.
    :return: Inverse fourier transformed data.
    """
    return fft.ifftshift(fft.ifftn(fft.fftshift(data, axes=axes), axes=axes, norm=norm), axes=axes)


def cfft(data, axis=-1, norm="ortho"):
    """ Centered fast fourier transform, 1-dimensional.

    :param data: Complex input data.
    :param axis: Axis along which to shift and transform.
    :return: Fourier transformed data.
    """
    return fft.fftshift(fft.fft(fft.ifftshift(data, axes=[axis]), axis=axis, norm=norm), axes=[axis])


def cifft(data, axis=-1, norm="ortho"):
    """ Centered inverse fast fourier transform, 1-dimensional.

    :param data: Complex input data.
    :param axis: Axis along which to shift.
    :return: Inverse fourier transformed data.
    """
    return fft.ifftshift(fft.ifft(fft.fftshift(data, axes=[axis]), axis=axis, norm=norm), axes=[axis])
