try:
    import pyfftw
    from pyfftw.interfaces.numpy_fft import fft, ifft, fftn, ifftn, fftshift, ifftshift
    import multiprocessing
    # Configure PyFFTW to use all cores (the default is single-threaded)
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
except ImportError:
    from numpy.fft import fft, ifft, fftn, ifftn, fftshift, ifftshift


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
