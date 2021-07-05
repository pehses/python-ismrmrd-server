#!/usr/bin/env python

from __future__ import division, print_function  # python 2 compatibility

# import os
# import argparse
import numpy as np
# import nibabel as nib
# import matplotlib.pyplot as plt
from numpy.fft import fftshift, ifftshift, ifftn, fftn
import collections


def kspace_to_image(sig, dim=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :returns: data in k-space (along transformed dimensions)
    """
    if dim is None:
        dim = range(sig.ndim)
    elif not isinstance(dim, collections.abc.Iterable):
        dim = [dim]

    sig = ifftshift(sig, axes=dim)
    sig = ifftn(sig, axes=dim)
    sig = ifftshift(sig, axes=dim)

    # sig = fftshift(fftn(fftshift(sig, axes=dim), axes=dim), axes=dim)
    # sig = ifftshift(ifftn(ifftshift(sig, axes=dim), axes=dim), axes=dim)

    return sig


def image_to_kspace(sig, dim=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :returns: data in k-space (along transformed dimensions)
    """
    if dim is None:
        dim = range(sig.ndim)
    elif not isinstance(dim, collections.abc.Iterable):
        dim = [dim]

    sig = fftshift(sig, axes=dim)
    sig = fftn(sig, axes=dim)
    sig = fftshift(sig, axes=dim)

    # sig = ifftshift(ifftn(ifftshift(sig, axes=dim), axes=dim), axes=dim)
    # sig = fftshift(fftn(fftshift(sig, axes=dim), axes=dim), axes=dim)

    return sig


def calc_fa(ste, fid):
    if np.issubdtype(fid.dtype, np.integer):
        ratio = abs(ste) / np.maximum(abs(fid), 1)
    else:  # floating point
        ratio = abs(ste) / np.maximum(abs(fid), np.finfo(fid.dtype).resolution)
    famap = np.rad2deg(np.arctan(np.sqrt(2. * ratio)))
    try:
        famap[famap < 0] = 0.
        famap[np.isnan(famap)] = 0.
        famap[np.isinf(famap)] = 0.
    except:
        pass
    return famap


def approx_sampling(shape, etl, tr=3e-8, dummies=1):

    def genCircularDistance(nz, ny):
        cy = ny // 2
        cz = nz // 2
        y = abs(np.arange(-cy, cy + ny % 2)) / float(cy)
        z = abs(np.arange(-cz, cz + nz % 2)) / float(cz)
        return np.sqrt(y**2 + z[:, np.newaxis]**2)

    ti = genCircularDistance(shape[0], shape[1])**2
    # generate elliptical scanning mask:
    mask = ti > 1
    ti *= etl * tr
    ti += dummies * tr
    ti[mask] = np.nan

    return ti


def DREAM_filter_read(alpha=60., beta=6., tr=3e-3, t1=2., nx=64, etd=1.):
    
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    
    ti = etd * abs(np.linspace(-1, 1, nx, False))**2
    r1s = 1/t1 - np.log(np.cos(beta))/tr
    
    return np.exp(-r1s*ti)[np.newaxis, np.newaxis, :]


def DREAM_filter_fid(alpha=60., beta=6., tr=3e-3, t1=2., ti=None):

    if ti is None:
        ti = tr * np.arange(200)

    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)

    # base signal equation for FID
    Sstst = np.sin(beta) * (1 - np.exp(-tr / t1)) / (1 - np.cos(beta) * np.exp(-tr / t1))
    S0 = np.sin(beta) * np.cos(alpha)**2
    
    r1s = 1/t1 - np.log(np.cos(beta))/tr
    # filt = 1/(1 + Sstst/S0*(np.exp(r1s*ti)-1))
    filt = S0/(S0 + Sstst*(np.exp(r1s*ti)-1.))
    if filt is None:
        filt = 0
    #filt[np.isnan(ti)] = 0
    print(filt)
    
    return filt


def applyFilter(sig, filt, axes=[0, 1], back_transform=True):
    # ifft in lin & par spatial dims
    sig = image_to_kspace(sig, axes)

    while np.ndim(filt) < np.ndim(sig):
        filt = filt[..., np.newaxis]
        
    # multiply with filter
    sig *= filt

    # fft in lin & par spatial dims
    if back_transform:
        sig = kspace_to_image(sig, axes)

    return sig

def global_filter(ste, fid, ti, alpha=60, beta=6, tr=3e-3, t1=2., blur_read=False):
    nz, ny = ste.shape[:2]
    mean_alpha = calc_fa(ste.mean(), fid.mean())
    mean_beta = mean_alpha / alpha * beta
    filt = DREAM_filter_fid(mean_alpha, mean_beta, tr, t1, ti)
    fid = applyFilter(fid, filt)
    
    if blur_read and ste.ndim > 2:
        nx = ste.shape[2]
        etd = np.nanmax(ti)
        filt_ro = DREAM_filter_read(alpha=mean_alpha, beta=mean_beta, tr=tr, t1=t1, nx=nx, etd=etd)
        fid = applyFilter(fid, filt_ro, axes=[2])
        ste = applyFilter(ste, filt_ro, axes=[2])

    # filter fid in k-space
    return calc_fa(abs(ste), abs(fid)), abs(fid)

# def local_filter(ste, fid, ti, alpha=60, beta=6, tr=3e-3, t1=2., blur_read=False, fmap=None, nbins=40, niter=2, store_iter = False):

#     def iteration(fmap, fid):

#         scale = fmap / alpha

#         # create nbins
#         edges = np.linspace(0., np.nanmax(scale), nbins+1)[1:]
#         inds = np.digitize(scale, edges, right=True)
        
#         fidnew = np.zeros(fid.shape, dtype=np.complex128)
#         for key, item in enumerate(edges):

#             # 1) obtain normalized signal evolution
#             alpha_ = item * alpha
#             beta_ = item * beta
#             filt = DREAM_filter_fid(alpha_, beta_, tr, t1, ti)

#             # 2) select all voxels in bin
#             sig = np.zeros_like(fid)
#             sig[inds == key] = fid[inds == key]

#             # 3) apply filter for current bin
#             # & add bin to solution
#             # fidnew += applyFilter(sig, filt, back_transform=False)
#             fidnew += applyFilter(sig, filt, back_transform=True)

#         # return abs(kspace_to_image(fidnew))
#         return abs(fidnew)

#     if blur_read and ste.ndim > 2:
#         nx = ste.shape[2]
#         etd = np.nanmax(ti)
#         mean_alpha = calc_fa(ste.mean(), fid.mean())
#         mean_beta = mean_alpha / alpha * beta
#         filt_ro = DREAM_filter_read(alpha=mean_alpha, beta=mean_beta, tr=tr, t1=t1, nx=nx, etd=etd)
#         fid = applyFilter(fid, filt_ro, axes=[2])
#         ste = applyFilter(ste, filt_ro, axes=[2])

#     if fmap is None:
#         fmap = calc_fa(ste, fid)

#     if store_iter:
#         fmap_iter = np.zeros(np.insert(fmap.shape, fmap.ndim, niter+1), dtype=fmap.dtype)
#         print(fmap.shape, fmap_iter.shape)
#         fmap_iter[..., 0] = fmap

#     for i in range(niter):
#         fidnew = iteration(fmap, fid)
#         fmap = calc_fa(ste, fidnew)
#         if store_iter:
#             fmap_iter[..., i+1] = fmap

#     if store_iter:
#         fmap = fmap_iter

#     return fmap, fidnew
