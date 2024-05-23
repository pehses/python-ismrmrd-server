""" Helper functions for reconstruction
"""

import numpy as np
import logging
import time
import os
import tempfile

from scipy.ndimage import  median_filter, gaussian_filter
from scipy.spatial import KDTree
from skimage.transform import resize
from skimage.restoration import unwrap_phase, denoise_nl_means, estimate_sigma
import despike
import nibabel as nib

from multiprocessing import Pool
import psutil

# Root sum of squares

def rss(img, axis=-1):
    # root sum of squares along a given axis
    return np.sqrt(np.sum(np.abs(img)**2, axis=axis))

## Noise-prewhitening

def calculate_prewhitening(noise, scale_factor=1.):
    '''Calculates the noise prewhitening matrix

    :param noise: Input noise data (2D array), ``[coil, nsamples]``
    :scale_factor: Applied on the noise covariance matrix. Used to
                   adjust for effective noise bandwith and difference in
                   sampling rate between noise calibration and actual measurement:
                   scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio

    :returns w: Prewhitening matrix, ``[coil, coil]``, w @ data is prewhitened
    '''
    dmtx = (noise @ np.conj(noise).T)/(noise.shape[1]-1)
    dmtx = np.linalg.inv(np.linalg.cholesky(dmtx))
    dmtx *= np.sqrt(2)*np.sqrt(scale_factor)
    return dmtx

def apply_prewhitening(data, dmtx):
    '''Apply the noise prewhitening matrix

    :param data: Input data (array or matrix), ``[coil, ...]``
    :param dmtx: Input noise prewhitening matrix

    :returns w_data: Prewhitened data, ``[coil, ...]``,
    '''
    return dmtx @ data

## Coil compression

def calibrate_scc(data):
    # input data: [ncha, nsamples]
    # output: [ncha, ncha]
    U, s, _ = np.linalg.svd(data, full_matrices=False)
    mtx = np.conj(U.T)
    return mtx, s

def calibrate_cc(items, ncc, apply_cc=True):
    """ Calibrate coil compression matrix

    items: list of ISMRMRD acquisitions
    ncc: number of virtual coils
    """
    
    if ncc == 0 or ncc >= items[0].data.shape[0]:
        return None  # nothing to do

    data = np.asarray([acq.data for acq in items])
    nc = data.shape[1]
    cc_data = np.moveaxis(data, 1, 0).reshape([nc, -1])
    subsamples = min(32768, cc_data.shape[1])
    choice = np.random.choice(cc_data.shape[1], subsamples)
    cc_data = cc_data[:,choice]
    cc_matrix, s = calibrate_scc(cc_data)
    cc_matrix = cc_matrix[:ncc, :]

    # apply coil compression
    if apply_cc:
        data = cc_matrix @ data

        # write data back to acsGroup:
        for acq, dat in zip(items, data):
            acq.resize(number_of_samples=dat.shape[-1], active_channels=dat.shape[0], trajectory_dimensions=acq.trajectory_dimensions)
            acq.data[:] = dat

    return cc_matrix

def apply_cc(item, cc_matrix):
    """ Apply coil compression matrix

    item: ISMRMRD acquisition
    cc_matrix: coil compression matrix [nc_new,nc_old]
    """
    if cc_matrix is None:
        return

    data = cc_matrix @ item.data

    # store modified data
    item.resize(number_of_samples=data.shape[-1], active_channels=data.shape[0], trajectory_dimensions=item.trajectory_dimensions)
    item.data[:] = data

## Oversampling removal
def remove_os(data, axis=0):
    '''Remove oversampling (assumes os factor 2)
    '''
    cut = slice(data.shape[axis]//4, (data.shape[axis]*3)//4)
    data = np.fft.ifft(data, axis=axis)
    data = np.delete(data, cut, axis=axis)
    data = np.fft.fft(data, axis=axis)
    return data

def remove_os_spiral(acq):
    '''Remove oversampling for spiral data (assumes os factor 2)
    '''

    # cut data
    data = acq.data[:]
    cut = slice(data.shape[1]//4, (data.shape[1]*3)//4)
    data = np.fft.ifft(data, axis=1)
    data = np.delete(data, cut, axis=1)
    data = np.fft.fft(data, axis=1)

    # delete every 2nd point in trajectory
    traj = acq.traj[:]
    traj = traj[::2]

    acq.resize(number_of_samples=data.shape[1], trajectory_dimensions=acq.trajectory_dimensions, active_channels=acq.active_channels)
    acq.data[:] = data
    acq.traj[:] = traj

## Rotations

def calc_rotmat(acq):
        phase_dir = np.asarray(acq.phase_dir)
        read_dir = np.asarray(acq.read_dir)
        slice_dir = np.asarray(acq.slice_dir)
        return np.round(np.concatenate([phase_dir[:,np.newaxis], read_dir[:,np.newaxis], slice_dir[:,np.newaxis]], axis=1), 6)

def pcs_to_dcs(coord, patient_position='HFS'):
    """ Convert from patient coordinate system (PCS, physical) 
        to device coordinate system (DCS, physical)
        this is valid for patient orientation head first/supine
    """
    coord = coord.copy()

    # only valid for head first/supine - other orientations see IDEA UserGuide
    if patient_position.upper() == 'HFS':
        coord[1] *= -1
        coord[2] *= -1
    else:
        raise ValueError

    return coord

def dcs_to_pcs(coord, patient_position='HFS'):
    """ Convert from device coordinate system (DCS, physical) 
        to patient coordinate system (PCS, physical)
        this is valid for patient orientation head first/supine
    """
    return pcs_to_dcs(coord, patient_position) # same sign switch
    
def gcs_to_pcs(coord, rotmat):
    """ Convert from gradient coordinate system (GCS, logical) 
        to patient coordinate system (DCS, physical)
    """
    return np.matmul(rotmat, coord)

def pcs_to_gcs(coord, rotmat):
    """ Convert from patient coordinate system (PCS, physical) 
        to gradient coordinate system (GCS, logical) 
    """
    return np.matmul(np.linalg.inv(rotmat), coord)

def gcs_to_dcs(coord, rotmat):
    """ Convert from gradient coordinate system (GCS, logical) 
        to device coordinate system (DCS, physical)
        this is valid for patient orientation head first/supine
    Parameters
    ----------
    coord : numpy array [3, samples]
            gradient to be converted
    rotmat: numpy array [3,3]
            rotation matrix from Siemens Raw Data header or ISMRMRD acquisition header
    Returns
    -------
    coord_cv : numpy.ndarray
               Converted gradient
    """
    coord = coord.copy()

    # rotation from GCS (PHASE,READ,SLICE) to patient coordinate system (PCS)
    coord = gcs_to_pcs(coord, rotmat)
    
    # PCS (SAG,COR,TRA) to DCS (X,Y,Z)
    # only valid for head first/supine - other orientations see IDEA UserGuide
    coord = pcs_to_dcs(coord)
    
    return coord


def dcs_to_gcs(coord, rotmat):
    """ Convert from device coordinate system (DCS, logical) 
        to gradient coordinate system (GCS, physical)
        this is valid for patient orientation head first/supine
    Parameters
    ----------
    coord : numpy array [3, samples]
            gradient to be converted
    rotmat: numpy array [3,3]
            rotation matrix from Siemens Raw Data header or ISMRMRD acquisition header
    Returns
    -------
    coord_cv : numpy.ndarray
               Converted gradient
    """
    coord = coord.copy()
    
    # DCS (X,Y,Z) to PCS (SAG,COR,TRA)
    # only valid for head first/supine - other orientations see IDEA UserGuide
    coord = dcs_to_pcs(coord)
    
    # PCS (SAG,COR,TRA) to GCS (PHASE,READ,SLICE)
    coord = pcs_to_gcs(coord, rotmat)
    
    return coord

def dcs_to_ras(coord, patient_position='HFS'):
    """
    Transform 3D coordinate from device coordinate systen to RAS+/Nifti coordinate system
    DCS: left+, anterior+, feet+
    RAS: right+, anterior+, head+

    This is only valid for head first/supine (HFS) orientation
    """

    if patient_position.upper() == 'HFS':
        coord[0] *= -1
        coord[2] *= -1
    else:
        raise ValueError
        
    return coord

def pcs_to_ras(coord, patient_position='HFS'):
    """
    Transform 3D coordinate from patient coordinate systen to RAS+/Nifti coordinate system
    PCS: left+, posterior+, head+
    RAS: right+, anterior+, head+

    This is only valid for head first/supine (HFS) orientation
    """
    coord = pcs_to_dcs(coord, patient_position)
    return  dcs_to_ras(coord, patient_position)

## FOV shifts

def fov_shift(sig, shift):
    """ Performs inplane fov shift for Cartesian data of shape [nx,ny,nz,nc]
    """
    fac_x = np.exp(-1j*shift[0]*2*np.pi*np.arange(sig.shape[0])/sig.shape[0])
    fac_y = np.exp(-1j*shift[1]*2*np.pi*np.arange(sig.shape[1])/sig.shape[1])

    sig = (sig.T * fac_x).T
    sig = (sig.T * fac_y[:,np.newaxis]).T
    return sig

def fov_shift_spiral(sig, trj, shift, matr_sz):
    """ 
    Shift the field of view of spiral data

    sig: raw data [ncha, nsamples]
    trj: trajectory [3, nsamples]
    shift: shift [x_shift, y_shift] in voxel
    matr_sz: matrix size of reco
    """

    if (abs(shift[0]) < 1e-2) and (abs(shift[1]) < 1e-2):
        # nothing to do
        return sig

    kmax = matr_sz/2
    sig *= np.exp(-1j*(shift[0]*np.pi*trj[0]/kmax+shift[1]*np.pi*trj[1]/kmax))[np.newaxis]

    return sig

def fov_shift_spiral_reapply(sig, pred_trj, base_trj, shift, matr_sz):
    """ 
    Re-apply FOV shift on spiral/noncartesian data, when FOV positioning in the Pulseq sequence is enabled
    first undo field of view shift with nominal, then reapply with predicted trajectory

    IMPORTANT: The nominal trajectory has to be shifted by -10us as the ADC frequency adjustment
               of the scanner is lagging behind be one gradient raster time (10 us).
               For Pulseq sequences this is done in pulseq_helper.py

    sig: signal data (dimensions as in ISMRMRD [coils, samples]) 
    pred_traj: predicted trajectory (dimensions as in ISMRMRD [samples, dims]) 
    base_trj: nominal trajectory (dimensions as in ISMRMRD [samples, dims])
    shift: shift [x_shift, y_shift] in voxel
    matr_sz: matrix size [x,y]
    """

    pred_trj = pred_trj[:,:2]
    base_trj = base_trj[:,:2]

    if (abs(shift[0]) < 1e-2) and (abs(shift[1]) < 1e-2):
        # nothing to do
        return sig

    kmax = (matr_sz/2+0.5).astype(np.int32)[:2]
    shift = shift[:2]

    # undo FOV shift from nominal traj
    sig *= np.exp(1j*np.pi*(shift*base_trj/kmax).sum(axis=-1))

    # redo FOV shift with predicted traj
    sig *= np.exp(-1j*np.pi*(shift*pred_trj/kmax).sum(axis=-1))

    return sig

## K-space filter

def filt_ksp(kspace, traj, filt_fac=0.95):
    """filter outer kspace data with hanning filter to avoid Gibbs Ringing

    kspace:   kspace data [coils, samples]
    traj:     k-space trajectory [dim, samples]

    filt_fac: filtering is done after filt_fac * max_radius is reached, 
              where max_radius is the maximum radius of the spiral
    """

    filt = np.sqrt(traj[0]**2+traj[1]**2) # trajectory radius
    filt[filt < filt_fac*np.max(filt)] = 1
    filt[filt >= filt_fac*np.max(filt)] = -1
    filt_len = len(filt[filt==-1])
    if filt_len%2 == 1:
        filt_len += 1

    # filter outer part of kspace
    if filt[0] == -1 and filt[-1] == -1: # both ends (e.g. double spiral)
        filt[:filt_len//2] = np.hamming(filt_len)[:filt_len//2]
        filt[-filt_len//2:] = np.hamming(filt_len)[filt_len//2:]
    elif filt[0] == -1: # beginning (e.g. spiral in)
        filt[:filt_len] = np.hamming(2*filt_len)[:filt_len]
    elif filt[-1] == -1: # end (e.g. spiral out)
        filt[-filt_len:] = np.hamming(2*filt_len)[filt_len:]

    return kspace * filt
    

## Array manipulation

def intp_axis(newgrid, oldgrid, data, axis=0):
    # interpolation along an axis (shape of newgrid, oldgrid and data see np.interp)
    tmp = np.moveaxis(data.copy(), axis, 0)
    newshape = (len(newgrid),) + tmp.shape[1:]
    tmp = tmp.reshape((len(oldgrid), -1))
    n_elem = tmp.shape[-1]
    intp_data = np.zeros((len(newgrid), n_elem), dtype=data.dtype)
    for k in range(n_elem):
        intp_data[:, k] = np.interp(newgrid, oldgrid, tmp[:, k])
    intp_data = intp_data.reshape(newshape)
    intp_data = np.moveaxis(intp_data, 0, axis)
    return intp_data 

def add_naxes(arr, n):
    """ Adds n empty dimensions to the end of an array
    """
    for k in range(n):
        arr = arr[...,np.newaxis]
    return arr

# Image coordinates

def calc_img_coord(metadata, acq, pulseq=True):
    """
    Calculate voxel coordinates for a given slice for use in higher order reconstructions.
    The coordinate system is the Siemens device coordinate system (DCS), as the Skope data is also acquired in that coordinate system:
    x: neg -> pos is right -> left
    y: neg -> pos is posterior -> anterior
    z: neg -> pos is head -> feet
    """

    # matrix size & rotmat
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2) # sms factor
    rotmat =  calc_rotmat(acq)

    # scaling
    res = 1e-3 * metadata.encoding[0].encodedSpace.fieldOfView_mm.x / nx
    slc_res = 1e-3 * metadata.encoding[0].encodedSpace.fieldOfView_mm.z

    # Slice separation for multiband imaging
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_slc_red = n_slc // nz
    slc_sep = n_slc_red * slc_res

    # slice offsets have to be calculated differently for the Pulseq sequence and the Siemens EPI sequence
    # as in the Siemens EPI, the center position of the respective slice is stored in the position field, 
    # whereas in Pulseq only the center of the whole volume (global offset is stored)
    if pulseq:
        slice_offset_gcs = slc_res*(acq.idx.slice-(n_slc-1)/2) # this is the offset of the first slice in a stack from the volumes center
        ix = np.linspace(nx/2*res,-(nx/2-1)*res, nx)
        iy = np.linspace(ny/2*res,-(ny/2-1)*res, ny)
        iz = np.linspace(0, (nz-1)*slc_sep, nz) + slice_offset_gcs
    else:
        slice_offset_gcs = 0 # put the slice center at (0,0,0) in GCS, as slice offsets will be applied later
        ix = np.linspace((nx/2-1)*res,-nx/2*res, nx) # ix is shifted by 1 voxel in the EPI (confirmed with affine from scanner)
        iy = np.linspace(ny/2*res,-(ny/2-1)*res, ny)
        iz = np.linspace(0, (nz-1)*slc_sep, nz) + slice_offset_gcs
    
    # Make grid in GCS (logical)
    grid = np.asarray(np.meshgrid(ix,iy,iz)).reshape([3,-1])

    # Coordinates to DCS (physical)
    grid_rot = gcs_to_dcs(grid, rotmat) # [dims, coords]

    # Add slice offset in DCS
    slice_offset_pcs = 1e-3 * pcs_to_dcs(np.asarray(acq.position)) # [mm] -> [m]
    grid_rot[0] += slice_offset_pcs[0]
    grid_rot[1] += slice_offset_pcs[1]
    grid_rot[2] += slice_offset_pcs[2]

    # reshape back
    grid_rot = grid_rot.reshape([3,len(ix),len(iy),len(iz)])

    return grid_rot

# WIP: Nifti affine from coordinates

def calc_affine(res, rotmat, refpt):
    """
    Calculate affine matrix from:
    res: voxel size [mm]
    rotmat: rotation matrix from logical (GCS) to patient coordinate system (PCS)
    refpt: reference point for nifti images at [right, posterior, feet] edge in transversal orientation [mm] in device coordinate system (DCS)

    refpt can be taken from image coordinates used for recon (see function "calc_img_coord")

    IMPORTANT: EXPERIMENTAL! The affine was only tested to be correct for base orientation transversal! In other orientation it might need to be shifted/sign-switched
    """
    affine = np.eye(4)
    
    # voxel size [mm]
    affine[0,0] = -1*res[0]
    affine[1,1] = res[1]
    affine[2,2] = -1*res[2]

    # rotation
    affine[:3,:3] = gcs_to_pcs(affine[:3,:3], rotmat)
    affine[:3,:3] = pcs_to_ras(affine[:3,:3])

    # reference point
    refpt = dcs_to_ras(refpt) # DCS to RAS+
    affine[:3,-1] = refpt

    return affine

# DCF
def calc_dcf(traj):
    """ Estimates the density compensation function for a given k-space trajectory
        taken from https://github.com/TardifLab/ESM_image_reconstruction/blob/main/general/cg_dcf_spiral.m
    """
    theta = np.arctan2(traj[:,0],traj[:,1])
    theta = np.unwrap(theta)
    dcf = theta[1:] - theta[:-1]
    dcf = np.append(dcf,dcf[-1]) * np.sqrt(traj[:,0]**2+traj[:,1]**2)
    return dcf

######################################
#### Ecalib (3D is done in chunks)
######################################

def ecaltwo(gpu_str, n_maps, nx, ny, sig, crop=0.8):
    from bart import bart
    maps = bart(1, f'ecaltwo {gpu_str} -c {crop} -m {n_maps} {nx} {ny} {sig.shape[2]}', sig)
    return np.moveaxis(maps, 2, 0)  # slice dim first since we need to concatenate it in the next step

def ecalib(acs, n_maps=1, crop=0.8, threshold=0.001, threads=8, kernel_size=6, chunk_sz=None, use_gpu=False):
    """
    Run parallel imaging calibration with ESPIRiT

    acs: ACS data (4D array [nx,ny,nz,nc] or 5D array [nx,ny,nz,nc,n_slc]), if 5D calculation will be parallelized on last dim
    n_maps: number of ESPRIT maps
    crop: crop factor for ESPIRiT
    threshold: threshold for ESPIRiT
    threads: number of threads for parallel processing, if 3D data is chunked
    kernel_size: kernel size for ESPIRiT
    chunk_sz: chunk size for 3D data, if None 3D data will be processed in one run
    use_gpu: use GPU for ESPIRiT
    """


    from multiprocessing import Pool
    from functools import partial
    from bart import bart

    logging.debug("Start sensitivity map calculation.")
    start = time.perf_counter()

    gpu_str = "-g" if use_gpu else ""

    ndim = acs.ndim
    nx, ny, nz, nc = acs.shape[:4]
    if chunk_sz is None:
        # this should usually work
        gpu_mem = 48 * 1024**3  # bytes
        chunk_sz = gpu_mem / (8*nx*ny*nc*nc*n_maps)
        if threads is not None and threads > 1:
            chunk_sz /= threads
        chunk_sz /= 2  # reserve some memory for overhead
        chunk_sz = 2 * (chunk_sz//2)  # round down to multiple of 2
        chunk_sz = int(max(1, chunk_sz))

    if chunk_sz <= 0 or chunk_sz >= nz:
        # espirit in one run:
        ecal_str = f'ecalib -k {kernel_size} -I {gpu_str} -m{n_maps} -c{crop} -t {threshold}'
        if ndim == 5 and acs.shape[-1] > 1:
            ecal_str = f'--parallel-loop {(acs.ndim-1)**2} -e {acs.shape[-1]} ' + ecal_str
            sensmaps = bart(1, ecal_str, acs)
        else:
            sensmaps = bart(1, ecal_str, acs)
    else:
        # espirit_econ: reduce memory footprint by chunking
        eon = bart(1, f'ecalib -k {kernel_size} -I {gpu_str} -m {n_maps} -t {threshold} -1', acs)  # currently, gpu doesn't help here but try anyway
        # use norms 'forward'/'backward' for consistent scaling with bart's espirit_econ.sh
        # scaling is very important for proper masking in ecaltwo!
        tic = time.perf_counter()
        eon = np.fft.ifft(eon, axis=2, norm='forward')
        tmp = np.zeros(eon.shape[:2] + (nz-eon.shape[2],) + eon.shape[-1:], dtype=eon.dtype)
        cutpos = eon.shape[2]//2
        eon = np.concatenate((eon[:, :, :cutpos, :], tmp, eon[:, :, cutpos:, :]), axis=2)
        eon = np.fft.fft(eon, axis=2, norm='backward')
        logging.info("FFT interpolation processing time: %.2f s" % (time.perf_counter()-tic))

        tic = time.perf_counter()
        logging.info(f"loop: 'bart ecaltwo {gpu_str} -c {crop} -m {n_maps} {nx} {ny} {chunk_sz}' with {threads} threads")

        slcs = (slice(i, i+chunk_sz) for i in range(0, nz, chunk_sz))
        chunks = (eon[:, :, sl] for sl in slcs)

        if threads is None or threads < 2:
            sensmaps = [ecaltwo(gpu_str, n_maps, nx, ny, sig, crop=crop) for sig in chunks]
        else:
            # WIP: BART has its own parallel looping/MPI since v0.9.00
            with Pool(threads) as p:
                sensmaps = p.map(partial(ecaltwo, gpu_str, n_maps, nx, ny, crop=crop), chunks)

        sensmaps = np.concatenate(sensmaps, axis=0)
        sensmaps = np.moveaxis(sensmaps, 0, 2)
        logging.info(f"ecalib with chunk_sz={chunk_sz} and {threads} thread(s): {time.perf_counter()-tic} s")

    while sensmaps.ndim < ndim:
        sensmaps = sensmaps[..., np.newaxis]

    logging.debug(f"Finished sensitivity map calculation after {time.perf_counter()-start:.2f} s.")

    return sensmaps

#############################
# Field map calculation
#############################

def calc_fmap(imgs, echo_times, metadata, online_recon=False):
    """ Calculate field maps from reference images with two different contrasts

        imgs: [slices,nx,ny,nz,nc,n_contr]
        echo_times: list of echo times [s]

        always returns field map in dimensions [slices/nz, nx, ny]
    """
    
    mc_fmap = True # calculate multi-coil field maps to remove outliers (Robinson, MRM. 2011) - recommended
    despike_filter = True # apply despiking
    median_filtering = False # apply median filtering
    gaussian_filtering = True # apply Gaussian filtering
    nlm_filter = False # apply non-local means filter to field map in the end
    std_filter = False # apply standard deviation filter (only if mc_fmap selected)
    std_fac = 1.5 # factor for standard deviation denoising (see below)
    romeo_fmap = False # use the ROMEO toolbox for field map calculation (set to True, if more than 2 echoes)
    romeo_uw = False # use ROMEO only for unwrapping (slower than unwrapping with skimage)

    cores = psutil.cpu_count(logical = False)

    logging.info("Starting field map calculation.")

    if online_recon:
        std_filter = False

    if len(echo_times) > 2:
        romeo_fmap = True

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    n_slc = imgs.shape[0]
    nc = imgs.shape[-2]
    if nc == 1:
        mc_fmap = False
        std_filter = False

    # from [slices,nx,ny,nz,coils,echoes] to either [slices,nx,ny,coils,echoes] or [nz,nx,ny,coils,echoes]
    if nz == 1:
        imgs = imgs[:,:,:,0] # 2D field map acquisition
    elif nz > 1:
        imgs = np.moveaxis(imgs[0],2,0) # 3D field map acquisition
    elif nz > 1 and n_slc > 1:
        raise ValueError("Multi-slab is not supported.")

    # mask with threshold as accuracy of the field map values is proportional to the signal intensity
    img_mask = rss(imgs[...,-1], axis=-1)
    mask = img_mask/np.max(img_mask)
    thresh = 0.2 * np.percentile(mask, 95)
    mask[mask<thresh] = 0
    mask[mask>=thresh] = 1
    nifti = nib.Nifti1Image(np.flip(np.transpose(mask,[1,2,0]), (0,1,2)), np.eye(4)) # save mask for easier debugging
    nib.save(nifti, "/tmp/share/debug/mask.nii.gz")

    if romeo_fmap:
        # ROMEO unwrapping and field map calculation (Dymerska, MRM, 2020)
        fmap = romeo_unwrap(imgs, echo_times, metadata, mc_unwrap=False, return_b0=True)
    elif mc_fmap:
        # Multi-coil field map calculation (Robinson, MRM, 2011)
        phasediff = imgs[...,1] * np.conj(imgs[...,0])
        if romeo_uw:
            phasediff_uw = romeo_unwrap(phasediff,[], mc_unwrap=True, return_b0=False)
        else:
            phasediff_uw = np.zeros_like(phasediff,dtype=np.float64)
            pool = Pool(processes=cores)
            results = [pool.apply_async(do_unwrap_phase, [phasediff[...,k]]) for k in range(phasediff.shape[-1])]
            for k, val in enumerate(results):                
                phasediff_uw[...,k] = val.get()
            pool.close()

        phasediff_uw_rs = phasediff_uw.reshape([-1,nc])
        img_mag = abs(imgs[...,0]).reshape([-1,nc])
        ix = np.argsort(phasediff_uw_rs, axis=-1)[:,nc//4:-nc//4] # remove lowest & highest quartile
        weights = np.take_along_axis(img_mag, ix, axis=-1) / np.sum(np.take_along_axis(img_mag, ix, axis=-1), axis=-1)[:,np.newaxis]
        fmap = np.sum(weights * np.take_along_axis(phasediff_uw_rs, ix, axis=-1), axis=-1)
        fmap = fmap.reshape(imgs.shape[:3])
        te_diff = echo_times[1] - echo_times[0]
        fmap = -1 * fmap/te_diff # the sign in Powergrid is inverted

        # Standard deviation filter (Robinson, MRM, 2011)
        if std_filter:
            std = phasediff_uw.std(axis=-1)
            median_std = np.median(std[std!=0])
            idx = np.argwhere(std>std_fac*median_std)
            n_cb = [2,2,1]
            fmap2 = fmap.copy()
            for ix in idx:
                voxel_cb = tuple([slice(max(0,ix[k]-n_cb[k]), ix[k]+n_cb[k]+1) for k in range(3)])
                fmap2[tuple(ix)] = np.median((fmap[voxel_cb])[np.nonzero(fmap[voxel_cb])])
            fmap = fmap2
    else:
        # Standard field mapping approach (Hermitian product & SOS coil combination)
        phasediff = imgs[...,1] * np.conj(imgs[...,0]) 
        phasediff = np.sum(phasediff, axis=-1) # coil combination
        if romeo_uw:
            phasediff_uw = romeo_unwrap(phasediff, [], mc_unwrap=False, return_b0=False)
        else:
            phasediff_uw = unwrap_phase(np.angle(phasediff))
        te_diff = echo_times[1] - echo_times[0]
        fmap = -1 * phasediff_uw/te_diff # the sign in Powergrid is inverted

    # fill all voxels outside the mask by the weighted mean of their 'k' nearest neighbors inside the mask
    fmap = fill_masked_voxels(fmap, mask, k=10)

    # Despike filter
    if despike_filter:
        pool = Pool(processes=cores)
        results = [pool.apply_async(do_despike, [fmap[k]]) for k in range(len(fmap))]
        for k, val in enumerate(results):
            fmap[k] = val.get()
        pool.close()

    # Gauss/median filter
    if gaussian_filtering:
        fmap = gaussian_filter(fmap, sigma=0.5)
    if median_filtering:
        fmap = median_filter(fmap, size=2)

    # interpolate to correct matrix size
    if nz == 1:
        newshape = [n_slc,ny,nx]
    else:
        newshape = [nz,ny,nx]
    fmap = resize(np.transpose(fmap,[0,2,1]), newshape, anti_aliasing=True)
    
    if nlm_filter:
        sigma_est = np.mean(estimate_sigma(fmap))
        patch_kw = dict(patch_size=3, patch_distance=6)
        fmap = denoise_nl_means(fmap, h=0.4*sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)

    logging.info("Field map calculation finished.")

    return fmap

def fill_masked_voxels(input_array, mask, k=10):
    """
    Replace values outside of the mask by the weighted mean of k nearest neighbors inside the mask.
    Weights are inversely proportional to the distance.

    input_array: 3D numpy array
        The input array containing values to be filled.
    mask: 3D numpy array
        A binary mask where 1 indicates the region with 'valid' values and 0 indicates the region with 'missing' values.
    k: int
        The number of nearest neighbors to consider for the weighted mean.
    """

    mask = mask.astype(bool)

    # Get indices of points inside the mask
    inside_mask_indices = np.argwhere(mask)

    # Build KDTree for points inside the mask
    tree = KDTree(inside_mask_indices)

    # Get indices of points outside the mask
    outside_mask_indices = np.argwhere(~mask)

    # Query the KDTree to find k nearest neighbors within the mask for all points outside the mask
    distances, ind = tree.query(outside_mask_indices, k=k)

    # Get the coordinates of the k nearest neighbors for each point outside the mask
    neighbor_coords = inside_mask_indices[ind]

    # Calculate weights as the inverse of the distances
    weights = 1 / (distances + 1e-8)  # Adding a small value to avoid division by zero

    # Normalize weights so they sum to 1
    weights /= weights.sum(axis=1, keepdims=True)

    # Get the values of the k nearest neighbors
    neighbor_values = input_array[neighbor_coords[:, :, 0], neighbor_coords[:, :, 1], neighbor_coords[:, :, 2]]

    # Calculate the weighted mean values for the k nearest neighbors
    weighted_mean_values = np.sum(neighbor_values * weights, axis=1)

    # Create an array to store results
    result_array = input_array.copy()

    # Assign the calculated weighted mean values to the corresponding points outside the mask
    result_array[outside_mask_indices[:, 0], outside_mask_indices[:, 1], outside_mask_indices[:, 2]] = weighted_mean_values

    return result_array

def load_external_fmap(path, shape):
    # Load an external field map (has to be a .npz file)
    if not os.path.exists(path):
        fmap = {'fmap': np.zeros(shape), 'name': 'No Field Map'}
        logging.debug("No field map file in dependency folder. Use zeros array instead. Field map should be .npz file.")
    else:
        fmap = dict(np.load(path, allow_pickle=True))
        if 'name' not in fmap:
            fmap['name'] = 'No name.'
        if 'fmap' not in fmap:
            logging.debug(f"No field map data found in external field map. Set field map to zero.")
            fmap = {'fmap': np.zeros(shape), 'name': 'No Field Map'}
    if shape != list(fmap['fmap'].shape):
        logging.debug(f"Field Map dimensions do not fit. Fmap shape: {list(fmap['fmap'].shape)}, Img Shape: {shape}. Try interpolating the field map.")
        fmap['fmap'] = resize(fmap['fmap'], shape, anti_aliasing=True)
        logging.debug(f"Field Map shape after interpolation: {list(fmap['fmap'].shape)}.")
    if 'params' in fmap:
        logging.debug("Field Map regularisation parameters: %s",  fmap['params'].item())

    return fmap

def do_unwrap_phase(phasediff):
    return unwrap_phase(np.angle(phasediff))

def do_despike(fmap):
    return despike.clean(fmap, n=0.8, size=2, mask='mean', fill_method='median', fill_size=2)

# Unwrapping with ROME0
def romeo_unwrap(imgs, echo_times, metadata, mask=None, mc_unwrap=False, return_b0=False):
    """
        Do phase unwrapping with romeo and optionally output B0 map (Dymerska, MRM, 2020)

        imgs: Input (multi-coil, multi-echo) images [x,y,z, coils, echoes] (coils & echoes are optional)
        echo_times: list of echo times [ms]
        metadata: ISMRMRD metadata
        mask: mask for unwrapping
        mc_unwrap: unwrap each channel individually (coil dimension has to be present)
        return_b0: return B0 map in rad/m (only if mc_unwrap=False)
    """

    import nibabel as nib
    import subprocess

    if mc_unwrap and imgs.ndim<4:
        raise ValueError("No multi-coil data available.")
    if len(echo_times) > 0:
        echo_times *= 1e3
    if imgs.ndim == 5:
        coildim = -2
        echodim = -1
        imgs = np.swapaxes(imgs, echodim, coildim) # romeo needs [x,y,z,echoes,coils]
        if imgs.shape[-1] == 1:
            imgs = imgs[...,0] # remove coil dimension if 1
    else:
        coildim = -1

    # tempfile.tempdir = "/dev/shm"
    tmpdir = tempfile.TemporaryDirectory()
    tempdir = tmpdir.name
    # tempdir = "/tmp/share/debug/"

    # set affine for nifti
    res_x = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x
    res_y = metadata.encoding[0].encodedSpace.fieldOfView_mm.y / metadata.encoding[0].encodedSpace.matrixSize.y
    res_z = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    affine = np.diag([res_x, res_y, res_z, 1])

    # check for bipolar phase correction
    up_base_prot = {item.name: item.value for item in metadata.userParameters.userParameterBase64}
    bipolar = False
    if 'bipolar' in up_base_prot and int(up_base_prot['bipolar']):
        logging.debug("Bipolar phase offset correction.")
        bipolar = True

    # write Niftis
    phs_in = np.angle(imgs)
    mag_in = abs(imgs)
    phs_name = tempdir+"/phs_romeo.nii.gz"
    mag_name = tempdir+"/mag_romeo.nii.gz"
    mag_romeo = nib.Nifti1Image(mag_in, affine)
    nib.save(mag_romeo, mag_name)

    mask_name = "robustmask"
    if mask is not None:
        mask_name = tempdir+"/mask_romeo.nii.gz"
        mask_romeo = nib.Nifti1Image(mask, affine)
        nib.save(mask_romeo, mask_name)

    subproc = f"romeo -p {phs_name} -m {mag_name} -k {mask_name} -t {echo_times} -o {tempdir} --temporal-uncertain-unwrapping"
    if bipolar:
        subproc += " --phase-offset-correction bipolar"

    if mc_unwrap:
        phs_uw = []
        for k in range(phs_in.shape[-1]):
            phs_romeo = nib.Nifti1Image(phs_in[...,k], affine)
            nib.save(phs_romeo, phs_name)
            process = subprocess.run(subproc, shell=True, check=True, text=True, executable='/bin/bash')
            phs_uw.append(nib.load(tempdir+"/unwrapped.nii").get_fdata().copy())
        phs_uw = np.stack(phs_uw)
        tmpdir.cleanup()
        return np.moveaxis(phs_uw,0,coildim) 
    else:
        phs_romeo = nib.Nifti1Image(phs_in, affine)
        nib.save(phs_romeo, phs_name)
        if return_b0:
            subproc += " -B"
            process = subprocess.run(subproc, shell=True, check=True, text=True, executable='/bin/bash')
            fmap = -1 * 2*np.pi * nib.load(tempdir+"/B0.nii").get_fdata() # to [rad/m]
            tmpdir.cleanup()
            return fmap # [x,y,z]
        else:
            process = subprocess.run(subproc, shell=True, check=True, text=True, executable='/bin/bash')
            phs_uw = nib.load(tempdir+"/unwrapped.nii").get_fdata()
            tmpdir.cleanup()
            return phs_uw

def check_dependency_data(file_list):
    
    """
    Check if field maps or acs data is available in dependency folder

    file_list: File lists with dependency data
    """

    base_folder = os.path.dirname(file_list)
    if not os.path.isfile(file_list):
        logging.debug(f"File list for dependency data not available: {file_list}.")
        return False
    else:
        file_name = np.loadtxt(file_list, dtype=str, ndmin=1)[-1]
        if not os.path.isfile(os.path.join(base_folder, file_name)):
            logging.debug(f"Dependency data: {file_name} not available.")
            return False
    
    return True       
    

## Old

# These are copied from ismrmrdtools.coils, which depends on scipy
def calculate_prewhitening_old(noise, scale_factor=1.0, normalize=True):
    '''Calculates the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, nsamples]``
    :scale_factor: Applied on the noise covariance matrix. Used to
                   adjust for effective noise bandwith and difference in
                   sampling rate between noise calibration and actual measurement:
                   scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio

    :returns w: Prewhitening matrix, ``[coil, coil]``, w*data is prewhitened
    '''

    noise = noise.reshape((noise.shape[0], noise.size//noise.shape[0]))

    R = np.cov(noise)
    if normalize:
        R /= np.mean(abs(np.diag(R)))
    R[np.diag_indices_from(R)] = abs(R[np.diag_indices_from(R)])
    # R = sqrtm(np.linalg.inv(R))
    R = np.linalg.inv(np.linalg.cholesky(R))

    return R

def apply_prewhitening_old(data, dmtx):
    '''Apply the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, ...]``
    :param dmtx: Input noise prewhitening matrix

    :returns w_data: Prewhitened data, ``[coil, ...]``,
    '''

    s = data.shape
    return np.matmul(dmtx, data.reshape(data.shape[0],data.size//data.shape[0])).reshape(s)
