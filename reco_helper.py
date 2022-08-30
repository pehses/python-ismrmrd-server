""" Helper functions for reconstruction
"""

import numpy as np


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
    dmtx = (noise @ np.conj(noise).T)/noise.shape[1]
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
    cc_matrix, s = calibrate_scc(np.moveaxis(data, 1, 0).reshape([nc, -1]))
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

def pcs_to_dcs(grads, patient_position='HFS'):
    """ Convert from patient coordinate system (PCS, physical) 
        to device coordinate system (DCS, physical)
        this is valid for patient orientation head first/supine
    """
    grads = grads.copy()

    # only valid for head first/supine - other orientations see IDEA UserGuide
    if patient_position.upper() == 'HFS':
        grads[1] *= -1
        grads[2] *= -1
    else:
        raise ValueError

    return grads

def dcs_to_pcs(grads, patient_position='HFS'):
    """ Convert from device coordinate system (DCS, physical) 
        to patient coordinate system (PCS, physical)
        this is valid for patient orientation head first/supine
    """
    return pcs_to_dcs(grads, patient_position) # same sign switch
    
def gcs_to_pcs(grads, rotmat):
    """ Convert from gradient coordinate system (GCS, logical) 
        to patient coordinate system (DCS, physical)
    """
    return np.matmul(rotmat, grads)

def pcs_to_gcs(grads, rotmat):
    """ Convert from patient coordinate system (PCS, physical) 
        to gradient coordinate system (GCS, logical) 
    """
    return np.matmul(np.linalg.inv(rotmat), grads)

def gcs_to_dcs(grads, rotmat):
    """ Convert from gradient coordinate system (GCS, logical) 
        to device coordinate system (DCS, physical)
        this is valid for patient orientation head first/supine
    Parameters
    ----------
    grads : numpy array [3, samples]
            gradient to be converted
    rotmat: numpy array [3,3]
            rotation matrix from Siemens Raw Data header or ISMRMRD acquisition header
    Returns
    -------
    grads_cv : numpy.ndarray
               Converted gradient
    """
    grads = grads.copy()

    # rotation from GCS (PHASE,READ,SLICE) to patient coordinate system (PCS)
    grads = gcs_to_pcs(grads, rotmat)
    
    # PCS (SAG,COR,TRA) to DCS (X,Y,Z)
    # only valid for head first/supine - other orientations see IDEA UserGuide
    grads = pcs_to_dcs(grads)
    
    return grads


def dcs_to_gcs(grads, rotmat):
    """ Convert from device coordinate system (DCS, logical) 
        to gradient coordinate system (GCS, physical)
        this is valid for patient orientation head first/supine
    Parameters
    ----------
    grads : numpy array [3, samples]
            gradient to be converted
    rotmat: numpy array [3,3]
            rotation matrix from Siemens Raw Data header or ISMRMRD acquisition header
    Returns
    -------
    grads_cv : numpy.ndarray
               Converted gradient
    """
    grads = grads.copy()
    
    # DCS (X,Y,Z) to PCS (SAG,COR,TRA)
    # only valid for head first/supine - other orientations see IDEA UserGuide
    grads = dcs_to_pcs(grads)
    
    # PCS (SAG,COR,TRA) to GCS (PHASE,READ,SLICE)
    grads = pcs_to_gcs(grads, rotmat)
    
    return grads

## FOV shifts

def fov_shift(sig, shift):
    """ Performs inplane fov shift for Cartesian data of shape [nx,ny,nz,nc]
    """
    fac_x = np.exp(-1j*shift[0]*2*np.pi*np.arange(sig.shape[0])/sig.shape[0])
    fac_y = np.exp(-1j*shift[1]*2*np.pi*np.arange(sig.shape[1])/sig.shape[1])

    sig *= fac_x[:,np.newaxis,np.newaxis,np.newaxis]
    sig *= fac_y[:,np.newaxis,np.newaxis]
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
               For GIRF predicted Pulseq trajectories this is done in pulseq_helper.py

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

## WIP: Calculate image space coordinates in physical coordinate system
# Can be used for higher order recon
def img_coord(metadata, acq):
    "Calculate voxel coordinates in physical coordinate system"

    # matrix size & rotmat
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2) # sms factor
    rotmat =  calc_rotmat(acq)

    # scaling
    res = 1e-3 * metadata.encoding[0].encodedSpace.fieldOfView_mm.x / nx
    slc_res = 1e-3 * metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_slc_red = n_slc // nz # number of reduced slices
    slc_sep = n_slc_red * slc_res

    # global offset & slice offset
    global_offset = 1e-3 * pcs_to_gcs(np.asarray(acq.position), rotmat)
    slice_offset = slc_res*(acq.idx.slice-(n_slc_red-1)/2)

    # Coordinates in GCS (logical)
    ix = np.linspace(-nx/2*res,(nx/2-1)*res, nx) + global_offset[0]
    iy = np.linspace(-ny/2*res,(nx/2-1)*res, ny) + global_offset[1]
    iz = np.linspace(-0.5*slc_sep*(nz-1), 0.5*slc_sep*(nz-1), nz) + global_offset[2] + slice_offset
    grid = np.asarray(np.meshgrid(ix,iy,iz)).reshape([3,-1])

    # Coordinates to DCS (physical)
    grid_rot = gcs_to_dcs(grid, rotmat) # [dims, coords]

    return grid_rot

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
