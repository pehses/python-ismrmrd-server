""" Helper functions for reconstruction
"""

import numpy as np

# These are copied from ismrmrdtools.coils, which depends on scipy
# (wip: import from ismrmrdtools and add scipy to container image)
def calculate_prewhitening(noise, scale_factor=1.0):
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
    R /= np.mean(abs(np.diag(R)))
    R[np.diag_indices_from(R)] = abs(R[np.diag_indices_from(R)])
    # R = sqrtm(np.linalg.inv(R))
    R = np.linalg.inv(np.linalg.cholesky(R))

    return R

def apply_prewhitening(data, dmtx):
    '''Apply the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, ...]``
    :param dmtx: Input noise prewhitening matrix

    :returns w_data: Prewhitened data, ``[coil, ...]``,
    '''

    s = data.shape
    return np.matmul(dmtx, data.reshape(data.shape[0],data.size//data.shape[0])).reshape(s)

def calc_rotmat(acq):
        phase_dir = np.asarray(acq.phase_dir)
        read_dir = np.asarray(acq.read_dir)
        slice_dir = np.asarray(acq.slice_dir)
        return np.round(np.concatenate([phase_dir[:,np.newaxis], read_dir[:,np.newaxis], slice_dir[:,np.newaxis]], axis=1), 6)

def remove_os(data, axis=0):
    '''Remove oversampling (assumes os factor 2)
    '''
    cut = slice(data.shape[axis]//4, (data.shape[axis]*3)//4)
    data = np.fft.ifft(data, axis=axis)
    data = np.delete(data, cut, axis=axis)
    data = np.fft.fft(data, axis=axis)
    return data

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
        to patient coordinate system (DCS, physical)
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

def intp_axis(newgrid, oldgrid, data, axis=0):
    """
    Interpolation along an axis (shape of newgrid, oldgrid and data see np.interp)
    """

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
    Shift field of view of spiral data

    sig: raw data [coils, samples]
    trj: trajectory [samples, dims]
    shift: FOV shift [x_shift, y_shift] in voxel
    matr_sz: matrix size [x,y] of reco
    """

    if (abs(shift[0]) < 1e-2) and (abs(shift[1]) < 1e-2):
        # nothing to do
        return sig

    kmax = matr_sz/2
    sig *= np.exp(-1j*(shift[0]*np.pi*trj[:,0]/kmax[0]+shift[1]*np.pi*trj[:,1]/kmax[1]))

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
    pred_trj = np.swapaxes(pred_trj,0,1) # [dims, samples]
    base_trj = np.swapaxes(base_trj,0,1)

    if (abs(shift[0]) < 1e-2) and (abs(shift[1]) < 1e-2):
        # nothing to do
        return sig

    kmax_x = int(matr_sz[0]/2+0.5)
    kmax_y = int(matr_sz[1]/2+0.5)

    # undo FOV shift from nominal traj
    sig *= np.exp(1j*(shift[0]*np.pi*base_trj[0]/kmax_x+shift[1]*np.pi*base_trj[1]/kmax_y))[np.newaxis]

    # redo FOV shift with predicted traj
    sig *= np.exp(-1j*(shift[0]*np.pi*pred_trj[0]/kmax_x+shift[1]*np.pi*pred_trj[1]/kmax_y))[np.newaxis]

    return sig

    
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
    