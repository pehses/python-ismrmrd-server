""" Functions for Pulseq metadata insertion

Includes trajectory prediction with the GIRF
"""

import ismrmrd
import h5py
import numpy as np
import os
import logging
from reco_helper import calc_rotmat, gcs_to_dcs, dcs_to_gcs, intp_axis

def insert_hdr(meta_file, hdr): 
    """
        Inserts the header from an ISMRMRD metadata file
        meta_file: ISMRMRD metadata file
        hdr:      Dataset header
    """

    #---------------------------
    # Read metadata
    #---------------------------

    if (os.path.splitext(metadata_file)[1] == ''):
        metadata_file += '.h5'
    try:
        metadata = ismrmrd.Dataset(metadata_file, create_if_needed=False)
    except:
        metadata_file = os.path.splitext(metadata_file)[0] + '.hdf5'
        try:
            metadata = ismrmrd.Dataset(metadata_file, create_if_needed=False)
        except:
            raise ValueError('Pulseq metadata file not found.')

    #---------------------------
    # Process the header 
    #---------------------------

    metadata_hdr = ismrmrd.xsd.CreateFromDocument(metadata.read_xml_header())

    # user parameters
    if metadata_hdr.userParameters is not None:
        dset_udbl = hdr.userParameters.userParameterDouble
        meta_udbl = metadata_hdr.userParameters.userParameterDouble
        for ix, param in enumerate(meta_udbl):
            dset_udbl[ix].name = param.name
            dset_udbl[ix].value = param.value

    # encoding
    dset_e1 = hdr.encoding[0]
    meta_e1 = metadata_hdr.encoding[0]
    dset_e1.trajectory = meta_e1.trajectory

    dset_e1.encodedSpace.matrixSize.x = meta_e1.encodedSpace.matrixSize.x
    dset_e1.encodedSpace.matrixSize.y = meta_e1.encodedSpace.matrixSize.y
    dset_e1.encodedSpace.matrixSize.z =  meta_e1.encodedSpace.matrixSize.z
    
    dset_e1.encodedSpace.fieldOfView_mm.x = meta_e1.encodedSpace.fieldOfView_mm.x
    dset_e1.encodedSpace.fieldOfView_mm.y = meta_e1.encodedSpace.fieldOfView_mm.y
    dset_e1.encodedSpace.fieldOfView_mm.z = meta_e1.encodedSpace.fieldOfView_mm.z
    
    dset_e1.reconSpace.matrixSize.x = meta_e1.reconSpace.matrixSize.x
    dset_e1.reconSpace.matrixSize.y = meta_e1.reconSpace.matrixSize.y
    dset_e1.reconSpace.matrixSize.z = meta_e1.reconSpace.matrixSize.z
    
    dset_e1.reconSpace.fieldOfView_mm.x = meta_e1.reconSpace.fieldOfView_mm.x
    dset_e1.reconSpace.fieldOfView_mm.y = meta_e1.reconSpace.fieldOfView_mm.y
    dset_e1.reconSpace.fieldOfView_mm.z = meta_e1.reconSpace.fieldOfView_mm.z

    dset_e1.encodingLimits.slice.minimum = meta_e1.encodingLimits.slice.minimum
    dset_e1.encodingLimits.slice.maximum = meta_e1.encodingLimits.slice.maximum
    dset_e1.encodingLimits.slice.center = meta_e1.encodingLimits.slice.center

    if meta_e1.encodingLimits.kspace_encoding_step_1 is not None:
        dset_e1.encodingLimits.kspace_encoding_step_1.minimum = meta_e1.encodingLimits.kspace_encoding_step_1.minimum
        dset_e1.encodingLimits.kspace_encoding_step_1.maximum = meta_e1.encodingLimits.kspace_encoding_step_1.maximum
        dset_e1.encodingLimits.kspace_encoding_step_1.center = meta_e1.encodingLimits.kspace_encoding_step_1.center
    if meta_e1.encodingLimits.average is not None:
        dset_e1.encodingLimits.average.minimum = meta_e1.encodingLimits.average.minimum
        dset_e1.encodingLimits.average.maximum = meta_e1.encodingLimits.average.maximum
        dset_e1.encodingLimits.average.center = meta_e1.encodingLimits.average.center
    if meta_e1.encodingLimits.phase is not None:
        dset_e1.encodingLimits.phase.minimum = meta_e1.encodingLimits.phase.minimum
        dset_e1.encodingLimits.phase.maximum = meta_e1.encodingLimits.phase.maximum
        dset_e1.encodingLimits.phase.center = meta_e1.encodingLimits.phase.center
    if meta_e1.encodingLimits.contrast is not None:
        dset_e1.encodingLimits.contrast.minimum = meta_e1.encodingLimits.contrast.minimum
        dset_e1.encodingLimits.contrast.maximum = meta_e1.encodingLimits.contrast.maximum
        dset_e1.encodingLimits.contrast.center = meta_e1.encodingLimits.contrast.center
    if meta_e1.encodingLimits.segment is not None:
        dset_e1.encodingLimits.segment.minimum = meta_e1.encodingLimits.segment.minimum
        dset_e1.encodingLimits.segment.maximum = meta_e1.encodingLimits.segment.maximum
        dset_e1.encodingLimits.segment.center = meta_e1.encodingLimits.segment.center

    # acceleration
    if meta_e1.parallelImaging is not None:
        dset_e1.parallelImaging.accelerationFactor.kspace_encoding_step_1 = meta_e1.parallelImaging.accelerationFactor.kspace_encoding_step_1
        dset_e1.parallelImaging.accelerationFactor.kspace_encoding_step_2 = meta_e1.parallelImaging.accelerationFactor.kspace_encoding_step_2 # used for SMS factor

    metadata.close()

def get_ismrmrd_arrays(metadata_file):
    """ Returns all arrays appended to the metadata file and their
        respective keys as a tuple

    """

    if (os.path.splitext(metadata_file)[1] == ''):
        metadata_file += '.h5'
    try:
        metadata = ismrmrd.Dataset(metadata_file, create_if_needed=False)
    except:
        metadata_file = os.path.splitext(metadata_file)[0] + '.hdf5'
        try:
            metadata = ismrmrd.Dataset(metadata_file, create_if_needed=False)
        except:
            raise ValueError('Pulseq metadata file not found.')

    # get array keys - didnt find a better way
    keys = list(metadata.list())
    keys.remove('data')
    keys.remove('xml')

    arr = {}
    for key in keys:
        arr[key] = metadata.read_array(key, 0)

    return arr

def check_signature(hdr, meta_hdr):
    """ Check the MD5 signature of the Pulseq sequence against the metadata file

    """
    try:
        hdr_signature = hdr.userParameters.userParameterString[1].value
        if hdr_signature != 'NONE':
            try:
                meta_signature = meta_hdr.userParameters.userParameterString[0].value
                if meta_signature in hdr_signature:
                    logging.debug(f"Signature check passed with signature {hdr_signature}.")
                else:
                    logging.debug("WARNING: Signature check failed. ISMRMRD metadata file has different MD5 Hash than sequence.")
            except:
                logging.debug("WARNING: Can not check signature as ISMRMRD file contains no signature.")
        else:
            logging.debug("Pulseq sequence has no signature.")
    except:
        logging.debug("Sequence signature not available.")

def read_acqs(filename):
    """ Reads all acquisitions of an ISMRMRD file and returns them as list.
        This is faster than reading them one by one with file.read_acquisition().
    """
    file = h5py.File(filename, mode='r', driver='core')
    acq_data = [item for item in file['dataset']['data']]
    file.close()
    acqs = [ismrmrd.Acquisition(acq['head']) for acq in acq_data]
    for n in range(len(acqs)):
        acqs[n].traj[:] = acq_data[n]['traj'].reshape((acqs[n].number_of_samples,acqs[n].trajectory_dimensions))[:]
        acqs[n].data[:] = acq_data[n]['data'].view(np.complex64).reshape((acqs[n].active_channels, acqs[n].number_of_samples))[:]
    return acqs

def insert_acq(meta_acq, dset_acq, hdr, noncartesian=True, return_basetrj=True):
    """
        Inserts acquisitions from an ISMRMRD metadata file
        
        meta_acq:     Acquisition from ISMRMRD metadata file
        dset_acq:     Dataset acquisition
        hdr:          Dataset header
        noncartesian: For noncartesian acquisitions a trajectory or readout gradients has to be provided
                      If readout gradients are provided, the GIRF is applied, but additional parameters have to be provided.
                      The unit for gradients is [T/m]
                      The unit for trajectories is [rad/m * FOV[m]/2pi], which is unitless (used by the BART toolbox & PowerGrid)
    """
  
    # #---------------------------
    # # Process acquisition
    # #---------------------------

    # convert positions for correct rotation matrix - this was experimentally validated on 20210709
    # Shifts and rotations in diffent directions lead to correctly shifted/rotated images and trajectories
    tmp = -1* np.asarray(dset_acq.phase_dir[:])
    dset_acq.phase_dir[:] = np.asarray(dset_acq.read_dir[:])
    dset_acq.read_dir[:] = tmp
    dset_acq.slice_dir[:] = -1 * np.asarray(dset_acq.slice_dir[:])

    # encoding counters
    dset_acq.idx.kspace_encode_step_1 = meta_acq.idx.kspace_encode_step_1
    dset_acq.idx.kspace_encode_step_2 = meta_acq.idx.kspace_encode_step_2
    dset_acq.idx.slice = meta_acq.idx.slice
    dset_acq.idx.contrast = meta_acq.idx.contrast
    dset_acq.idx.phase = meta_acq.idx.phase
    dset_acq.idx.average = meta_acq.idx.average
    dset_acq.idx.repetition = meta_acq.idx.repetition
    dset_acq.idx.set = meta_acq.idx.set
    dset_acq.idx.segment = meta_acq.idx.segment

    # flags
    if meta_acq.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
        dset_acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
    if meta_acq.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
        dset_acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)
    if meta_acq.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
        dset_acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
        return
    if meta_acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
        dset_acq.setFlag(ismrmrd.ACQ_IS_PHASECORR_DATA)
        return
    if meta_acq.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):
        dset_acq.setFlag(ismrmrd.ACQ_IS_DUMMYSCAN_DATA)
        return
    if meta_acq.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
        dset_acq.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
        # for Jemris reconstructions we always need the trajectory, even if its Cartesian
        dset_acq.resize(trajectory_dimensions=meta_acq.traj[:].shape[1], number_of_samples=dset_acq.number_of_samples, active_channels=dset_acq.active_channels)
        if dset_acq.traj.shape[-1] > 0:
            dset_acq.traj[:] = meta_acq.traj[:]
        return

    # deal with noncartesian trajectories
    base_trj = None
    if noncartesian and dset_acq.idx.segment == 0:
        
        # calculate full number of samples
        nsamples = dset_acq.number_of_samples
        nsegments = hdr.encoding[0].encodingLimits.segment.maximum + 1 # number of ADC segments in the Pulseq sequence
        nsamples_full = int(nsamples*nsegments+0.5)
        nsamples_max = 65535
        if nsamples_full > nsamples_max:
            raise ValueError("The number of samples exceed the maximum allowed number of 65535 (uint16 maximum).")
       
        # save data as it gets corrupted by the resizing, dims are [nc, samples]
        data_tmp = dset_acq.data[:]
        traj_tmp = dset_acq.traj[:]

        # resize data - traj_dims: [kx,ky,kz]
        dset_acq.resize(trajectory_dimensions=3, number_of_samples=nsamples_full, active_channels=dset_acq.active_channels)

        # trajectory already inserted in raw data file?
        if traj_tmp.size > 0:
            reco_trj = traj_tmp[:,:3]
            base_trj = reco_trj.copy()
        # trajectory in metadata file? check should be pretty robust
        elif meta_acq.traj.shape[0] == dset_acq.data.shape[1] and meta_acq.traj[:,:3].max() > 1:
            reco_trj = meta_acq.traj[:,:3]
            base_trj = reco_trj.copy()
        # gradients in metadata file? 
        else:
            use_girf = False
            if hdr.acquisitionSystemInformation.systemModel == 'Investigational_Device_7T_Plus':
                use_girf = True
            rotmat = calc_rotmat(dset_acq)
            base_trj, reco_trj = calc_traj(meta_acq, hdr, nsamples_full, rotmat, use_girf=use_girf) # [samples, dims]

        # fill extended part of data with zeros
        dset_acq.data[:] = np.concatenate((data_tmp, np.zeros([dset_acq.active_channels, nsamples_full - nsamples])), axis=-1)
        dset_acq.traj[:,:3] = reco_trj.copy()
    
    if return_basetrj:
        return base_trj
 

def calc_traj(acq, hdr, ncol, rotmat, use_girf=True):
    """ Calculates the kspace trajectory from any gradient using Girf prediction and interpolates it on the adc raster

        acq: acquisition from metadata file
        hdr: (merged) dataset (metadata) header
        ncol: number of samples
    """
    
    dt_grad = 10e-6 # [s]
    dt_skope = 1e-6 # [s]
    gammabar = 42.577e6

    grad = np.swapaxes(acq.traj[:],0,1) # [dims, samples] [T/m]
    dims = grad.shape[0]

    # WIP: currently unclear how to set z-FOV / scale trajectory correctly for SMS-recon with phase blips
    fov = np.array([hdr.encoding[0].reconSpace.fieldOfView_mm.x,
                    hdr.encoding[0].reconSpace.fieldOfView_mm.y,
                    hdr.encoding[0].reconSpace.fieldOfView_mm.z])

    dwelltime = 1e-6 * hdr.userParameters.userParameterDouble[0].value
    
    # delay before trajectory begins
    gradshift = hdr.userParameters.userParameterDouble[1].value

    # ADC sampling time
    adctime = dwelltime * np.arange(0.5, ncol)

    # add some zeros around gradient for correct interpolation
    zeros = 10
    grad = np.concatenate((np.zeros([dims,zeros]), grad, np.zeros([dims,zeros])), axis=1)
    gradshift -= zeros*dt_grad

    # time vector for interpolation
    gradtime = dt_grad * np.arange(grad.shape[-1]) + gradshift
    gradtime += dt_grad/2 - dt_skope/2 # account for cumsum - dt_skope/2 seems to be necessary

    # add zero z-dir if necessary
    if dims == 2:
        grad = np.concatenate((grad, np.zeros([1, grad.shape[1]])), axis=0)

    ##############################
    ## girf trajectory prediction:
    ##############################
    if use_girf:
        dependencyFolder = "/tmp/share/dependency"
        girf = np.load(os.path.join(dependencyFolder, "girf_10us.npy"))

        # rotation to phys coord system
        grad_phys = gcs_to_dcs(grad, rotmat)

        # gradient prediction
        pred_grad = grad_pred(grad_phys, girf)
        pred_grad = pred_grad[1:]

        # rotate back to logical system
        pred_grad = dcs_to_gcs(pred_grad, rotmat).real
    else:
        pred_grad = grad.copy()

    pred_trj = np.cumsum(pred_grad, axis=1) * dt_grad * gammabar # calculate trajectory [1/m]
    pred_trj *= 1e-3 * fov[:,np.newaxis] # scale with FOV for BART & PowerGrid recon
    base_trj = np.cumsum(grad, axis=1) * dt_grad * gammabar
    base_trj *= 1e-3 * fov[:,np.newaxis]

    # set z-axis for 3D imaging if trajectory is two-dimensional 
    # this only works for Cartesian sampling in kz (works also with CAIPI)
    if dims == 2:
        nz = hdr.encoding[0].encodedSpace.matrixSize.z
        partition = acq.idx.kspace_encode_step_2
        kz = partition - nz//2
        pred_trj[2] =  kz * np.ones(pred_trj.shape[1])        
        base_trj[2] =  kz * np.ones(base_trj.shape[1])

    pred_trj = intp_axis(adctime, gradtime, pred_trj, axis=1) # align trajectory to scanner ADC
    pred_trj = np.swapaxes(pred_trj,0,1) # switch array order to [samples, dims]   
    base_trj = intp_axis(adctime, gradtime-1e-5, base_trj, axis=1) # shift base_trj by 10us for undoing the FOV shift (see fov_shift_spiral_reapply in reco_helper.py)
    base_trj = np.swapaxes(base_trj,0,1)

    return base_trj, pred_trj
        
def grad_pred(grad, girf):
    """
    gradient prediction with girf
    
    Parameters:
    ------------
    grad: nominal gradient [dims, samples]
    girf: gradient impulse response function [input dims, output dims (incl k0), samples] in frequency space
    """
    ndim = grad.shape[0]
    grad_sampl = grad.shape[-1]
    girf_sampl = girf.shape[-1]

    # zero-fill grad to number of girf samples (add check?)
    if girf_sampl > grad_sampl:
        grad = np.concatenate([grad.copy(), np.zeros([ndim, girf_sampl-grad_sampl])], axis=-1)
    if grad_sampl > girf_sampl:
        logging.debug("WARNING: GIRF is interpolated, check trajectory result carefully.")
        oldgrid = np.linspace(0,girf_sampl,girf_sampl)
        newgrid = np.linspace(0,girf_sampl,grad_sampl)
        girf = intp_axis(newgrid, oldgrid, girf, axis=-1)

    # FFT
    grad = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(grad, axes=-1), axis=-1), axes=-1)

    # apply girf to nominal gradients
    pred_grad = np.zeros_like(girf[0])
    for dim in range(ndim+1):
        pred_grad[dim]=np.sum(grad*girf[np.newaxis,:ndim,dim,:], axis=1)

    # IFFT
    pred_grad = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(pred_grad, axes=-1), axis=-1), axes=-1)
    
    # cut out relevant part
    pred_grad = pred_grad[:,:grad_sampl]

    return pred_grad
