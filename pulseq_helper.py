""" Functions for Pulseq protocol insertion

Includes trajectory prediction with the GIRF
"""

import ismrmrd
import h5py
import numpy as np
import os
import logging
from reco_helper import calc_rotmat, gcs_to_dcs, dcs_to_gcs, intp_axis

def insert_hdr(prot_file, metadata): 
    """
        Inserts the header from an ISMRMRD protocol file
        prot_file:    ISMRMRD protocol file
        metadata:     Dataset header
    """

    #---------------------------
    # Read protocol
    #---------------------------

    if (os.path.splitext(prot_file)[1] == ''):
        prot_file += '.h5'
    try:
        prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
    except:
        prot_file = os.path.splitext(prot_file)[0] + '.hdf5'
        try:
            prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
        except:
            raise ValueError('Pulseq protocol file not found.')

    #---------------------------
    # Process the header 
    #---------------------------

    prot_hdr = ismrmrd.xsd.CreateFromDocument(prot.read_xml_header())

    # user parameters
    if prot_hdr.userParameters is not None:
        dset_udbl_dict = {item.name: item.value for item in metadata.userParameters.userParameterDouble}
        prot_udbl_dict = {item.name: item.value for item in prot_hdr.userParameters.userParameterDouble}
        merged_dict = {**dset_udbl_dict, **prot_udbl_dict} # by merging the dicts, dummy user parameters in the parameter_map are not necessary anymore
        metadata.userParameters.userParameterDouble.clear()
        for key in merged_dict:
            up = ismrmrd.xsd.userParameterDoubleType(name=key, value=merged_dict[key])
            metadata.userParameters.userParameterDouble.append(up)

    # encoding
    dset_e1 = metadata.encoding[0]
    prot_e1 = prot_hdr.encoding[0]
    dset_e1.trajectory = prot_e1.trajectory

    dset_e1.encodedSpace.matrixSize.x = prot_e1.encodedSpace.matrixSize.x
    dset_e1.encodedSpace.matrixSize.y = prot_e1.encodedSpace.matrixSize.y
    dset_e1.encodedSpace.matrixSize.z =  prot_e1.encodedSpace.matrixSize.z
    
    dset_e1.encodedSpace.fieldOfView_mm.x = prot_e1.encodedSpace.fieldOfView_mm.x
    dset_e1.encodedSpace.fieldOfView_mm.y = prot_e1.encodedSpace.fieldOfView_mm.y
    dset_e1.encodedSpace.fieldOfView_mm.z = prot_e1.encodedSpace.fieldOfView_mm.z
    
    dset_e1.reconSpace.matrixSize.x = prot_e1.reconSpace.matrixSize.x
    dset_e1.reconSpace.matrixSize.y = prot_e1.reconSpace.matrixSize.y
    dset_e1.reconSpace.matrixSize.z = prot_e1.reconSpace.matrixSize.z
    
    dset_e1.reconSpace.fieldOfView_mm.x = prot_e1.reconSpace.fieldOfView_mm.x
    dset_e1.reconSpace.fieldOfView_mm.y = prot_e1.reconSpace.fieldOfView_mm.y
    dset_e1.reconSpace.fieldOfView_mm.z = prot_e1.reconSpace.fieldOfView_mm.z

    dset_e1.encodingLimits.slice.minimum = prot_e1.encodingLimits.slice.minimum
    dset_e1.encodingLimits.slice.maximum = prot_e1.encodingLimits.slice.maximum
    dset_e1.encodingLimits.slice.center = prot_e1.encodingLimits.slice.center

    if prot_e1.encodingLimits.kspace_encoding_step_1 is not None:
        dset_e1.encodingLimits.kspace_encoding_step_1.minimum = prot_e1.encodingLimits.kspace_encoding_step_1.minimum
        dset_e1.encodingLimits.kspace_encoding_step_1.maximum = prot_e1.encodingLimits.kspace_encoding_step_1.maximum
        dset_e1.encodingLimits.kspace_encoding_step_1.center = prot_e1.encodingLimits.kspace_encoding_step_1.center
    if prot_e1.encodingLimits.average is not None:
        dset_e1.encodingLimits.average.minimum = prot_e1.encodingLimits.average.minimum
        dset_e1.encodingLimits.average.maximum = prot_e1.encodingLimits.average.maximum
        dset_e1.encodingLimits.average.center = prot_e1.encodingLimits.average.center
    if prot_e1.encodingLimits.repetition is not None:
        dset_e1.encodingLimits.repetition.minimum = prot_e1.encodingLimits.repetition.minimum
        dset_e1.encodingLimits.repetition.maximum = prot_e1.encodingLimits.repetition.maximum
        dset_e1.encodingLimits.repetition.center = prot_e1.encodingLimits.repetition.center
    if prot_e1.encodingLimits.phase is not None:
        dset_e1.encodingLimits.phase.minimum = prot_e1.encodingLimits.phase.minimum
        dset_e1.encodingLimits.phase.maximum = prot_e1.encodingLimits.phase.maximum
        dset_e1.encodingLimits.phase.center = prot_e1.encodingLimits.phase.center
    if prot_e1.encodingLimits.contrast is not None:
        dset_e1.encodingLimits.contrast.minimum = prot_e1.encodingLimits.contrast.minimum
        dset_e1.encodingLimits.contrast.maximum = prot_e1.encodingLimits.contrast.maximum
        dset_e1.encodingLimits.contrast.center = prot_e1.encodingLimits.contrast.center
    if prot_e1.encodingLimits.segment is not None:
        dset_e1.encodingLimits.segment.minimum = prot_e1.encodingLimits.segment.minimum
        dset_e1.encodingLimits.segment.maximum = prot_e1.encodingLimits.segment.maximum
        dset_e1.encodingLimits.segment.center = prot_e1.encodingLimits.segment.center

    # acceleration
    if prot_e1.parallelImaging is not None:
        dset_e1.parallelImaging.accelerationFactor.kspace_encoding_step_1 = prot_e1.parallelImaging.accelerationFactor.kspace_encoding_step_1
        dset_e1.parallelImaging.accelerationFactor.kspace_encoding_step_2 = prot_e1.parallelImaging.accelerationFactor.kspace_encoding_step_2 # also used for SMS factor

    prot.close()

def get_ismrmrd_arrays(prot_file):
    """ Returns all arrays appended to the protocol file and their
        respective keys as a tuple

    """

    if (os.path.splitext(prot_file)[1] == ''):
        prot_file += '.h5'
    try:
        prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
    except:
        prot_file = os.path.splitext(prot_file)[0] + '.hdf5'
        try:
            prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
        except:
            raise ValueError('Pulseq protocol file not found.')

    # get array keys - didnt find a better way
    keys = list(prot.list())
    keys.remove('data')
    keys.remove('xml')

    arr = {}
    for key in keys:
        arr[key] = prot.read_array(key, 0)

    return arr

def check_signature(metadata, prot_hdr):
    """ Check the MD5 signature of the Pulseq sequence against the protocol file

    """
    try:
        hdr_signature = metadata.userParameters.userParameterString[1].value
        if hdr_signature != 'NONE':
            try:
                prot_signature = prot_hdr.userParameters.userParameterString[0].value
                if prot_signature in hdr_signature:
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

def insert_acq(prot_acq, dset_acq, metadata, noncartesian=True, return_basetrj=True):
    """
        Inserts acquisitions from an ISMRMRD protocol file
        
        prot_acq:     ISMRMRD protocol acquisition
        dset_acq:     Dataset acquisition
        metadata:     ISMRMRD header
        noncartesian: For noncartesian acquisitions a trajectory or readout gradients has to be provided
                      If readout gradients are provided, the GIRF is applied, but additional parameters have to be provided.
                      The unit for gradients is [T/m]
                      The unit for trajectories is [rad/m * FOV[m]/2pi], which is unitless (used by the BART toolbox & PowerGrid)
    """
  
    # #---------------------------
    # # Process acquisition
    # #---------------------------

    # convert positions for correct rotation matrix - this was experimentally validated on 20210709
    # In Pulseq interpreter v1.4 this is valid for the "old/compat" FOV positioning mode
    # Shifts and rotations in diffent directions lead to correctly shifted/rotated images and trajectories
    tmp = -1* np.asarray(dset_acq.phase_dir[:])
    dset_acq.phase_dir[:] = np.asarray(dset_acq.read_dir[:])
    dset_acq.read_dir[:] = tmp
    dset_acq.slice_dir[:] = -1 * np.asarray(dset_acq.slice_dir[:])

    # encoding counters
    dset_acq.idx.kspace_encode_step_1 = prot_acq.idx.kspace_encode_step_1
    dset_acq.idx.kspace_encode_step_2 = prot_acq.idx.kspace_encode_step_2
    dset_acq.idx.slice = prot_acq.idx.slice
    dset_acq.idx.contrast = prot_acq.idx.contrast
    dset_acq.idx.phase = prot_acq.idx.phase
    dset_acq.idx.average = prot_acq.idx.average
    dset_acq.idx.repetition = prot_acq.idx.repetition
    dset_acq.idx.set = prot_acq.idx.set
    dset_acq.idx.segment = prot_acq.idx.segment

    # flags
    if prot_acq.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
        dset_acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
    if prot_acq.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
        dset_acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)
    if prot_acq.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
        dset_acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
        return
    if prot_acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
        dset_acq.setFlag(ismrmrd.ACQ_IS_PHASECORR_DATA)
        return
    if prot_acq.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):
        dset_acq.setFlag(ismrmrd.ACQ_IS_DUMMYSCAN_DATA)
        return
    if prot_acq.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
        dset_acq.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
        # for Jemris reconstructions we always need the trajectory, even if its Cartesian
        dset_acq.resize(trajectory_dimensions=prot_acq.traj[:].shape[1], number_of_samples=dset_acq.number_of_samples, active_channels=dset_acq.active_channels)
        if dset_acq.traj.shape[-1] > 0:
            dset_acq.traj[:] = prot_acq.traj[:]
        return

    # deal with noncartesian trajectories
    base_trj = None
    if noncartesian and dset_acq.idx.segment == 0:
        
        use_girf = False
        if metadata.acquisitionSystemInformation.systemModel == 'Investigational_Device_7T_Plus':
            use_girf = True

        # calculate full number of samples - for segmented ADCs
        nsamples = dset_acq.number_of_samples
        nsegments = metadata.encoding[0].encodingLimits.segment.maximum + 1
        nsamples_full = int(nsamples*nsegments+0.5)
        if dset_acq.traj[:].size > 0:
            nsamples_full = dset_acq.traj.shape[0] # samples should already be reshaped if trajectory was added to the rawdata
        nsamples_max = 65535
        if nsamples_full > nsamples_max:
            raise ValueError("The number of samples exceed the maximum allowed number of 65535 (uint16 maximum).")
       
        # save data as it gets corrupted by the resizing, dims are [nc, samples]
        data_tmp = dset_acq.data[:].copy()
        traj_tmp = dset_acq.traj[:].copy()

        # resize data - traj_dims: [kx,ky,kz,t,(GIRF-)k0]
        dset_acq.resize(trajectory_dimensions=5, number_of_samples=nsamples_full, active_channels=dset_acq.active_channels)
        dset_acq.traj[:,3] = np.zeros(nsamples_full) # space for time vector for B0 correction

        # trajectory already inserted from Skope system
        # kz should be set to zero, if trajectory is 2-dimensional
        if traj_tmp.size > 0:
            dset_acq.traj[:,:3] = traj_tmp[:,:3]
            rotmat = calc_rotmat(dset_acq)
            base_trj = calc_traj(prot_acq, metadata, nsamples_full, rotmat, use_girf=False)[0] # for FOV shift reapply
            if traj_tmp.shape[1] > 3: # Skope k0 is saved in 3rd dim -> move to 4th dim
                dset_acq.traj[:,4] = traj_tmp[:,3]
            else:
                dset_acq.traj[:,4] = np.zeros(len(dset_acq.traj[:]))              

        # trajectory from protocol file - check via shape
        elif prot_acq.traj.shape[0] == dset_acq.data.shape[1] and prot_acq.traj[:,:3].max() > 1:
            dset_acq.traj[:,:3] = prot_acq.traj[:,:3]
            dset_acq.traj[:,4] = np.zeros(len(dset_acq.traj))
            base_trj = dset_acq.traj[:].copy()

        # gradients from protocol file - calculate trajectory with girf
        else:
            rotmat = calc_rotmat(dset_acq)
            base_trj, dset_acq.traj[:,:3], dset_acq.traj[:,4] = calc_traj(prot_acq, metadata, nsamples_full, rotmat, use_girf=use_girf) # [samples, dims]

        # fill extended part of data with zeros
        dset_acq.data[:] = np.concatenate((data_tmp, np.zeros([dset_acq.active_channels, nsamples_full - nsamples])), axis=-1)

        # remove first ADCs of spirals as they can be corrupted
        delay = metadata.userParameters.userParameterDouble[1].value
        trajtype = metadata.encoding[0].trajectory.value
        if delay > 0 and trajtype=='spiral': # only do this if trajectory was sufficiently delayed
            dwelltime = 1e-6 * metadata.userParameters.userParameterDouble[0].value
            rm_ix = int(delay/dwelltime)
            if rm_ix%2: rm_ix -= 1 # stay at even number of samples
            data_tmp = dset_acq.data[:,rm_ix:]
            traj_tmp = dset_acq.traj[rm_ix:]
            dset_acq.resize(trajectory_dimensions=dset_acq.trajectory_dimensions, number_of_samples=dset_acq.number_of_samples-rm_ix, active_channels=dset_acq.active_channels)
            dset_acq.data[:] = data_tmp
            dset_acq.traj[:] = traj_tmp
            base_trj = base_trj[rm_ix:]

    if return_basetrj:
        return base_trj
 

def calc_traj(acq, hdr, ncol, rotmat, use_girf=True):
    """ Calculates the kspace trajectory from any gradient using Girf prediction and interpolates it on the adc raster

        acq: acquisition from hdf5 protocol file
        hdr: protocol header
        ncol: number of samples
        rotmat: rotation matrix
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
    
    # delay before trajectory begins - WIP: allow to provide an array of delays - this would be useful e.g. for EPI
    gradshift = hdr.userParameters.userParameterDouble[1].value

    # ADC sampling time
    adctime = dwelltime * np.arange(0.5, ncol)

    # add some zeros around gradient for right interpolation
    zeros = 10
    grad = np.concatenate((np.zeros([dims,zeros]), grad, np.zeros([dims,zeros])), axis=1)
    gradshift -= zeros*dt_grad

    # time vector for interpolation
    gradtime = dt_grad * np.arange(grad.shape[-1]) + gradshift
    gradtime += dt_grad/2 - dt_skope/2 # account for cumsum (assumes rects for integration, we have triangs) - dt_skope/2 seems to be necessary

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
        k0 = pred_grad[0] # 0th order field [T]
        pred_grad = pred_grad[1:]

        # rotate back to logical system
        pred_grad = dcs_to_gcs(pred_grad, rotmat).real

        # calculate global phase term k0 [rad]
        k0 = np.cumsum(k0.real) * dt_grad * gammabar * 2*np.pi
        k0 = intp_axis(adctime, gradtime, k0, axis=0)
    else:
        pred_grad = grad.copy()
        k0 = None

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

    return base_trj, pred_trj, k0

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
