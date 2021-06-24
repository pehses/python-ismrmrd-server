
import ismrmrd
import os
import itertools
import logging
import numpy as np
import base64

from bart import bart
from cfft import cfftn, cifftn
from reco_helper import calculate_prewhitening, apply_prewhitening
from pulseq_prot import get_ismrmrd_arrays

""" Reconstruction of simulation data from Jemris with the BART toolbox
    WIP: support also Pulseq data acquired with Jemris sequences

"""


# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")

########################
# Main Function
########################

def process(connection, config, metadata):
  
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")
    
    logging.info("Config: \n%s", config)

    # Check for GPU availability
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        gpu = True
    else:
        gpu = False

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("Trajectory type '%s', matrix size (%s x %s x %s), field of view (%s x %s x %s)mm^3", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # # Initialize lists for datasets
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    # n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1

    acqGroup = [[] for _ in range(n_slc)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * n_slc
    sensmaps_jemris = []
    dmtx = None

    try:
        for item in connection:

            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                # run noise decorrelation - WIP: noise scans not yet supported in Jemris
                if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                    noiseGroup.append(item)
                    continue
                elif len(noiseGroup) > 0 and dmtx is None:
                    noise_data = []
                    for acq in noiseGroup:
                        noise_data.append(acq.data)
                    noise_data = np.concatenate(noise_data, axis=1)
                    # calculate pre-whitening matrix
                    dmtx = calculate_prewhitening(noise_data)
                    del(noise_data)
                    noiseGroup.clear()
                
                # Other flags
                if item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # ADCs with no specific purpose
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                # Jemris sensitivity maps
                elif item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA):
                    sensmap_coil = item.data[:].reshape(item.traj[0].astype(np.int))
                    if np.sum(sensmap_coil.shape) == 3: # simulated without coil array
                        continue
                    sens_fov = item.user_int[0]
                    sensmap_coil = intp_sensmaps(sensmap_coil, sens_fov, metadata)
                    sensmaps_jemris.append(sensmap_coil)
                    continue
                # Sensitivity maps from calibration scan - WIP: not tested yet
                elif item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    continue
                elif sensmaps[item.idx.slice] is None:
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx, gpu)
                    acsGroup[item.idx.slice].clear()

                # Accumulate all imaging readouts in a group
                acqGroup[item.idx.slice].append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    logging.info("Processing a group of k-space data")
                    images = process_raw(acqGroup[item.idx.slice], config, metadata, dmtx, sensmaps[item.idx.slice], sensmaps_jemris, gpu)
                    logging.debug("Sending images to client:\n%s", images)
                    connection.send_image(images)
                    acqGroup[item.idx.slice].clear() # free memory

    finally:
        connection.send_close()

# %%
#########################
# Process Data
#########################

def process_raw(group, config, metadata, dmtx=None, sensmaps=None, sensmaps_jemris=None, gpu=False):

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    
    data, trj = sort_data(group, metadata, dmtx)
    nc = data.shape[-1]

    if gpu:
        nufft_config = 'nufft -g -i -t -d %d:%d:%d'%(nx, nx, nz)
        ecalib_config = 'ecalib -g -m 1 -I'
        pics_config = 'pics -g -S -e -i 50 -t'
    else:
        nufft_config = 'nufft -i -t -d %d:%d:%d'%(nx, nx, nz)
        ecalib_config = 'ecalib -m 1 -I'
        pics_config = 'pics -S -e -i 50 -t'

    # Take sensmaps from Jemris
    if sensmaps is None and sensmaps_jemris:
        sensmaps = np.stack(sensmaps_jemris).T # [z,y,x,coils]
        if nz==2: # 2D
            sensmaps = sensmaps[group[0].idx.slice]
            sensmaps = np.transpose(sensmaps[np.newaxis], [2,1,0,3]) # [x,y,1,coils]
        else: # 3D
            sensmaps = np.transpose(sensmaps, [2,1,0,3]) # [x,y,z,coils]
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)

    # calculate sensitivity maps from imaging data, if selected
    force_pics = False
    if sensmaps is None and force_pics:
        sensmaps = bart(1, nufft_config, trj, data) # nufft
        if data.shape[-1] != nc:
            data = data[...,np.newaxis]
        sensmaps = cfftn(sensmaps, [k for k in range(len(data)-1)]) # back to k-space
        sensmaps = bart(1, ecalib_config, sensmaps)  # ESPIRiT calibration

    # Recon
    if sensmaps is None:
        data = bart(1, nufft_config, trj, data) # nufft
        if data.shape[-1] == nc:
            data = np.sqrt(np.sum(np.abs(data)**2, axis=-1)) # Sum of squares coil combination
    else:
        data = bart(1, pics_config , trj, data, sensmaps)
    data = np.abs(data)
    data = data[:,::-1] # correct orientation

    # make sure that data is at least 3D
    while np.ndim(data) < 3:
        data = data[..., np.newaxis]
    
    logging.debug("Image data is size %s" % (data.shape,))
    if group[0].idx.slice == 0:
        np.save(debugFolder + "/" + "img.npy", data)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768'})
    xml = meta.serialize()
    
    images = []
    n_par = data.shape[-1]
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    # n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1

    # Format as ISMRMRD image data
    if n_par > 1:
        for par in range(n_par):
            image = ismrmrd.Image.from_array(data[...,par], acquisition=group[0])
            image.image_index = 1 + par # slices/partitions
            image.image_series_index = 1 # e.g. different contrasts (not supported yet)
            image.slice = 0
            image.attribute_string = xml
            images.append(image)
    else:
        image = ismrmrd.Image.from_array(data[...,0], acquisition=group[0])
        image.image_index = 1 + group[0].idx.slice #slices/partitions
        image.image_series_index = 1 + group[0].idx.repetition # e.g. different contrasts
        image.slice = 0
        image.attribute_string = xml
        images.append(image)

    return images

def process_acs(group, config, metadata, dmtx=None, gpu=False):
    if len(group)>0:
        data, trj = sort_data(group, metadata, dmtx)

        nx = metadata.encoding[0].encodedSpace.matrixSize.x
        ny = metadata.encoding[0].encodedSpace.matrixSize.y
        nz = metadata.encoding[0].encodedSpace.matrixSize.z
        if gpu:
            nufft_config = 'nufft -g -i -l 0.001 -t -d %d:%d:%d'%(nx, ny, nz)
            ecalib_config = 'ecalib -g -m 1 -I'
        else:
            nufft_config = 'nufft -i -l 0.001 -t -d %d:%d:%d'%(nx, nx, nz)
            ecalib_config = 'ecalib -m 1 -I'

        sensmaps = bart(1, nufft_config, trj, data) # nufft
        sensmaps = cfftn(sensmaps, [0, 1, 2]) # back to k-space
        sensmaps = bart(1, ecalib_config, data)  # ESPIRiT calibration

        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None

# %%
#########################
# Sort Data
#########################

def sort_data(group, metadata, dmtx=None):
    
    fov_x = metadata.encoding[0].reconSpace.fieldOfView_mm.x
    fov_y = metadata.encoding[0].reconSpace.fieldOfView_mm.y
    fov_z = metadata.encoding[0].reconSpace.fieldOfView_mm.z

    sig = list()
    trj = list()
    for acq in group:
       
        # data with optional prewhitening
        if dmtx is None:
            sig.append(acq.data)
        else:
            sig.append(apply_prewhitening(acq.data, dmtx))

        # trajectory
        traj = np.swapaxes(acq.traj,0,1) # [samples, dims] to [dims, samples]
        traj[0] *= fov_x / (2*np.pi) # rad/mm -> bart (dimensionless)
        traj[1] *= fov_y / (2*np.pi)
        traj[2] *= fov_z / (2*np.pi)
        trj.append(traj)
   
    # convert lists to numpy arrays
    trj = np.asarray(trj) # current size: (nacq, 3, ncol)
    sig = np.asarray(sig) # current size: (nacq, ncha, ncol)

    # rearrange trj & sig for bart
    trj = np.transpose(trj, [1, 2, 0]) # [3, ncol, nacq]
    sig = np.transpose(sig, [2, 0, 1])[np.newaxis] # [1, ncol, nacq, ncha]
    logging.debug("Trajectory shape = %s , Signal Shape = %s "%(trj.shape, sig.shape))
    
    np.save(debugFolder + "/" + "trj.npy", trj)
    np.save(debugFolder + "/" + "raw.npy", sig)

    return sig, trj

def intp_sensmaps(sensmap_coil, sens_fov, metadata):

    fov_x = int(metadata.encoding[0].reconSpace.fieldOfView_mm.x)
    fov_y = int(metadata.encoding[0].reconSpace.fieldOfView_mm.y)
    fov_z = int(metadata.encoding[0].reconSpace.fieldOfView_mm.z)

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z

    from skimage.transform import resize
    
    # Resize to full fov
    if sensmap_coil.shape[2] == 1:
        newshape = [sens_fov, sens_fov, 1]
    else:
        newshape = [sens_fov, sens_fov, sens_fov]
    sensmap_coil = resize(sensmap_coil.real, newshape, anti_aliasing=True) + 1j*resize(sensmap_coil.imag, newshape, anti_aliasing=True)

    # Cut out measurement fov
    if fov_x > sens_fov or fov_y > sens_fov or fov_z > sens_fov:
        print("WARNING: Sensitivity map FOV is smaller than measurement FOV")
    slc_x = (sens_fov-fov_x)//2
    slc_y = (sens_fov-fov_y)//2
    slc_z = (sens_fov-fov_z)//2
    if sensmap_coil.shape[2] == 1:
        sensmap_coil = sensmap_coil[slc_x:-slc_x, slc_y:-slc_y]
    else:
        sensmap_coil = sensmap_coil[slc_x:-slc_x, slc_y:-slc_y, slc_z:-slc_z]

    # Interpolate to reconstruction matrix
    newshape = [nx, ny, nz]
    sensmap_coil = resize(sensmap_coil.real, newshape, anti_aliasing=True) + 1j*resize(sensmap_coil.imag, newshape, anti_aliasing=True)

    return sensmap_coil