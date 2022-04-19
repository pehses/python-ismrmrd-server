
import ismrmrd
import os
import logging
import numpy as np
import ctypes

from bart import bart
from cfft import cfftn, cifftn
from reco_helper import calculate_prewhitening, apply_prewhitening
from pulseq_helper import insert_hdr, insert_acq, read_acqs

""" Reconstruction of simulation data from Jemris
    and of scanner data acquired with JEMRIS sequences
    with the BART toolbox    
"""


# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

########################
# Main Function
########################

def process(connection, config, hdr, meta_file=None):
  
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")
    
    logging.info("Config: \n%s", config)
  
    # Check if Pulseq or simulated data
    if meta_file is not None:
        logging.debug("Reconstruction of scanner data.")
        insert_hdr(meta_file, hdr)
        simu = False
    else:
        logging.debug("Reconstruction of simulated data.")
        simu = True

    # Check for GPU availability
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        gpu = True
    else:
        gpu = False

    try:
        logging.info("Incoming dataset contains %d encodings", len(hdr.encoding))
        logging.info("Trajectory type '%s', matrix size (%s x %s x %s), field of view (%s x %s x %s)mm^3", 
            hdr.encoding[0].trajectory.value, 
            hdr.encoding[0].encodedSpace.matrixSize.x, 
            hdr.encoding[0].encodedSpace.matrixSize.y, 
            hdr.encoding[0].encodedSpace.matrixSize.z, 
            hdr.encoding[0].encodedSpace.fieldOfView_mm.x, 
            hdr.encoding[0].encodedSpace.fieldOfView_mm.y, 
            hdr.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted header: \n%s", hdr)

    # # Initialize lists for datasets
    n_slc = hdr.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = hdr.encoding[0].encodingLimits.contrast.maximum + 1
    n_sets = hdr.encoding[0].encodingLimits.set.maximum + 1
    n_avgs = hdr.encoding[0].encodingLimits.average.maximum + 1

    acqGroup = [[[[[] for _ in range(n_slc)] for _ in range(n_contr)] for _ in range(n_sets)] for _ in range(n_avgs)]
    noiseGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * n_slc
    sensmaps_jemris = []
    dmtx = None

    # read metadata acquisitions - faster than doing it one by one
    if meta_file is not None:
        logging.debug("Reading in metadata acquisitions.")
        acqs = read_acqs(meta_file)

    try:
        for item in connection:

            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                # Insert metadata of acquisitions, if Pulseq data
                if meta_file is not None:
                    insert_acq(acqs[0], item, hdr, return_basetrj=False)
                    acqs.pop(0)

                # Jemris sensitivity maps - only if simulated data (ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA is set in scanner data for some reason)
                elif item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA):
                    sensmap_coil = item.data[:].reshape(item.traj[0].astype(int))
                    if np.sum(sensmap_coil.shape) == 3: # simulated without coil array
                        continue
                    sens_fov = item.user_int[0]
                    sensmap_coil = intp_sensmaps(sensmap_coil, sens_fov, hdr)
                    sensmaps_jemris.append(sensmap_coil)
                    continue

                # run noise decorrelation
                if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                    noiseGroup.append(item)
                    continue
                elif len(noiseGroup) > 0 and dmtx is None:
                    noise_data = []
                    for acq in noiseGroup:
                        noise_data.append(acq.data)
                    noise_data = np.concatenate(noise_data, axis=1)
                    if noise_data.shape[0] == 1:
                        logging.debug("Single Coil data. No prewhitening.")
                    else:
                        dmtx = calculate_prewhitening(noise_data) # calculate pre-whitening matrix
                        if np.isnan(dmtx).any():
                            logging.debug("Dont use noise whitening matrix as it is nan. Check if noise was added in simulation.")
                            dmtx = None
                    del(noise_data)
                    noiseGroup.clear()
                
                # Other flags
                if item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # ADCs with no specific purpose
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                # Sensitivity maps from calibration scan
                elif item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING):
                        pass
                    else:
                        continue
                elif sensmaps[item.idx.slice] is None:
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], hdr, dmtx, gpu)
                    acsGroup[item.idx.slice].clear()

                # Accumulate all imaging readouts in a group
                acqGroup[item.idx.average][item.idx.set][item.idx.contrast][item.idx.slice].append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    logging.info("Processing a group of k-space data")
                    group = acqGroup[item.idx.average][item.idx.set][item.idx.contrast][item.idx.slice]
                    images = process_raw(group, hdr, dmtx, sensmaps[item.idx.slice], sensmaps_jemris, gpu, simu)
                    logging.debug("Sending images to client:\n%s", images)
                    connection.send_image(images)
                    acqGroup[item.idx.average][item.idx.set][item.idx.contrast][item.idx.slice].clear() # free memory

    finally:
        connection.send_close()

# %%
#########################
# Process Data
#########################

def process_raw(group, hdr, dmtx=None, sensmaps=None, sensmaps_jemris=None, gpu=False, simu=True):

    force_pi = False # force parallel imaging recon by calculating sensitivity maps from raw data
    use_jemris_sens = True # take sensmaps from Jemris if no reference scan was used and sensitivities were not calculated from raw data

    nx = hdr.encoding[0].encodedSpace.matrixSize.x
    ny = hdr.encoding[0].encodedSpace.matrixSize.y
    nz = hdr.encoding[0].encodedSpace.matrixSize.z
    
    data, trj = sort_data(group, dmtx)
    logging.debug("Trajectory shape = %s , Signal Shape = %s "%(trj.shape, data.shape))
    nc = data.shape[-1]

    if gpu and nz>1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
        nufft_config = 'nufft -g -i -m 15 -l 0.05 -t -d %d:%d:%d'%(nx, ny, nz)
        ecalib_config = 'ecalib -g -m 1 -I'
        pics_config = 'pics -g -S -e -i 50 -t'
    else:
        nufft_config = 'nufft -i -m 15 -l 0.05 -t -d %d:%d:%d'%(nx, ny, nz)
        ecalib_config = 'ecalib -m 1 -I'
        pics_config = 'pics -S -e -i 50 -t'

    # Optional: Calculate sensmaps from raw data if selected
    if sensmaps is None and force_pi:
        sensmaps = bart(1, nufft_config, trj, data) # nufft
        if sensmaps.shape[-1] != nc:
            sensmaps = sensmaps[...,np.newaxis]
        sensmaps = cfftn(sensmaps, [k for k in range(data.ndim-1)]) # back to k-space
        sensmaps = bart(1, ecalib_config, sensmaps)  # ESPIRiT calibration
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        
    # Optional: Take sensmaps from Jemris, if available
    if sensmaps is None and use_jemris_sens and sensmaps_jemris:
        sensmaps = np.stack(sensmaps_jemris).T # [z,y,x,coils]
        if nz==1: # 2D
            sensmaps = sensmaps[0]
            sensmaps = np.transpose(sensmaps[np.newaxis], [2,1,0,3]) # [x,y,1,coils]
        else: # 3D
            sensmaps = np.transpose(sensmaps, [2,1,0,3]) # [x,y,z,coils]
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)

    # Recon
    logging.debug("Do BART reconstruction.")
    if simu:
        data_ch = bart(1, nufft_config, trj, data) # nufft
        data_ch_mag = np.abs(data_ch)
        data_ch_phs = np.angle(data_ch)
        if sensmaps is None:
            if nc != 1:
                data_mag = np.sqrt(np.sum(np.abs(data)**2, axis=-1)) # Sum of squares coil combination
                data_phs = np.sum(data_ch_phs, axis=-1)
            else: # this is the default as for nc>1, we should have JEMRIS sensitivity maps
                data_mag = data_ch_mag.copy()
                data_phs = data_ch_phs.copy()
        else:
            data = bart(1, pics_config , trj, data, sensmaps)
            data_mag = np.abs(data)
            data_phs = np.angle(data)
        # extend dimensions if necessary
        while data_mag.ndim < 3: data_mag = data_mag[...,np.newaxis]
        while data_phs.ndim < 3: data_phs = data_phs[...,np.newaxis]
        while data_ch_mag.ndim < 4: data_ch_mag = data_ch_mag[...,np.newaxis]
        while data_ch_phs.ndim < 4: data_ch_phs = data_ch_phs[...,np.newaxis]
    else:
        if sensmaps is None:
            data = bart(1, nufft_config, trj, data) # nufft
            if nc != 1:
                data = np.sqrt(np.sum(np.abs(data)**2, axis=-1)) # Sum of squares coil combination
        else:
            data = bart(1, pics_config , trj, data, sensmaps)
        data = np.abs(data)
        # make sure the data is at least 3D
        while data.ndim < 3:
            data = data[...,np.newaxis]
        # correct orientation at scanner (consistent with ICE)
        data = np.swapaxes(data, 0, 1)
        data = np.flip(data, (0,1,2))
        # Normalize and convert to int16
        data *= 32767/data.max()
        data = np.around(data)
        data = data.astype(np.int16)

    logging.debug("Image data is size %s" % (data.shape,))
    if group[0].idx.slice == 0:
        np.save(debugFolder + "/" + "img.npy", data)

    # Set ISMRMRD Meta Attributes (only important for online reco)
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768',
                         'Keep_image_geometry':    '1'})
    xml = meta.serialize()
    
    images = []
    n_slc = hdr.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = hdr.encoding[0].encodingLimits.contrast.maximum + 1
    n_sets = hdr.encoding[0].encodingLimits.set.maximum + 1
    n_avgs = hdr.encoding[0].encodingLimits.average.maximum + 1

    # Format as ISMRMRD image data
    if simu:
        n_par = data_mag.shape[-1]
        for par in range(n_par):
            image_mag = ismrmrd.Image.from_array(data_mag[...,par], acquisition=group[0])
            image_mag.image_series_index = 1
            image_phs = ismrmrd.Image.from_array(data_phs[...,par], acquisition=group[0])
            image_phs.image_series_index = 2
            images.append(image_mag)
            images.append(image_phs)
            image_ch_mag = ismrmrd.Image.from_array(data_ch_mag[...,par,:], acquisition=group[0])
            image_ch_mag.image_series_index = 3
            image_ch_phs = ismrmrd.Image.from_array(data_ch_phs[...,par,:], acquisition=group[0])
            image_ch_phs.image_series_index = 4
            images.append(image_ch_phs)
            images.append(image_ch_mag)

    else:
        n_par = data.shape[-1]
        if n_par > 1:
            for par in range(n_par):
                image = ismrmrd.Image.from_array(data[...,par], acquisition=group[0])
                image.image_index = 1 + group[0].idx.contrast * n_par + par
                image.image_series_index = 1 + group[0].idx.average *n_sets + group[0].idx.set
                image.slice = 0
                image.attribute_string = xml
                image.field_of_view = (ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.x), 
                                    ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.y), 
                                    ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.z))
                images.append(image)
        else:
            image = ismrmrd.Image.from_array(data[...,0], acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
            image.image_series_index = 1 + group[0].idx.average *n_sets + group[0].idx.set
            image.slice = 0
            image.attribute_string = xml
            image.field_of_view = (ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.x), 
                                    ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.y), 
                                    ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.z))
            images.append(image)

    return images

def process_acs(group, hdr, dmtx=None, gpu=False):
    if len(group)>0:
        data, trj = sort_data(group, dmtx)

        nx = hdr.encoding[0].encodedSpace.matrixSize.x
        ny = hdr.encoding[0].encodedSpace.matrixSize.y
        nz = hdr.encoding[0].encodedSpace.matrixSize.z

        if gpu and data.shape[2]>1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
            nufft_config = 'nufft -g -i -l 0.001 -t'
            ecalib_config = 'ecalib -g -m 1 -I'
        else:
            nufft_config = 'nufft -i -l 0.001 -t'
            ecalib_config = 'ecalib -m 1 -I'

        # nufft
        try:
            sensmaps = bart(1, nufft_config, trj, data)
        except: # fall back to CPU, if GPU doesnt work
            nufft_config = 'nufft -i -l 0.001 -t'
            sensmaps = bart(1, nufft_config, trj, data)

        np.save(debugFolder + "/" + "acs_img.npy", sensmaps)
        if sensmaps.ndim == 2:
            sensmaps = cfftn(sensmaps, [0, 1]) # back to Cartesian k-space
        else:
            sensmaps = cfftn(sensmaps, [0, 1, 2])
            
        while sensmaps.ndim < 4:
            sensmaps = sensmaps[...,np.newaxis]
        sensmaps = bart(1, 'resize -c 0 %d 1 %d 2 %d'%(nx, ny, nz), sensmaps)

        # ESPIRiT calibration
        try:
            sensmaps = bart(1, ecalib_config, sensmaps)
        except: # if the array is too big for the GPU RAM (especially for a large 3D ACS scan) we do it on CPU
            ecalib_config = 'ecalib -m 1 -I'
            sensmaps = bart(1, ecalib_config, sensmaps)

        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None

# %%
#########################
# Sort Data
#########################

def sort_data(group, dmtx=None):
    
    sig = list()
    trj = list()
    for acq in group:
       
        # data with optional prewhitening
        if dmtx is None:
            sig.append(acq.data)
        else:
            sig.append(apply_prewhitening(acq.data, dmtx))

        # trajectory
        traj = np.swapaxes(acq.traj,0,1)[:3] # [samples, dims] to [dims, samples]
        trj.append(traj)
   
    # convert lists to numpy arrays
    trj = np.asarray(trj) # current size: (nacq, 3, ncol)
    sig = np.asarray(sig) # current size: (nacq, ncha, ncol)
    if trj.ndim == 2: trj = trj[np.newaxis]
    if sig.ndim == 2: sig = sig[np.newaxis]

    # rearrange trj & sig for bart
    trj = np.transpose(trj, [1, 2, 0]) # [3, ncol, nacq]
    sig = np.transpose(sig, [2, 0, 1])[np.newaxis] # [1, ncol, nacq, ncha]
    
    np.save(debugFolder + "/" + "trj.npy", trj)
    np.save(debugFolder + "/" + "raw.npy", sig)

    return sig, trj

def intp_sensmaps(sensmap_coil, sens_fov, hdr):

    fov_x = int(hdr.encoding[0].reconSpace.fieldOfView_mm.x)
    fov_y = int(hdr.encoding[0].reconSpace.fieldOfView_mm.y)
    fov_z = int(hdr.encoding[0].reconSpace.fieldOfView_mm.z)

    nx = hdr.encoding[0].encodedSpace.matrixSize.x
    ny = hdr.encoding[0].encodedSpace.matrixSize.y
    nz = hdr.encoding[0].encodedSpace.matrixSize.z

    from skimage.transform import resize
    
    # Resize to full fov
    if sensmap_coil.shape[2] == 1:
        newshape = [sens_fov, sens_fov, 1]
    else:
        newshape = [sens_fov, sens_fov, sens_fov]
    sensmap_coil = resize(sensmap_coil.real, newshape, anti_aliasing=True) + 1j*resize(sensmap_coil.imag, newshape, anti_aliasing=True)

    # Cut out measurement fov
    if fov_x > sens_fov or fov_y > sens_fov or fov_z > sens_fov:
        logging.debug("WARNING: Sensitivity map FOV is smaller than measurement FOV")
    slc_x = (sens_fov-fov_x)//2
    if slc_x == 0: 
        slc_x = slice(None,None)
    else:
        slc_x = slice(slc_x,-slc_x)
    slc_y = (sens_fov-fov_y)//2
    if slc_y == 0: 
        slc_y = slice(None,None)
    else:
        slc_y = slice(slc_y,-slc_y)
    slc_z = (sens_fov-fov_z)//2
    if slc_z == 0: 
        slc_z = slice(None,None)
    else:
        slc_z = slice(slc_z,-slc_z)
    if sensmap_coil.shape[2] == 1:
        sensmap_coil = sensmap_coil[slc_x, slc_y]
    else:
        sensmap_coil = sensmap_coil[slc_x, slc_y, slc_z]

    # Interpolate to reconstruction matrix
    newshape = [nx, ny, nz]
    sensmap_coil = resize(sensmap_coil.real, newshape, anti_aliasing=True) + 1j*resize(sensmap_coil.imag, newshape, anti_aliasing=True)

    return sensmap_coil

################################
# Check if data is sampled on Cartesian grid
# atm this is not used, as the check might be not robust enough and sorting the data into a correct Cartesian grid is complicated
################################

def check_cart_grid(trj, phs_enc):

    """ Check if trajectory is on Cartesian grid

    trj: trajectory [dims, readout, phase]
    phs_enc: list with number of phase encoding steps [PE1, PE2]
    """

    pe1 = phs_enc[0]
    pe2 = phs_enc[1]
    if pe2 > 1:
        dim3 = True
    else:
        dim3 = False
    
    axes = [0,1,2]
    # if 2D, one dimension does not change
    if np.allclose(trj[0], trj[0,0,0], atol=1e-2):
        axes.remove(0)
        dim3 = False
    if np.allclose(trj[1], trj[1,0,0], atol=1e-2):
        axes.remove(1)
        dim3 = False
    if np.allclose(trj[2], trj[2,0,0], atol=1e-2):
        axes.remove(2)
        dim3 = False

    cart_grid = True
    # check readout direction
    for k in range(trj.shape[1]):
        for ax in axes:
            if np.allclose(trj[ax,k,:],trj[ax,k,0], atol=1e-2):
                readout = ax
                continue
            else:
                cart_grid = False
                break
    if cart_grid:
        # check 1st phase encoding direction
        axes.remove(readout)
        for k in range(trj.shape[2]):
            for ax in axes:
                if np.allclose(trj[ax,:,k],trj[ax,0,k], atol=1e-2):
                    phs_enc1 = ax
                    continue
                else:
                    cart_grid = False
                    break
    if cart_grid and dim3:
        axes.remove(phs_enc1)
        trj_tmp = trj.reshape([trj.shape[0],trj.shape[1],pe1,pe2], order='f')
        for k in range(trj_tmp.shape[3]):
            if(np.allclose(trj_tmp[axes[0],:,:,k],trj_tmp[axes[0],0,0,k], atol=1e-2)):
                continue
            else:
                cart_grid = False
    return cart_grid
