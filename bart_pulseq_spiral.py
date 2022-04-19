
import ismrmrd
import os
import itertools
import logging
import numpy as np
import base64
import ctypes

from bart import bart
from cfft import cfftn, cifftn
from pulseq_helper import insert_hdr, insert_acq, read_acqs
from reco_helper import calculate_prewhitening, apply_prewhitening, calc_rotmat, fov_shift_spiral_reapply, pcs_to_gcs, remove_os
from reco_helper import fov_shift_spiral_reapply #, fov_shift_spiral, fov_shift 

""" Reconstruction of imaging data acquired with the Pulseq Sequence via the FIRE framework
    Reconstruction is done with the BART toolbox

"""


# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

########################
# Main Function
########################

def process_spiral(connection, config, hdr, meta_file):
  
    # Set a slice for single slice reconstruction
    slc_sel = None

    # Insert metadata header
    insert_hdr(meta_file, hdr)
    
    logging.info("Config: \n%s", config)

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

    acqGroup = [[[] for _ in range(n_slc)] for _ in range(n_contr)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * n_slc
    dmtx = None

    # different contrasts need different scaling
    process_raw.imascale = [None] * 256

    # parameters for reapplying FOV shift
    nsegments = hdr.encoding[0].encodingLimits.segment.maximum + 1
    matr_sz = np.array([hdr.encoding[0].encodedSpace.matrixSize.x, hdr.encoding[0].encodedSpace.matrixSize.y])
    res = np.array([hdr.encoding[0].encodedSpace.fieldOfView_mm.x / matr_sz[0], hdr.encoding[0].encodedSpace.fieldOfView_mm.y / matr_sz[1], 1])

    base_trj = None

    # read acquisitions - faster than doing it one by one
    logging.debug("Reading in metadata acquisitions.")
    acqs = read_acqs(meta_file)

    try:
        for item in connection:

            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                # insert metadata in acquisitions
                # base_trj is used to correct FOV shift (see below)
                base_traj = insert_acq(acqs[0], item, hdr)
                acqs.pop(0)
                if base_traj is not None:
                    base_trj = base_traj

                # run noise decorrelation
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
                    
                # skip slices in single slice reconstruction
                if slc_sel is not None and item.idx.slice != slc_sel:
                    continue
                
                # Accumulate all imaging readouts in a group
                if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # skope sync scans
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                        # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                        sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], hdr, dmtx, gpu)
                        acsGroup[item.idx.slice].clear()
                    continue

                if item.idx.segment == 0:
                    acqGroup[item.idx.contrast][item.idx.slice].append(item)

                    # for reapplying FOV shift (see below)
                    pred_trj = item.traj[:]
                    rotmat = calc_rotmat(item)
                    shift = pcs_to_gcs(np.asarray(item.position), rotmat) / res
                else:
                    # append data to first segment of ADC group
                    idx_lower = item.idx.segment * item.number_of_samples
                    idx_upper = (item.idx.segment+1) * item.number_of_samples
                    acqGroup[item.idx.contrast][item.idx.slice][-1].data[:,idx_lower:idx_upper] = item.data[:]
                if item.idx.segment == nsegments - 1:
                    # Reapply FOV Shift with predicted trajectory - only works, if GIRF trajectory prediction was used
                    sig = acqGroup[item.idx.contrast][item.idx.slice][-1].data[:]
                    acqGroup[item.idx.contrast][item.idx.slice][-1].data[:] = fov_shift_spiral_reapply(sig, pred_trj, base_trj, shift, matr_sz)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    logging.info("Processing a group of k-space data")
                    images = process_raw(acqGroup[item.idx.contrast][item.idx.slice], hdr, dmtx, sensmaps[item.idx.slice], gpu)
                    logging.debug("Sending images to client:\n%s", images)
                    connection.send_image(images)
                    acqGroup[item.idx.contrast][item.idx.slice].clear() # free memory

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # just pass along
                connection.send_image(item)
                continue

            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Extract raw ECG waveform data. Basic sorting to make sure that data 
        # is time-ordered, but no additional checking for missing data.
        # ecgData has shape (5 x timepoints)
        if len(waveformGroup) > 0:
            waveformGroup.sort(key = lambda item: item.time_stamp)
            ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
            ecgData = np.concatenate(ecgData,1)

        # Process any remaining groups of raw or image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if item is not None:
            if len(acqGroup[item.idx.contrast][item.idx.slice]) > 0:
                logging.info("Processing a group of k-space data (untriggered)")
                if sensmaps[item.idx.slice] is None:
                    # run parallel imaging calibration
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], hdr, dmtx) 
                image = process_raw(acqGroup[item.idx.contrast][item.idx.slice], hdr, dmtx, sensmaps[item.idx.slice])
                logging.debug("Sending image to client:\n%s", image)
                connection.send_image(image)
                acqGroup = []

    finally:
        connection.send_close()

# %%
#########################
# Process Data
#########################

def process_raw(group, hdr, dmtx=None, sensmaps=None, gpu=False):

    nx = hdr.encoding[0].encodedSpace.matrixSize.x
    ny = hdr.encoding[0].encodedSpace.matrixSize.y
    nz = hdr.encoding[0].encodedSpace.matrixSize.z
    
    rNx = hdr.encoding[0].reconSpace.matrixSize.x
    rNy = hdr.encoding[0].reconSpace.matrixSize.y
    rNz = hdr.encoding[0].reconSpace.matrixSize.z

    data, trj = sort_spiral_data(group, hdr, dmtx)
    
    if gpu and nz>1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
        nufft_config = 'nufft -g -i -m 15 -l 0.005 -t -d %d:%d:%d'%(nx, nx, nz)
        ecalib_config = 'ecalib -g -m 1 -I'
        pics_config = 'pics -g -S -e -l1 -r 0.001 -i 50 -t'
    else:
        nufft_config = 'nufft -i -m 15 -l 0.005 -t -d %d:%d:%d'%(nx, nx, nz)
        ecalib_config = 'ecalib -m 1 -I'
        pics_config = 'pics -S -e -l1 -r 0.001 -i 50 -t'

    force_pics = False
    if sensmaps is None and force_pics:
        sensmaps = bart(1, nufft_config, trj, data) # nufft
        sensmaps = cfftn(sensmaps, [0, 1, 2]) # back to k-space
        sensmaps = bart(1, ecalib_config, sensmaps)  # ESPIRiT calibration

    if sensmaps is None:
        logging.debug("no pics necessary, just do standard recon")
            
        # bart nufft
        data = bart(1, nufft_config, trj, data) # nufft

        # Sum of squares coil combination
        data = np.sqrt(np.sum(np.abs(data)**2, axis=-1))
    else:
        data = bart(1, pics_config , trj, data, sensmaps)
        data = np.abs(data)
        # make sure that data is at least 3d:
        while np.ndim(data) < 3:
            data = data[..., np.newaxis]
    
    if group[0].idx.slice == 0 and sensmaps is not None:
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)

    if nz > rNz:
        # remove oversampling in slice direction
        data = data[:,:,(nz - rNz)//2:-(nz - rNz)//2]

    logging.debug("Image data is size %s" % (data.shape,))
    if group[0].idx.slice == 0:
        np.save(debugFolder + "/" + "img.npy", data)

    # correct orientation at scanner (consistent with ICE)
    data = np.swapaxes(data, 0, 1)
    data = np.flip(data, (0,1,2))

    # Normalize and convert to int16
    # save one scaling in 'static' variable
    contr = group[0].idx.contrast
    if process_raw.imascale[contr] is None:
        process_raw.imascale[contr] = 0.8 / data.max()
    data *= 32767 * process_raw.imascale[contr]
    data = np.around(data)
    data = data.astype(np.int16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768',
                         'Keep_image_geometry':    '1'})
    xml = meta.serialize()
    
    images = []
    n_par = data.shape[-1]
    n_slc = hdr.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = hdr.encoding[0].encodingLimits.contrast.maximum + 1

    # Format as ISMRMRD image data
    if n_par > 1:
        for par in range(n_par):
            image = ismrmrd.Image.from_array(data[...,par], acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_par + par
            image.image_series_index = 1 + group[0].idx.repetition
            image.slice = 0
            image.attribute_string = xml
            image.field_of_view = (ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.z))
            images.append(image)
    else:
        image = ismrmrd.Image.from_array(data[...,0], acquisition=group[0])
        image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
        image.image_series_index = 1 + group[0].idx.repetition
        image.slice = 0
        image.attribute_string = xml
        image.field_of_view = (ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(hdr.encoding[0].reconSpace.fieldOfView_mm.z))
        images.append(image)

    return images

def process_acs(group, hdr, dmtx=None, gpu=False):
    if len(group)>0:
        data = sort_into_kspace(group, hdr, dmtx, zf_around_center=True)

        if gpu and data.shape[2] > 1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
            sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I', data)  # ESPIRiT calibration, WIP: use smaller radius -r ?
        else:
            sensmaps = bart(1, 'ecalib -m 1 -k 6 -I', data)

        refimg = cifftn(data,axes=[0,1,2])
        np.save(debugFolder + "/" + "refimg.npy", refimg)

        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None

# %%
#########################
# Sort Data
#########################

def sort_spiral_data(group, hdr, dmtx=None):
    
    nx = hdr.encoding[0].encodedSpace.matrixSize.x
    nz = hdr.encoding[0].encodedSpace.matrixSize.z
    res = hdr.encoding[0].reconSpace.fieldOfView_mm.x / hdr.encoding[0].encodedSpace.matrixSize.x
    rot_mat = calc_rotmat(group[0])

    sig = list()
    trj = list()
    enc = list()
    for acq in group:

        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        kz = enc2 - nz//2
        enc.append([enc1, enc2])
        
        # append data after optional prewhitening
        if dmtx is None:
            sig.append(acq.data)
        else:
            sig.append(apply_prewhitening(acq.data, dmtx))

        # update trajectory
        traj = np.swapaxes(acq.traj[:,:3],0,1) # [samples, dims] to [dims, samples]
        trj.append(traj)

    np.save(debugFolder + "/" + "enc.npy", enc)
    
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

def sort_into_kspace(group, hdr, dmtx=None, zf_around_center=False):
    # initialize k-space
    enc1_min, enc1_max = int(999), int(0)
    enc2_min, enc2_max = int(999), int(0)
    for acq in group:
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        if enc1 < enc1_min:
            enc1_min = enc1
        if enc1 > enc1_max:
            enc1_max = enc1
        if enc2 < enc2_min:
            enc2_min = enc2
        if enc2 > enc2_max:
            enc2_max = enc2

        # Oversampling removal - WIP: assumes 2x oversampling at the moment
        data = remove_os(acq.data[:], axis=-1)
        acq.resize(number_of_samples=data.shape[-1], active_channels=data.shape[0])
        acq.data[:] = data

    nc = hdr.acquisitionSystemInformation.receiverChannels
    nx = hdr.encoding[0].encodedSpace.matrixSize.x
    ny = hdr.encoding[0].encodedSpace.matrixSize.y
    nz = hdr.encoding[0].encodedSpace.matrixSize.z

    kspace = np.zeros([ny, nz, nc, nx], dtype=group[0].data.dtype)
    counter = np.zeros([ny, nz], dtype=np.uint16)

    logging.debug("nx/ny/nz: %s/%s/%s; enc1 min/max: %s/%s; enc2 min/max:%s/%s, ncol: %s" % (nx, ny, nz, enc1_min, enc1_max, enc2_min, enc2_max, group[0].data.shape[-1]))

    for acq in group:
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2

        # in case dim sizes smaller than expected, sort data into k-space center (e.g. for reference scans)
        ncol = acq.data.shape[-1]
        cx = nx // 2
        ccol = ncol // 2
        col = slice(cx - ccol, cx + ccol)

        if zf_around_center:
            cy = ny // 2
            cz = nz // 2

            cenc1 = (enc1_max+1) // 2
            cenc2 = (enc2_max+1) // 2

            # sort data into center k-space (assuming a symmetric acquisition)
            enc1 += cy - cenc1
            enc2 += cz - cenc2
        
        if dmtx is None:
            kspace[enc1, enc2, :, col] += acq.data
        else:
            kspace[enc1, enc2, :, col] += apply_prewhitening(acq.data, dmtx)
        counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc)
    kspace = np.transpose(kspace, [3, 0, 1, 2])

    return kspace
