
import ismrmrd
import os
import logging
import numpy as np
import ctypes

from bart import bart
from cfft import cfftn, cifftn
from pulseq_helper import insert_hdr, insert_acq, get_ismrmrd_arrays, read_acqs
from reco_helper import calculate_prewhitening, apply_prewhitening, calc_rotmat, pcs_to_gcs, remove_os, remove_os_spiral
from reco_helper import fov_shift_spiral_reapply
from reco_helper import filt_ksp
from DreamMap import calc_fa, DREAM_filter_fid

from skimage import filters
from scipy.ndimage import binary_fill_holes, binary_dilation

""" Spiral subscript to reconstruct dream B1 map
"""

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

########################
# Main Function
########################

def process_spiral_dream(connection, config, metadata, prot_file):
    
    logging.debug("Spiral DREAM reconstruction")
    
    # Insert protocol header
    insert_hdr(prot_file, metadata)
    
    logging.info("Config: \n%s", config)

    # Check for GPU availability
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        gpu = True
    else:
        gpu = False

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier

    try:
        # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))
        # logging.info("Metadata: \n%s", metadata.serialize())

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("Trajectory type '%s', matrix size (%s x %s x %s), field of view (%s x %s x %s)mm^3", 
            metadata.encoding[0].trajectory.value, 
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
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1

    acqGroup = [[[] for _ in range(n_slc)] for _ in range(n_contr)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * n_slc
    dmtx = None

    # read protocol arrays
    prot_arrays = get_ismrmrd_arrays(prot_file)
    up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}

    # for B1 Dream map
    process_raw.imagesets = [None] * n_contr
    process_raw.rawdata = [None] * n_contr
    
    # different contrasts need different scaling
    process_raw.imascale = [None] * n_contr
    
    # parameters for reapplying FOV shift
    nsegments = metadata.encoding[0].encodingLimits.segment.maximum + 1
    matr_sz = np.array([metadata.encoding[0].encodedSpace.matrixSize.x, metadata.encoding[0].encodedSpace.matrixSize.y])
    res = np.array([metadata.encoding[0].encodedSpace.fieldOfView_mm.x / matr_sz[0], metadata.encoding[0].encodedSpace.fieldOfView_mm.y / matr_sz[1], 1])

    base_trj = None

    # read protocol acquisitions - faster than doing it one by one
    logging.debug("Reading in protocol acquisitions.")
    acqs = read_acqs(prot_file)

    try:
        for item in connection:

            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                # insert acquisition protocol
                # base_trj is used to correct FOV shift (see below)
                base_traj = insert_acq(acqs[0], item, metadata)
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
                                   
                # Accumulate all imaging readouts in a group
                if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # skope sync scans
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                        sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], metadata, dmtx, gpu)
                        acsGroup[item.idx.slice].clear()
                    continue
                
                # Copy sensitivity maps if less slices were acquired (in 2D sequence)
                n_sensmaps = len([x for x in sensmaps if x is not None])
                if (n_sensmaps != len(sensmaps) and n_sensmaps != 0):                    
                    if (n_slc % 2 == 0):
                        for i in range(0,n_slc-1,2):
                            sensmaps[i] = sensmaps[i+1].copy()
                    else:
                        for i in range(1,n_slc,2):
                            sensmaps[i] = sensmaps[i-1].copy()
                    
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
                    # Reapply FOV Shift with predicted trajectory
                    sig = acqGroup[item.idx.contrast][item.idx.slice][-1].data[:]
                    acqGroup[item.idx.contrast][item.idx.slice][-1].data[:] = fov_shift_spiral_reapply(sig, pred_trj, base_trj, shift, matr_sz)
                    # remove ADC oversampling
                    os_factor = up_double["os_factor"] if "os_factor" in up_double else 1
                    if os_factor == 2:
                        remove_os_spiral(acqGroup[item.idx.contrast][item.idx.slice][-1])


                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    logging.info("Processing a group of k-space data")
                    images = process_raw(acqGroup[item.idx.contrast][item.idx.slice], metadata, dmtx, sensmaps[item.idx.slice], gpu, prot_arrays)
                    logging.debug("Sending images to client.")
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
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], metadata, dmtx) 
                image = process_raw(acqGroup[item.idx.contrast][item.idx.slice], metadata, dmtx, sensmaps[item.idx.slice])
                logging.debug("Sending image to client:\n%s", image)
                connection.send_image(image)
                acqGroup = []

    finally:
        connection.send_close()

# %%
#########################
# Process Data
#########################

def process_raw(group, metadata, dmtx=None, sensmaps=None, gpu=False, prot_arrays=None):

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    
    rNx = metadata.encoding[0].reconSpace.matrixSize.x
    rNy = metadata.encoding[0].reconSpace.matrixSize.y
    rNz = metadata.encoding[0].reconSpace.matrixSize.z

    data, trj = sort_spiral_data(group, dmtx)
    process_raw.rawdata[group[0].idx.contrast] = data.copy() # save rawdata of the two contrasts for FID filter calculation

    if gpu and nz>1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
        nufft_config = 'nufft -g -i -l 0.005 -t -d %d:%d:%d'%(nx, nx, nz)
        # pics_config = 'pics -g -S -e -R T:7:0:.0001 -i 50 -t'
        pics_config = 'pics -g -S -e -l1 0.0005  -i 50 -t'
    else:
        nufft_config = 'nufft -i -m 20 -l 0.005 -c -t -d %d:%d:%d'%(nx, nx, nz)
        # pics_config = 'pics -S -e -R T:7:0:.0001 -i 50 -t'
        pics_config = 'pics -g -S -e -l1 0.0005  -i 50 -t'

    # dream array : [ste_contr,flip_angle_ste,TR,flip_angle,prepscans,t1]
    dream = prot_arrays['dream']
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    ste_ix = int(dream[0])
    fid_ix = n_contr-1-ste_ix
    if ste_ix != 0:
        raise ValueError(f"This reconstruction currently supports only STE first, but STE has index {ste_ix} and FID has index {fid_ix}.")

    # FID filter
    filt_fid = True 
    if group[0].idx.contrast == fid_ix and dream.size > 2 and filt_fid:
        logging.info("Global FID filter activated.")
        # Blurring compensation parameters
        alpha = dream[1]     # preparation FA
        tr = dream[2]        # [s]
        beta = dream[3]      # readout FA
        dummies = dream[4]   # number of dummy scans before readout echo train starts
        # T1 estimate:
        t1 = dream[5]        # [s] - approximately Gufi Phantom at 7T
    
        ste_data = np.asarray(process_raw.rawdata[ste_ix])
        fid_data = np.asarray(process_raw.rawdata[fid_ix])
        mean_alpha = calc_fa(ste_data.mean(), fid_data.mean())

        mean_beta = mean_alpha / alpha * beta
        shot = group[0].idx.set
        ctr = 0
        for i,acq in enumerate(group):
            ctr += 1
            if acq.idx.set != shot: # reset counter at start of new shot (= new STEAM prep)
                ctr = 0
            shot = acq.idx.set

            ti = tr * (dummies + ctr) # TI estimate (time from STEAM prep to readout) [s]
            # Global filter:
            filt = DREAM_filter_fid(mean_alpha, mean_beta, tr, t1, ti)
            # apply filter:
            data[:,:,i,:] *= filt

    if sensmaps is None:                
        data = bart(1, nufft_config, trj, data)
        data = np.sqrt(np.sum(np.abs(data)**2, axis=-1))
    else:
        data = bart(1, pics_config , trj, data, sensmaps)
        data = np.abs(data)
    while np.ndim(data) < 3:
        data = data[..., np.newaxis]
    if nz > rNz:
        # remove oversampling in slice direction
        data = data[:,:,(nz - rNz)//2:-(nz - rNz)//2]

    logging.debug("Image reconstructed with size %s" % (data.shape,))
    
    # Check if both contrasts were reconstructed for calculation of FA map
    process_raw.imagesets[group[0].idx.contrast] = data.copy()
    full_set_check = all(elem is not None for elem in process_raw.imagesets)
    if full_set_check:
        logging.info("Calculation of B1 map.")
        ste = np.asarray(process_raw.imagesets[ste_ix])
        fid = np.asarray(process_raw.imagesets[fid_ix])

        # image mask for fa-map (use of filtered or unfiltered fid) - not used atm
        mask = np.zeros(fid.shape)
        for nz in range(fid.shape[-1]):
            val = filters.threshold_otsu(fid[:,:,nz])
            otsu = fid[:,:,nz] > val
            imfill = binary_fill_holes(otsu) * 1
            mask[:,:,nz] = binary_dilation(imfill)
                
        # FA map and RefVoltage map
        fa_map = calc_fa(abs(ste), abs(fid))
        # fa_map *= mask
        up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}
        current_refvolt = up_double["RefVoltage"]
        nom_fa = dream[1]
        logging.info("current_refvolt = %sV and nom_fa = %s°^" % (current_refvolt, nom_fa))
        ref_volt = current_refvolt * (nom_fa/fa_map)
        
        fa_map *= 10 # increase dynamic range
        fa_map = np.around(fa_map)
        fa_map = fa_map.astype(np.int16)
        logging.debug("fa map is size %s" % (fa_map.shape,))
        
        ref_volt = np.around(ref_volt)
        ref_volt = ref_volt.astype(np.int16)
        logging.debug("ref_volt map is size %s" % (ref_volt.shape,))
        
        process_raw.imagesets = [None] * n_contr # free list
        process_raw.rawdata   = [None] * n_contr # free list

        # correct orientation at scanner (consistent with ICE)
        fa_map = np.swapaxes(fa_map, 0, 1)
        fa_map = np.flip(fa_map, (0,1,2))
        ref_volt = np.swapaxes(ref_volt, 0, 1)
        ref_volt = np.flip(ref_volt, (0,1,2))
    else:
        fa_map = None
        ref_volt = None
    
    # correct orientation at scanner (consistent with ICE)
    data = np.swapaxes(data, 0, 1)
    data = np.flip(data, (0,1,2))

    # Normalize and convert to int16
    #save one scaling in 'static' variable
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
                         'Keep_image_geometry':    1})
    xml = meta.serialize()

    meta2 = ismrmrd.Meta({'DataRole':               'Quantitative',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '512',
                         'WindowWidth':            '1024',
                         'Keep_image_geometry':    1})
    xml2 = meta2.serialize()

    meta3 = ismrmrd.Meta({'DataRole':               'Quantitative',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '512',
                         'WindowWidth':            '1024',
                         'Keep_image_geometry':    1})
    xml3 = meta3.serialize()
    
    images = []
    n_par = data.shape[-1]
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    
    # Format as ISMRMRD image data
    if n_par > 1:
        for par in range(n_par):
            image = ismrmrd.Image.from_array(data[...,par], acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_par + par
            image.image_series_index = 1
            image.slice = 0
            image.contrast = group[0].idx.contrast
            image.attribute_string = xml
            image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
            images.append(image)
        
        if fa_map is not None:
            for par in range(n_par):
                image = ismrmrd.Image.from_array(fa_map[...,par], acquisition=group[0])
                image.image_index = 1 + par
                image.image_series_index = 2
                image.slice = 0
                image.attribute_string = xml2
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
                images.append(image)
        
        if ref_volt is not None:
            for par in range(n_par):
                image = ismrmrd.Image.from_array(ref_volt[...,par], acquisition=group[0])
                image.image_index = 1 + par
                image.image_series_index = 3
                image.slice = 0
                image.attribute_string = xml3
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
                images.append(image)
        
    else:
        image = ismrmrd.Image.from_array(data[...,0], acquisition=group[0])
        image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
        image.image_series_index = 1
        image.slice = 0
        image.contrast = group[0].idx.contrast
        image.attribute_string = xml
        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        images.append(image)
        
        if fa_map is not None:
            image = ismrmrd.Image.from_array(fa_map[...,0], acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
            image.image_series_index = 2
            image.slice = 0
            image.attribute_string = xml2
            image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
            images.append(image)
        
        if ref_volt is not None:
            image = ismrmrd.Image.from_array(ref_volt[...,0], acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
            image.image_series_index = 3
            image.slice = 0
            image.attribute_string = xml3
            image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
            images.append(image)

    return images

def process_acs(group, metadata, dmtx=None, gpu=False):
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
        data = data[...,0] # currently only first contrast used
    
        if gpu and data.shape[2]>1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
            sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I', data)  # ESPIRiT calibration
        else:
            sensmaps = bart(1, 'ecalib -m 1 -k 6 -I', data)  # ESPIRiT calibration
        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None

# %%
#########################
# Sort Data
#########################

def sort_spiral_data(group, dmtx=None):
    
    sig = list()
    trj = list()
    enc = list()   
    for acq in group:

        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
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
    
    # readout filter to remove Gibbs ringing
    for nacq in range(sig.shape[0]):
        sig[nacq,:,:] = filt_ksp(kspace=sig[nacq,:,:], traj=trj[nacq,:,:], filt_fac=1)
    
    # rearrange trj & sig for bart
    trj = np.transpose(trj, [1, 2, 0]) # [3, ncol, nacq]
    sig = np.transpose(sig, [2, 0, 1])[np.newaxis] # [1, ncol, nacq, ncha]
    logging.debug("Trajectory shape = %s , Signal Shape = %s "%(trj.shape, sig.shape))
    
    np.save(debugFolder + "/" + "trj.npy", trj)

    return sig, trj

def sort_into_kspace(group, metadata, dmtx=None, zf_around_center=False):
    # initialize k-space
    enc1_min, enc1_max = int(999), int(0)
    enc2_min, enc2_max = int(999), int(0)
    contr_max = 0
    for acq in group:
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        contr = acq.idx.contrast
        if enc1 < enc1_min:
            enc1_min = enc1
        if enc1 > enc1_max:
            enc1_max = enc1
        if enc2 < enc2_min:
            enc2_min = enc2
        if enc2 > enc2_max:
            enc2_max = enc2
        if contr > contr_max:
            contr_max = contr

        # Oversampling removal - WIP: assumes 2x oversampling at the moment
        data = remove_os(acq.data[:], axis=-1)
        acq.resize(number_of_samples=data.shape[-1], active_channels=data.shape[0])
        acq.data[:] = data

    nc = metadata.acquisitionSystemInformation.receiverChannels
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    n_contr = contr_max + 1

    kspace = np.zeros([ny, nz, nc, nx, n_contr], dtype=group[0].data.dtype)
    counter = np.zeros([ny, nz], dtype=np.uint16)

    logging.debug("nx/ny/nz: %s/%s/%s; enc1 min/max: %s/%s; enc2 min/max:%s/%s, ncol: %s" % (nx, ny, nz, enc1_min, enc1_max, enc2_min, enc2_max, group[0].data.shape[-1]))
    
    for acq in group:
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        contr = acq.idx.contrast

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
            kspace[enc1, enc2, :, col, contr] += acq.data
        else:
            kspace[enc1, enc2, :, col, contr] += apply_prewhitening(acq.data, dmtx)
        if contr==0:
            counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc, ncontr)
    kspace = np.transpose(kspace, [3, 0, 1, 2, 4])

    return kspace
