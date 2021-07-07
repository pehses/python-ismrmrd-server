
""" Pulseq Reco for Cartesian datasets
"""

import ismrmrd
import os
import itertools
import logging
import numpy as np
from cfft import cifftn, cfftn
import base64

from bart import bart
from reco_helper import calculate_prewhitening, apply_prewhitening, fov_shift, calc_rotmat, pcs_to_gcs, remove_os
from pulseq_prot import insert_hdr, insert_acq, get_ismrmrd_arrays
from DreamMap import global_filter, calc_fa

from skimage import filters
import scipy

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

def process_cartesian_dream(connection, config, metadata, prot_file):
    
    logging.debug("in process_cartesian_dream")
    
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
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1

    acqGroup = [[[] for _ in range(n_slc)] for _ in range(n_contr)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * 256
    dmtx = None
    
    # read protocol arrays
    prot_arrays = get_ismrmrd_arrays(prot_file)

    # for B1 Dream map
    if "dream" in prot_arrays:
        process_raw.imagesets = [None] * n_contr
        process_raw.fid_unfiltered = True
    
    # different contrasts need different scaling
    process_raw.imascale = [None] * 256

    try:
        for acq_ctr, item in enumerate(connection):
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                insert_acq(prot_file, item, acq_ctr, noncartesian=False)

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
                    continue
                elif sensmaps[item.idx.slice] is None:
                    # run parallel imaging calibration
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx, gpu)
                    acsGroup[item.idx.slice].clear()

                acqGroup[item.idx.contrast][item.idx.slice].append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup[item.idx.contrast][item.idx.slice], config, metadata, dmtx, sensmaps[item.idx.slice], prot_arrays, gpu)
                    logging.debug("Sending image to client:\n%s", image)
                    connection.send_image(image)
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
            if len(acqGroup) > 0:
                logging.info("Processing a group of k-space data (untriggered)")
                if sensmaps[item.idx.slice] is None:
                    # run parallel imaging calibration
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx)
                image = process_raw(acqGroup[item.idx.contrast][item.idx.slice], config, metadata, dmtx, sensmaps[item.idx.slice])
                logging.debug("Sending image to client:\n%s", image)
                connection.send_image(image)
                acqGroup = []

    finally:
        connection.send_close()


def sort_into_kspace(group, metadata, dmtx=None, zf_around_center=False):
    # initialize k-space
    nc = metadata.acquisitionSystemInformation.receiverChannels
    nx = group[0].number_of_samples

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

    nx = 2*metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z

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


def process_acs(group, config, metadata, dmtx=None, gpu=False):
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
        data = remove_os(data)

        # fov shift
        rotmat = calc_rotmat(group[0])
        if not rotmat.any(): rotmat = -1*np.eye(3) # compatibility if Pulseq rotmat not in protocol
        res = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x
        shift = pcs_to_gcs(np.asarray(group[0].position), rotmat) / res
        data = fov_shift(data, shift)

        if gpu:
            sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I', data)  # ESPIRiT calibration
        else:
            sensmaps = bart(1, 'ecalib -m 1 -k 6 -I', data)  # ESPIRiT calibration
        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None


def process_raw(group, config, metadata, dmtx=None, sensmaps=None, prot_arrays=None, gpu=False):

    data = sort_into_kspace(group, metadata, dmtx)
    data = remove_os(data)

    # fov shift
    rotmat = calc_rotmat(group[0])
    if not rotmat.any(): rotmat = -1*np.eye(3) # compatibility if Pulseq rotmat not in protocol
    res = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x
    shift = pcs_to_gcs(np.asarray(group[0].position), rotmat) / res
    data = fov_shift(data, shift)

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    if sensmaps is None:
        logging.debug("no pics necessary, just do standard fft")
        data = cifftn(data, axes=(0, 1, 2))

        # Sum of squares coil combination
        data = np.sqrt(np.sum(np.abs(data)**2, axis=-1))
    else:
        if gpu:
            data = bart(1, 'pics -g -S -e -l1 -r 0.001 -i 50', data, sensmaps)
        else:
            data = bart(1, 'pics -S -e -l1 -r 0.001 -i 50', data, sensmaps)
        data = np.abs(data)

    logging.debug("Image data is size %s" % (data.shape,))
    
    # B1 Map calculation (Dream approach)
    if 'dream' in prot_arrays: #dream = ([ste_contr,flip_angle_ste,TR,flip_angle,prepscans,t1])
        dream = prot_arrays['dream']
        n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
        
        process_raw.imagesets[group[0].idx.contrast] = data.copy()
        full_set_check = all(elem is not None for elem in process_raw.imagesets)
        if full_set_check:
            logging.info("B1 map calculation using Dream")
            ste = np.asarray(process_raw.imagesets[int(dream[0])])
            fid = np.asarray(process_raw.imagesets[int(n_contr-1-dream[0])])
            
            #image processing filter
            dil = np.zeros(fid.shape)
            for nz in range(fid.shape[-1]):
                # otsu
                val = filters.threshold_otsu(fid[:,:,nz])
                otsu = fid[:,:,nz] > val
                # fill holes
                imfill = scipy.ndimage.morphology.binary_fill_holes(otsu) * 1
                # dilation
                dil[:,:,nz] = scipy.ndimage.morphology.binary_dilation(imfill)
            
            if dream.size > 2 :                                 # without filter: dream = ([ste_contr,flip_angle_ste])
                logging.info("Global filter approach")
                
                # save unfiltered fid if wanted
                if process_raw.fid_unfiltered:
                    logging.debug("fid_unfiltered to scanner")
                    fid_unfilt = fid.copy()
                    np.save(debugFolder + "/" + "fid_unf.npy", fid_unfilt)
                else:
                    fid_unfilt = None
                # move from [nx,ny,nz] to [ny,nz,nx]
                ste = np.transpose(ste,[1,2,0])
                fid = np.transpose(fid,[1,2,0])

                # Blurring compensation parameters
                alpha = dream[1]     # preparation FA
                tr = dream[2]        # [s]
                beta = dream[3]      # readout FA
                dummies = dream[4]   # number of dummy scans before readout echo train starts
                # T1 estimate:
                t1 = dream[5]        # [s] - approximately Gufi Phantom at 7T
                # TI estimate (the time after DREAM preparation after which each k-space line is acquired):
                ti = np.zeros([metadata.encoding[0].encodedSpace.matrixSize.y, metadata.encoding[0].encodedSpace.matrixSize.z])
                for i,acq in enumerate(group):
                    ti[acq.idx.kspace_encode_step_1, acq.idx.kspace_encode_step_2] = i
                ti = tr * (dummies + ti) # [s]
                np.save(debugFolder + "/" + "ti.npy", ti)
                # Global filter:
                fa_map,fid_filt = global_filter(ste, fid, ti, alpha, beta, tr, t1)
                fa_map = np.transpose(fa_map, [2,0,1])
                # fid = fid_filt:
                fid_filt = np.transpose(fid_filt, [2,0,1])
                np.save(debugFolder + "/" + "fid_filt.npy", fid_filt)
                data = fid_filt.copy()
            else:
                fa_map = calc_fa(abs(ste), abs(fid))
                fid_unfilt = None
            
            fa_map *= dil
            current_refvolt = metadata.userParameters.userParameterDouble[5].value_
            nom_fa = dream[1]
            logging.info("current_refvolt = %sV und nom_fa = %s°^" % (current_refvolt, nom_fa))
            ref_volt = current_refvolt * (nom_fa/fa_map)
            
            fa_map = np.around(fa_map)
            fa_map = fa_map.astype(np.int16)
            np.save(debugFolder + "/" + "fa.npy", fa_map)
            logging.debug("fa map is size %s" % (fa_map.shape,))
            
            ref_volt = np.around(ref_volt)
            ref_volt = ref_volt.astype(np.int16)
            np.save(debugFolder + "/" + "ref_volt.npy", ref_volt)
            logging.debug("ref_volt map is size %s" % (ref_volt.shape,))
            
            process_raw.imagesets = [None] * n_contr # free list
        else:
            fa_map = None
            fid_unfilt = None
            ref_volt = None
    else:
        fa_map = None
        fid_unfilt = None
        ref_volt = None
        logging.info("no dream B1 mapping")

    # Normalize and convert to int16
    # save one scaling in 'static' variable

    contr = group[0].idx.contrast
    if process_raw.imascale[contr] is None:
        process_raw.imascale[contr] = 0.8 / data.max()
    data *= 32767 * process_raw.imascale[contr]
    data = np.around(data)
    data = data.astype(np.int16)

    np.save(debugFolder + "/" + "img.npy", data)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768'})
    xml = meta.serialize()

    # Format as ISMRMRD image data
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    
    n_par = data.shape[-1]
    images = []
    if n_par > 1:
        for par in range(n_par):
            image = ismrmrd.Image.from_array(data[...,par].T, acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_par + par
            image.image_series_index = 1
            image.slice = 0
            image.attribute_string = xml
            images.append(image)

        if fa_map is not None:
            for par in range(n_par):
                image = ismrmrd.Image.from_array(fa_map[...,par].T, acquisition=group[0])
                image.image_index = 1 + par
                image.image_series_index = 2
                image.slice = 0
                image.attribute_string = xml
                images.append(image)
        
        if fid_unfilt is not None:
            for par in range(n_par):
                image = ismrmrd.Image.from_array(fid_unfilt[...,par].T, acquisition=group[0])
                image.image_index = 1 + par
                image.image_series_index = 3
                image.slice = 0
                image.attribute_string = xml
                images.append(image)
        
        if ref_volt is not None:
            for par in range(n_par):
                image = ismrmrd.Image.from_array(ref_volt[...,par].T, acquisition=group[0])
                image.image_index = 1 + par
                image.image_series_index = 4
                image.slice = 0
                image.attribute_string = xml
                images.append(image)
        
    else:
        image = ismrmrd.Image.from_array(data[...,0].T, acquisition=group[0])
        image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
        image.image_series_index = 1
        image.slice = 0
        image.attribute_string = xml
        images.append(image)

        if fa_map is not None:
            image = ismrmrd.Image.from_array(fa_map[...,0].T, acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
            image.image_series_index = 2
            image.slice = 0
            image.attribute_string = xml
            images.append(image)
        
        if fid_unfilt is not None:
            image = ismrmrd.Image.from_array(fid_unfilt[...,0].T, acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
            image.image_series_index = 3
            image.slice = 0
            image.attribute_string = xml
            images.append(image)
        
        if ref_volt is not None:
            image = ismrmrd.Image.from_array(ref_volt[...,0].T, acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
            image.image_series_index = 4
            image.slice = 0
            image.attribute_string = xml
            images.append(image)


    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images
