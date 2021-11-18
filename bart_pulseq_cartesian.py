
""" Pulseq Reco for Cartesian datasets
"""

import ismrmrd
import os
import itertools
import logging
import numpy as np
from cfft import cfftn, cifftn
import base64
import ctypes

from bart import bart
from reco_helper import calculate_prewhitening, apply_prewhitening, remove_os #, fov_shift, calc_rotmat, pcs_to_gcs
from pulseq_prot import insert_hdr, insert_acq, get_ismrmrd_arrays
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import unwrap_phase

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

def process_cartesian(connection, config, metadata, prot_file):

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
            metadata.encoding[0].trajectory.value, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Slice and contrast information
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1

    # for B0 mapping example
    ismrmrd_arr = get_ismrmrd_arrays(prot_file)
    if 'echo_times' in ismrmrd_arr:
        process_raw.phs_imgs = [None] * n_contr

    acqGroup = [[[] for _ in range(n_slc)] for _ in range(n_contr)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * 256
    dmtx = None

    # different contrasts need different scaling
    process_raw.imascale = [None] * 256

    # read protocol acquisitions - faster than doing it one by one
    logging.debug("Reading in protocol acquisitions.")
    acqs = []
    prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
    for n in range(prot.number_of_acquisitions()):
        acqs.append(prot.read_acquisition(n))
    prot.close()

    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                insert_acq(acqs[0], item, metadata, noncartesian=False, return_basetrj=False)
                acqs.pop(0)

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
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], metadata, dmtx, gpu)
                    acsGroup[item.idx.slice].clear()

                acqGroup[item.idx.contrast][item.idx.slice].append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup[item.idx.contrast][item.idx.slice], metadata, ismrmrd_arr, dmtx, sensmaps[item.idx.slice], gpu)
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
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], metadata, dmtx)
                image = process_raw(acqGroup[item.idx.contrast][item.idx.slice], metadata, ismrmrd_arr, dmtx, sensmaps[item.idx.slice])
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


def process_acs(group, metadata, dmtx=None, gpu=False):
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
        data = remove_os(data)

        if gpu:
            sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I', data)  # ESPIRiT calibration
        else:
            sensmaps = bart(1, 'ecalib -m 1 -k 6 -I', data)  # ESPIRiT calibration
        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None


def process_raw(group, metadata, ismrmrd_arr, dmtx=None, sensmaps=None, gpu=False):

    data = sort_into_kspace(group, metadata, dmtx)
    data = remove_os(data)

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # for B0 mapping example
    if 'echo_times' in ismrmrd_arr:
        echo_times = ismrmrd_arr['echo_times']
        te_diff = echo_times[1] - echo_times[0] # [s]
    else:
        te_diff = None

    if sensmaps is None:
        logging.debug("no pics necessary, just do standard FFT")
        data_uncmb = cifftn(data, axes=(0, 1, 2))
        data = np.sqrt(np.sum(np.abs(data_uncmb)**2, axis=-1)) # Sum of squares coil combination

        # B0 mapping example - atm only dual echo
        # use own code for masking & regularizing, though FSL would also be possible (if installed in docker)
        if te_diff is not None:
            contr = group[0].idx.contrast
            n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
            process_raw.phs_imgs[contr] = data_uncmb.copy()
            if contr == n_contr-1:
                img1 = process_raw.phs_imgs[0][:,:,0] # remove 3rd dim - only 2D
                img2 = process_raw.phs_imgs[1][:,:,0]
                phasediff = img1 * np.conj(img2) # phase difference
                phasediff = np.sum(phasediff, axis=-1) # coil combination
                phasediff = unwrap_phase(np.angle(phasediff))
                fmap = phasediff/(2*np.pi*te_diff)

                # simple threshold mask
                mask = data[:,:,0]/np.max(data) 
                mask[mask<0.06] = 0
                mask[mask>=0.06] = 1

                # apply masking and some regularization
                fmap *= mask
                fmap = gaussian_filter(fmap, sigma=0.5)
                fmap = median_filter(fmap, size=2)
                fmap = np.around(fmap)
                fmap = fmap.astype(np.int16)
            else:
                fmap = None
    else:
        if gpu:
            data = bart(1, 'pics -g -S -e -l1 -r 0.001 -i 50', data, sensmaps)
        else:
            data = bart(1, 'pics -S -e -l1 -r 0.001 -i 50', data, sensmaps)
        data = np.abs(data)

    # correct orientation at scanner (consistent with ICE)
    data = np.swapaxes(data, 0, 1)
    data = np.flip(data, (0,1,2))

    logging.debug("Image data is size %s" % (data.shape,))

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
                         'WindowWidth':            '32768',
                         'Keep_image_geometry':    '1'})
    xml = meta.serialize()

    # Format as ISMRMRD image data
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_par = data.shape[-1]
    images = []
    if n_par > 1:
        for par in range(n_par):
            image = ismrmrd.Image.from_array(data[...,par], acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_par + par # contains image index (slices/partitions)
            image.image_series_index = 1 + group[0].idx.repetition # contains image series index, e.g. different contrasts
            image.slice = 0
            image.attribute_string = xml
            image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
            images.append(image)

    else:
        image = ismrmrd.Image.from_array(data[...,0], acquisition=group[0])
        image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice # contains image index (slices/partitions)
        image.image_series_index = 1 + group[0].idx.repetition # contains image series index, e.g. different contrasts
        image.slice = 0
        image.attribute_string = xml
        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        images.append(image)

    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images
