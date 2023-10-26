
""" Pulseq Reco for Cartesian datasets
"""

import ismrmrd
import os
import logging
import numpy as np
from cfft import cfftn, cifftn
import ctypes

import importlib

from bart import bart
import reco_helper as rh
from pulseq_helper import insert_hdr, insert_acq, get_ismrmrd_arrays, read_acqs

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

def process_cartesian(connection, config, metadata, prot_file):

    # reload reco helper
    importlib.reload(rh)

    # Check protocol arrays to process with possible subscript
    prot_arrays = get_ismrmrd_arrays(prot_file)
    if "dream" in prot_arrays:
        import bart_pulseq_cartesian_dream
        importlib.reload(bart_pulseq_cartesian_dream)
        bart_pulseq_cartesian_dream.process_cartesian_dream(connection, config, metadata, prot_file)
        return

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

    # for B0 mapping
    ismrmrd_arr = get_ismrmrd_arrays(prot_file)
    fmap_scan = False
    if 'echo_times' in ismrmrd_arr:
        phs_imgs = [[[] for _ in range(n_slc)] for _ in range(n_contr)]
        fmap_scan = True

    acqGroup = [[[] for _ in range(n_slc)] for _ in range(n_contr)]
    noiseGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * 256
    dmtx = None

    # read protocol acquisitions - faster than doing it one by one
    logging.debug("Reading in protocol acquisitions.")
    acqs = read_acqs(prot_file)

    image_list = [[[] for _ in range(n_slc)] for _ in range(n_contr)]
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
                    dmtx = rh.calculate_prewhitening(noise_data)
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
                    img, img_uncmb = process_raw(acqGroup[item.idx.contrast][item.idx.slice], metadata, dmtx, sensmaps[item.idx.slice], gpu)
                    image_list[item.idx.contrast][item.idx.slice] = img
                    if fmap_scan and img_uncmb is not None:
                        phs_imgs[item.idx.contrast][item.idx.slice] = img_uncmb # save data for field map calculation
                    else:
                        fmap_scan = False
                
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
                    images = send_images(image_list, metadata, acqGroup[0][0])
                    connection.send_image(images)
                    if fmap_scan:
                        phs_imgs = np.moveaxis(np.asarray(phs_imgs), 0,-1)  # to [slices,nx,ny,nz,nc,n_contr]
                        images = calc_fieldmap(phs_imgs, ismrmrd_arr['echo_times'], metadata, acqGroup[0][0])
                        connection.send_image(images)

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
                image, _ = process_raw(acqGroup[item.idx.contrast][item.idx.slice], metadata, dmtx, sensmaps[item.idx.slice])
                logging.debug("Sending image to client:\n%s", image)
                connection.send_image(image)
                acqGroup = []

    finally:
        connection.send_close()


def sort_into_kspace(group, metadata, dmtx=None, zf_around_center=False):
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
        data = rh.remove_os(acq.data[:], axis=-1)
        acq.resize(number_of_samples=data.shape[-1], active_channels=data.shape[0])
        acq.data[:] = data

    nc = metadata.acquisitionSystemInformation.receiverChannels
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
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
            kspace[enc1, enc2, :, col] += rh.apply_prewhitening(acq.data, dmtx)
        counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc)
    kspace = np.transpose(kspace, [3, 0, 1, 2])

    return kspace


def process_acs(group, metadata, dmtx=None, gpu=False):
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)

        # ESPIRiT
        if gpu and data.shape[2] > 1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
            sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I', data)
        else:
            sensmaps = bart(1, 'ecalib -m 1 -k 6 -I', data) 
        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None


def process_raw(group, metadata, dmtx=None, sensmaps=None, gpu=False):

    data = sort_into_kspace(group, metadata, dmtx)

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    if sensmaps is None:
        logging.debug("no pics necessary, just do standard FFT")
        data_uncmb = cifftn(data, axes=(0, 1, 2))
        data = np.sqrt(np.sum(np.abs(data_uncmb)**2, axis=-1)) # Sum of squares coil combination
    else:
        data_uncmb = None
        if gpu and data.shape[2] > 1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
            data = bart(1, 'pics -g -S -e -l1 -r 0.001 -i 50', data, sensmaps)
        else:
            data = bart(1, 'pics -S -e -l1 -r 0.001 -i 50', data, sensmaps)
        data = np.abs(data)

    # correct orientation at scanner (consistent with ICE)
    data = np.swapaxes(data, 0, 1)
    data = np.flip(data, (0,1,2))

    logging.debug("Image data is size %s" % (data.shape,))

    return data, data_uncmb

def send_images(imgs, metadata, group):

    # Normalize and convert to int16
    # save one scaling in 'static' variable
    scale = np.max(imgs)
    for k,contr in enumerate(imgs):
        for j,slc in enumerate(contr):
            imgs[k][j] *= 32767 / scale
            imgs[k][j] = np.around(imgs[k][j])
            imgs[k][j] = imgs[k][j].astype(np.int16)

    # flip slice dimension if necessary
    imgs = np.asarray(imgs)
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    if n_slc > 1:
        imgs = np.flip(imgs,1)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768',
                         'Keep_image_geometry':    1})
    xml = meta.serialize()

    # Format as ISMRMRD image data
    n_par = imgs[0][0].shape[-1]
    slc_res = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    rotmat = rh.calc_rotmat(group[0])
    
    images = []
    for k,contr in enumerate(imgs):
        for j,data in enumerate(contr):
            if n_par > 1:
                image = ismrmrd.Image.from_array(data, acquisition=group[0])
                image.image_index = j
                image.image_series_index = k
                image.attribute_string = xml
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
                images.append(image)

            else:
                offset = [0, 0, -1*slc_res*(j-(n_slc-1)/2)] # slice offset in GCS
                pos_offset = rh.gcs_to_pcs(offset, rotmat) # correct image position in PCS
                image = ismrmrd.Image.from_array(data[...,0], acquisition=group[0])
                image.image_index = j # slice
                image.image_series_index = k # contrast
                image.attribute_string = xml
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                        ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                        ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
                image.position[:] += pos_offset
                images.append(image)

    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images

def calc_fieldmap(imgs, echo_times, metadata, group):
    """ Calculate field maps from reference images with two different contrasts

        imgs: [slices,nx,ny,nz,nc,n_contr] - atm: n_contr=2 mandatory
        echo_times: list of echo times [s]

    """
    
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    n_slc = imgs.shape[0]

    fmap, mask = rh.calc_fmap(imgs, echo_times, metadata, dep_folder=dependencyFolder)

    # save in dependency - swap x/y axes for correct orientation in PowerGrid
    np.savez(dependencyFolder+"/fmap.npz", fmap=np.swapaxes(fmap,1,2), mask=mask, name='Field map from external scan.')

    # correct orientation at scanner (consistent with ICE)
    fmap = np.transpose(fmap, [1,2,0])
    fmap = np.flip(fmap, (0,1,2))

    # send in Hz
    fmap /= (2*np.pi)

    # Convert to int16
    fmap = np.around(fmap)
    fmap = fmap.astype(np.int16)

    # send images
    meta = ismrmrd.Meta({'DataRole':           'Quantitative',
                    'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                    'WindowCenter':           '0',
                    'WindowWidth':            '512',
                    'Keep_image_geometry':    1})

    # for position offset
    slc_res = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    rotmat = rh.calc_rotmat(group[0])
   
    images = []

    if nz > 1: # send as 3D volume
        image = ismrmrd.Image.from_array(fmap, acquisition=group[0])
        image.image_index = 1 # contains image index
        image.image_series_index = 1
        image.slice = 1
        image.attribute_string = meta.serialize()
        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        images.append(image)

    else:
        for ix in range(fmap.shape[-1]): # send as 2D volume
            offset = [0, 0, -1*slc_res*(ix-(n_slc-1)/2)] # slice offset in GCS
            pos_offset = rh.gcs_to_pcs(offset, rotmat) # correct image position in PCS

            image = ismrmrd.Image.from_array(fmap[...,ix], acquisition=group[0])
            image.image_index = ix
            image.image_series_index = 2
            image.slice = 0
            image.attribute_string = meta.serialize()
            image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
            image.position[:] += pos_offset
            image.image_type = ismrmrd.IMTYPE_REAL
            images.append(image)

    return images
