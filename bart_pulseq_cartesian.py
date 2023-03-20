
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
import reco_helper as rh
from pulseq_helper import insert_hdr, insert_acq, get_ismrmrd_arrays, read_acqs
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import unwrap_phase

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

def process_cartesian(connection, config, metadata, prot_file):

    # Check protocol arrays to process with possible subscript
    prot_arrays = get_ismrmrd_arrays(prot_file)
    if "dream" in prot_arrays:
        import bart_pulseq_cartesian_dream
        import importlib
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

    # different contrasts need different scaling
    process_raw.imascale = [None] * 256

    # read protocol acquisitions - faster than doing it one by one
    logging.debug("Reading in protocol acquisitions.")
    acqs = read_acqs(prot_file)

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
                    image, img_uncmb = process_raw(acqGroup[item.idx.contrast][item.idx.slice], metadata, dmtx, sensmaps[item.idx.slice], gpu)
                    logging.debug("Sending image to client:\n%s", image)
                    connection.send_image(image)
                    if fmap_scan and img_uncmb is not None:
                        phs_imgs[item.idx.contrast][item.idx.slice] = img_uncmb # save data for field map calculation
                    else:
                        fmap_scan = False
                
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT) and fmap_scan:
                    phs_imgs = np.moveaxis(np.asarray(phs_imgs), 0,-1)  # to [slices,nx,ny,nz,nc,n_contr]
                    image = calc_fmap(phs_imgs, ismrmrd_arr['echo_times'], metadata, acqGroup[0][0])
                    connection.send_image(image)

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
                         'Keep_image_geometry':    1})
    xml = meta.serialize()

    # Format as ISMRMRD image data
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_par = data.shape[-1]
    slc_res = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    rotmat = rh.calc_rotmat(group[0])
    offset = [0, 0, -1*slc_res*(group[0].idx.slice-(n_slc-1)/2)] # slice offset in GCS
    pos_offset = rh.gcs_to_pcs(offset, rotmat) # correct image position in PCS
    
    images = []
    if n_par > 1:
        image = ismrmrd.Image.from_array(data, acquisition=group[0])
        image.image_index = 1 # contains image index
        image.image_series_index = 1
        image.slice = 1
        image.attribute_string = xml
        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        images.append(image)

    else:
        image = ismrmrd.Image.from_array(data[...,0], acquisition=group[0])
        image.image_index = (n_contr*n_slc) - (group[0].idx.contrast * n_slc + group[0].idx.slice) # contains image index (slices/partitions)
        image.image_series_index = 1
        image.slice = group[0].idx.slice
        image.attribute_string = xml
        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        image.position[:] += pos_offset
        images.append(image)

    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images, data_uncmb


def calc_fmap(imgs, echo_times, metadata, group):
    """ Calculate field maps from reference images with two different contrasts

        imgs: [slices,nx,ny,nz,nc,n_contr] - atm: n_contr=2 mandatory
        echo_times: list of echo times [s]

        always saves field map in dimensions [slices/nz, nx, ny]
    """
    
    from scipy.ndimage import  median_filter, gaussian_filter, binary_fill_holes, binary_dilation
    from skimage.transform import resize
    from skimage.restoration import unwrap_phase
    from dipy.segment.mask import median_otsu

    mc_fmap = True # calculate multi-coil field maps to remove outliers (Robinson, MRM. 2011) - recommended
    std_filter = True # apply standard deviation filter (only if mc_fmap selected)
    std_fac = 1.1 # factor for standard deviation denoising (see below)
    romeo_fmap = False # use the ROMEO toolbox for field map calculation
    romeo_uw = False # use ROMEO only for unwrapping (slower than unwrapping with skimage)
    filtering = False # apply Gaussian and median filtering (not recommended for mc_fmap)

    logging.info("Starting field map calculation.")

    if len(echo_times) > 2:
        romeo_fmap = True # more than one echo time only possible with ROMEO

    if romeo_fmap or romeo_uw:
        result_path = dependencyFolder+"/romeo_results"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    n_slc = imgs.shape[0]

    # from [slices,nx,ny,nz,coils,echoes] to either [slices,nx,ny,coils,echoes] or [nz,nx,ny,coils,echoes]
    if nz == 1:
        imgs = imgs[:,:,:,0] # 2D field map acquisition
    elif nz > 1:
        imgs = np.moveaxis(imgs[0],2,0) # 3D field map acquisition
    elif nz > 1 and n_slc > 1:
        raise ValueError("Multi-slab is not supported.")

    if romeo_fmap:
        # ROMEO unwrapping and field map calculation (Dymerska, MRM, 2020)
        fmap = rh.romeo_unwrap(imgs, echo_times, result_path, mc_unwrap=False, return_b0=True)
    elif mc_fmap:
        # Multi-coil field map calculation (Robinson, MRM, 2011)
        phasediff = imgs[...,1] * np.conj(imgs[...,0])
        if romeo_uw:
            phasediff_uw = rh.romeo_unwrap(phasediff,[], result_path, mc_unwrap=True, return_b0=False)
        else:
            phasediff_uw = np.zeros_like(phasediff,dtype=np.float64)
            for k in range(phasediff.shape[-1]):
                phasediff_uw[...,k] = unwrap_phase(np.angle(phasediff[...,k])) # unwrap phase for each coil

        nc = phasediff_uw.shape[-1]
        phasediff_uw_rs = phasediff_uw.reshape([-1,nc])
        img_mag = abs(imgs[...,0]).reshape([-1,nc])
        ix = np.argsort(phasediff_uw_rs, axis=-1)[:,nc//4:-nc//4] # remove lowest & highest quartile
        weights = np.take_along_axis(img_mag, ix, axis=-1) / np.sum(np.take_along_axis(img_mag, ix, axis=-1), axis=-1)[:,np.newaxis]
        fmap = np.sum(weights * np.take_along_axis(phasediff_uw_rs, ix, axis=-1), axis=-1)
        fmap = fmap.reshape(imgs.shape[:3])
        te_diff = echo_times[1] - echo_times[0]
        fmap = -1 * fmap/te_diff # the sign in Powergrid is inverted
    else:
        # Standard field mapping approach (Hermitian product & SOS coil combination)
        phasediff = imgs[...,1] * np.conj(imgs[...,0]) 
        phasediff = np.sum(phasediff, axis=-1) # coil combination
        if romeo_uw:
            phasediff_uw = rh.romeo_unwrap(phasediff, [], result_path, mc_unwrap=False, return_b0=False)
        else:
            phasediff_uw = unwrap_phase(np.angle(phasediff))
        te_diff = echo_times[1] - echo_times[0]
        fmap = -1 * phasediff_uw/te_diff # the sign in Powergrid is inverted

    # mask with threshold and median otsu from dipy
    # do it in 2D as it works better and hole filling is easier
    img_mask = rh.rss(imgs[...,0], axis=-1)
    mask = np.zeros_like(img_mask)
    for k,img in enumerate(img_mask):
        _, mask_otsu = median_otsu(img, median_radius=1, numpass=20)

        # simple threshold mask
        thresh = 0.13
        mask_thresh = img/np.max(img)
        mask_thresh[mask_thresh<thresh] = 0
        mask_thresh[mask_thresh>=thresh] = 1

        # combine masks
        mask[k] = mask_thresh + mask_otsu
        mask[k][mask[k]>0] = 1
        mask[k] = binary_fill_holes(mask[k])
        mask[k] = binary_dilation(mask[k], iterations=2) # some extrapolation

    # Standard deviation filter (Robinson, MRM, 2011)
    if std_filter and mc_fmap:
        phasediff_uw *= mask[...,np.newaxis]
        std = phasediff_uw.std(axis=-1)
        median_std = np.median(std[std!=0])
        idx = np.argwhere(std>std_fac*median_std)
        n_cb = [2,2,1]
        for ix in idx:
            voxel_cb = tuple([slice(max(0,ix[k]-n_cb[k]), ix[k]+n_cb[k]+1) for k in range(3)])
            fmap[tuple(ix)] = np.median((fmap[voxel_cb])[np.nonzero(fmap[voxel_cb])])

    # Gauss/median filter
    if filtering:
        fmap *= mask
        fmap = gaussian_filter(fmap, sigma=0.5)
        fmap = median_filter(fmap, size=2)

    # interpolate to correct matrix size
    if nz == 1:
        newshape = [n_slc,ny,nx]
    else:
        newshape = [nz,ny,nx]
    fmap = resize(np.transpose(fmap,[0,2,1]), newshape, anti_aliasing=True)
    mask = resize(np.transpose(mask,[0,2,1]), newshape, anti_aliasing=False)
    mask[mask>0] = 1 # fix interpolation artifacts in binary mask

    fmap *= mask

    # save in dependency - swap x/y axes for correct orientation in PowerGrid
    np.savez(dependencyFolder+"/fmap.npz", fmap=np.swapaxes(fmap,1,2), mask=mask, name='Field map from external scan.')

    # correct orientation at scanner (consistent with ICE)
    fmap = np.transpose(fmap, [1,2,0])
    fmap = np.flip(fmap, (0,1,2))

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
        for ix, _ in enumerate(fmap): # send as 2D volume
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
