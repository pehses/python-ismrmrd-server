
""" Pulseq Reco for Cartesian datasets
"""

import ismrmrd
import os
import logging
import numpy as np
from cfft import cfftn, cifftn
import ctypes
from datetime import datetime

import importlib

from bart import bart
import reco_helper as rh
from pulseq_helper import insert_hdr, insert_acq, get_ismrmrd_arrays, read_acqs

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

force_pics = False # force parallel imaging reco
parallel_reco = False # parallel reconstruction of slices/contrasts

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
    phs_imgs = [[[] for _ in range(n_slc)] for _ in range(n_contr)]
    ismrmrd_arr = get_ismrmrd_arrays(prot_file)
    fmap_scan = False
    if 'echo_times' in ismrmrd_arr and len(ismrmrd_arr['echo_times']) > 1:
        fmap_scan = True

    acqGroup = [[[] for _ in range(n_slc)] for _ in range(n_contr)]
    noiseGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = None
    dmtx = None

    # Save k-space data for calculating sensitivity maps in following accelerated scans (e.g. EPI/spiral)
    acs_ref = [[] for _ in range(n_slc)]

    # read protocol acquisitions - faster than doing it one by one
    logging.debug("Reading in protocol acquisitions.")
    acqs = read_acqs(prot_file)

    # initilaize encoding info
    process_acs.enc_info = [999, 0, 999, 0]

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
                    set_enc_info(item)
                    continue
                elif sensmaps is None and len(acsGroup[0]) > 0:
                    # run parallel imaging calibration
                    acs_ksp = []
                    for slc in acsGroup:
                        acs_ksp.append(sort_into_kspace(slc, metadata, dmtx, zf_around_center=True))
                    sensmaps = process_acs(acsGroup, gpu)
                    process_acs.enc_info = [999, 0, 999, 0]

                acqGroup[item.idx.contrast][item.idx.slice].append(item)
                set_enc_info(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if (item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION)) and not parallel_reco:
                    logging.info("Processing a group of k-space data")
                    img, img_uncmb, refdata = process_raw(acqGroup[item.idx.contrast][item.idx.slice], metadata, dmtx, sensmaps, gpu)
                    image_list[item.idx.contrast][item.idx.slice] = img
                    phs_imgs[item.idx.contrast][item.idx.slice] = img_uncmb # phase data for field map calculation
                    if item.idx.contrast == 0:
                        acs_ref[item.idx.slice] = refdata
                
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
                    if parallel_reco:
                        logging.debug("Parallel reconstruction of slices/contrasts")
                        imgs, phs_imgs, acs_ref = process_raw(acqGroup, metadata, dmtx, sensmaps, gpu, parallel=True)
                        for k,contr in enumerate(imgs):
                            for j,slc in enumerate(contr):
                                image_list[k][j] = slc

                    images = send_images(image_list, metadata, acqGroup[0][0])
                    connection.send_image(images)
                    if fmap_scan:
                        phs_imgs = np.moveaxis(np.asarray(phs_imgs), 0,-1)  # to [slices,nx,ny,nz,nc,n_contr]                       
                        metadata.userParameters.userParameterString[0].value = os.path.basename(os.path.splitext(prot_file)[0])
                        images = calc_fieldmap(phs_imgs, ismrmrd_arr['echo_times'], metadata, acqGroup[0][0])
                        connection.send_image(images)
                    
                    # save acs data
                    current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
                    acs_name = f"{current_datetime}_acs_{os.path.basename(os.path.splitext(prot_file)[0])}.npy"
                    acs_folder = os.path.join(dependencyFolder, "acs_data")
                    if not os.path.exists(acs_folder):
                        os.makedirs(acs_folder, mode=0o774)
                    np.save(os.path.join(acs_folder, acs_name), np.array(acs_ref)) # save acs reference data
                    # append name to txt file
                    with open(os.path.join(acs_folder, "acs_list.txt"), "a") as f:
                        f.write(acs_name + '\n')

    finally:
        connection.send_close()

def set_enc_info(item):
    enc1 = item.idx.kspace_encode_step_1
    enc2 = item.idx.kspace_encode_step_2
    if enc1 < process_acs.enc_info[0]:
        process_acs.enc_info[0] = enc1
    if enc1 > process_acs.enc_info[1]:
        process_acs.enc_info[1] = enc1
    if enc2 < process_acs.enc_info[2]:
        process_acs.enc_info[2] = enc2
    if enc2 > process_acs.enc_info[3]:
        process_acs.enc_info[3] = enc2

def sort_into_kspace(group, metadata, dmtx=None, zf_around_center=False):

    # initialize k-space
    nc = metadata.acquisitionSystemInformation.receiverChannels
    nx = metadata.encoding[0].reconSpace.matrixSize.x
    ny = metadata.encoding[0].reconSpace.matrixSize.y
    nz = metadata.encoding[0].reconSpace.matrixSize.z

    kspace = np.zeros([ny, nz, nc, nx], dtype=group[0].data.dtype)
    counter = np.zeros([ny, nz], dtype=np.uint16)

    # logging.debug("nx/ny/nz: %s/%s/%s; enc1 min/max: %s/%s; enc2 min/max:%s/%s, ncol: %s" % 
    #               (nx, ny, nz, process_acs.enc_info[0], process_acs.enc_info[1], process_acs.enc_info[2], process_acs.enc_info[3], group[0].data.shape[-1]))

    for acq in group:

        # check reverse flag
        if acq.is_flag_set(ismrmrd.ACQ_IS_REVERSE):
            acq.data[:] = np.flip(acq.data[:], -1)

        # Oversampling removal - WIP: assumes 2x oversampling at the moment
        data = rh.remove_os(acq.data[:], axis=-1)

        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2

        # in case dim sizes smaller than expected, sort data into k-space center (e.g. for reference scans)
        ncol = data.shape[-1]
        cx = nx // 2
        ccol = ncol // 2
        col = slice(cx - ccol, cx + ccol)

        if zf_around_center:
            cy = ny // 2
            cz = nz // 2

            cenc1 = (process_acs.enc_info[1]+1) // 2
            cenc2 = (process_acs.enc_info[3]+1) // 2

            # sort data into center k-space (assuming a symmetric acquisition)
            enc1 += cy - cenc1
            enc2 += cz - cenc2
        
        if dmtx is None:
            kspace[enc1, enc2, :, col] += data
        else:
            kspace[enc1, enc2, :, col] += rh.apply_prewhitening(data, dmtx)
        counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc)
    kspace = np.transpose(kspace, [3, 0, 1, 2])

    return kspace


def process_acs(acs_ksp, gpu=False):
    """
    Process the ACS data to calculate sensitivity maps
    acs_ksp: ACS k-space data, dims: [n_slc, nx, ny, nz, nc] 
    """

    acs_ksp = np.moveaxis(np.asarray(acs_ksp),0,-1) # [n_slc, nx, ny, nz, nc] -> [nx, ny, nz, nc, n_slc]

    # ESPIRiT
    if gpu and acs_ksp.shape[2] > 1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
        sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I', acs_ksp)
    elif acs_ksp.shape[-1] > 1:
        sensmaps = bart(1, f'--parallel-loop {2**(acs_ksp.ndim-1)} -e {acs_ksp.shape[-1]} ecalib -m 1 -k 6 -I', acs_ksp) 
    else: 
        sensmaps = bart(1, 'ecalib -m 1 -k 6 -I', acs_ksp) 

    while sensmaps.ndim < 5:
        sensmaps = sensmaps[...,np.newaxis]

    # back to [n_slc, nx, ny, nz, nc]
    acs_ksp = np.moveaxis(np.asarray(acs_ksp),-1,0)
    sensmaps = np.moveaxis(np.asarray(sensmaps),-1,0)

    return sensmaps

def process_raw(group, metadata, dmtx=None, sensmaps=None, gpu=False, parallel=False):

    if parallel:
        ksp = []
        for contr in group:
            for slc in contr:
                ksp.append(sort_into_kspace(slc, metadata, dmtx, zf_around_center=True))
        ksp = np.asarray(ksp)
    else:
        ksp = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)

    logging.debug("Raw data is size %s" % (ksp.shape,))

    if force_pics and sensmaps is None:
        logging.debug("force pics")
        if parallel:
            n_slc = len(group[0])
            acs_data = ksp[:n_slc]
        else:
            acs_data = ksp[np.newaxis]
        sensmaps = process_acs(acs_data, gpu)
        if parallel:
            n_contr = len(group)
            sensmaps = np.tile(sensmaps, (n_contr, 1, 1, 1, 1))
        else:
            sensmaps = sensmaps[0]
    
    if sensmaps is None:
        logging.debug("no pics necessary, just do standard FFT")
        coildim = -1
        if parallel:
            ksp = np.moveaxis(ksp, 0, -1)
            coildim = -2
        img_uncmb = cifftn(ksp, axes=(0, 1, 2))
    else:
        if parallel:
            ksp = np.moveaxis(ksp, 0, -1) # move slices/contrasts to last dimension
            sensmaps = np.moveaxis(sensmaps, 0, -1)
        elif sensmaps.ndim == 5:
            sensmaps = sensmaps[group[0].idx.slice]

        pics_str = 'pics -S -e -l1 -r 0.001 -i 50'
        if gpu and ksp.shape[2] > 1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
            pics_str += ' -g'
        if parallel:
            pics_str = f'--parallel-loop {2**(ksp.ndim-1)} -e {ksp.shape[-1]} ' + pics_str

        img_uncmb = bart(1, pics_str, ksp, sensmaps)
        while img_uncmb.ndim < 4:
            img_uncmb = img_uncmb[...,np.newaxis] # add nz, nc axis if necessary

    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    rNz = metadata.encoding[0].reconSpace.matrixSize.z
    if nz > 1:
        img_uncmb = rh.fov_shift_img_axis(img_uncmb, 0.5, axis=2)
    if nz > rNz:
        img_uncmb = img_uncmb[:,:,(nz - rNz)//2:-(nz - rNz)//2] # remove oversampling in slice direction

    # coil combination if necessary
    if sensmaps is None:
        img = np.sqrt(np.sum(np.abs(img_uncmb)**2, axis=coildim)) # Sum of squares coil combination
    else:
        img = np.abs(img_uncmb[...,0,:])

    # correct orientation at scanner (consistent with ICE)
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, (0,1,2))

    if parallel:
        n_contr = len(group)
        n_slc = len(group[0])
        newshape_img_uncmb = [n_contr, n_slc] + list(img_uncmb.shape[:4])
        img_uncmb = np.moveaxis(img_uncmb, -1, 0).reshape(newshape_img_uncmb)
        newshape_img = [n_contr, n_slc] + list(img.shape[:3])
        img = np.moveaxis(img, -1, 0).reshape(newshape_img)
        ksp = np.moveaxis(ksp, -1, 0)[:n_slc]

    logging.debug("Image data is size %s" % (img.shape,))

    return img, img_uncmb, ksp

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
                         'Keep_image_geometry':    1,
                         'ImgType':                'Imgdata'})
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
    n_contr = imgs.shape[-1]

    fmap = rh.calc_fmap(imgs, echo_times, metadata)

    # save in dependency - swap x/y axes for correct orientation in PowerGrid
    prot_file = metadata.userParameters.userParameterString[0].value
    current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
    fmap_name = f"{current_datetime}_fmap_{prot_file}.npz"
    fmap_folder = os.path.join(dependencyFolder, "fmaps")
    if not os.path.exists(fmap_folder):
        os.makedirs(fmap_folder, mode=0o774)
    np.savez(os.path.join(fmap_folder, fmap_name), fmap=np.swapaxes(fmap,1,2), name=fmap_name)
    # append name to txt file
    with open(os.path.join(fmap_folder, "fmap_list.txt"), "a") as f:
        f.write(fmap_name + '\n')

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
                    'Keep_image_geometry':    1,
                    'ImgType':                'fmap'})

    # for position offset
    slc_res = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    rotmat = rh.calc_rotmat(group[0])
   
    images = []

    if nz > 1: # send as 3D volume
        image = ismrmrd.Image.from_array(fmap, acquisition=group[0])
        image.image_index = 1 # contains image index
        image.image_series_index = n_contr
        image.slice = 0
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
            image.image_series_index = n_contr
            image.slice = 0
            image.attribute_string = meta.serialize()
            image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
            image.position[:] += pos_offset
            image.image_type = ismrmrd.IMTYPE_REAL
            images.append(image)

    return images
