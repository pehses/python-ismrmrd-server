
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import base64
import ctypes
import multiprocessing

import tempfile
try:
    from bartpy.wrapper import bart
    from bartpy.utils import cfl
except:
    from bart import bart
    from bart import cfl


# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

use_multiprocessing = False
use_gpu = True
os_removal = True
sel_x = None  # option to reduce x dim for faster & low-mem recon
# sel_x = slice(30, 280)
sel_x = slice(20, 134)

def getCoilInfo(coilname='nova_ptx'):
    coilname = coilname.lower()
    if coilname=='nova_ptx':
        fft_scale = np.array([
            1.024957, 0.960428, 0.991236, 1.037026, 1.071855, 1.017678,
            1.02946 , 1.026439, 1.083618, 1.124822, 1.169501, 1.148701,
            1.220159, 1.211465, 1.212671, 1.160536, 1.072906, 1.049849,
            1.046032, 1.018297, 1.024308, 0.975085, 0.977127, 0.975455,
            0.966018, 0.945748, 0.943535, 0.964435, 1.009673, 0.9225  ,
            0.962792, 0.935691])
        rawdata_corrfactors = np.array([
            -7.869929+3.80047j , -7.727324+4.071778j, -7.88741 +3.761298j,
            -7.746147+4.034059j, -7.905413+3.721748j, -7.681937+3.962719j,
            -7.869919+3.756741j, -7.708525+3.898736j, -7.344962+4.680127j,
            -7.219433+4.857659j, -7.362616+4.62574j , -7.207232+4.834069j,
            -7.335363+4.60201j , -7.103662+4.94855j , -7.339441+4.680904j,
            -7.114415+4.918804j, -7.366599+4.685465j, -7.150412+4.849619j,
            -7.338072+4.695826j, -7.179264+4.87732j , -7.334629+4.790239j,
            -7.097607+4.900652j, -7.325254+4.716376j, -7.147962+4.788579j,
            -7.354259+4.671206j, -7.1664  +4.843273j, -7.292011+4.672282j,
            -7.171817+4.863891j, -7.357615+4.663175j, -7.049273+4.926576j,
            -7.300245+4.660961j, -6.767411+4.967862j])
    else:
        raise IndexError("coilname not known")
    return fft_scale, rawdata_corrfactors


def cfftn(data, axes, norm="ortho"):
    """ Centered fast fourier transform, n-dimensional.

    :param data: Complex input data.
    :param axes: Axes along which to shift and transform.
    :return: Fourier transformed data.
    """
    return fft.fftshift(fft.fftn(fft.ifftshift(data, axes=axes), axes=axes, norm=norm), axes=axes)


def cifftn(data, axes, norm="ortho"):
    """ Centered inverse fast fourier transform, n-dimensional.

    :param data: Complex input data.
    :param axes: Axes along which to shift.
    :return: Inverse fourier transformed data.
    """
    return fft.ifftshift(fft.ifftn(fft.fftshift(data, axes=axes), axes=axes, norm=norm), axes=axes)


def adjust_columns(item, optmat, metadata):
    # pre-whitening & os-removal & slicing & resizing

    ncol = item.data.shape[-1]
    nx = metadata.encoding[0].encodedSpace.matrixSize.x 
    
    # operations to perform:
    resize = nx != ncol
    reslice = sel_x is not None
    whiten = optmat is not None

    if not (os_removal or resize or reslice or whiten):
        return  # nothing to do

    data = item.data

    if resize or reslice or os_removal:
        if resize:
            # first pad refscan to full col size (cutting not yet supported)
            dsz = nx - ncol
            data = np.pad(data, ((0, 0), (dsz//2, dsz//2)))
            ncol = data.shape[-1]

        # the next operations need to be performed in image space
        data = cifftn(data, [-1])
        
        if os_removal:
            data = data[:, slice(ncol//4, (ncol*3)//4)]

        if reslice:
            data = data[..., sel_x]
        
        # back to k-space
        data = cfftn(data, [-1])
    
    if whiten:
        data = optmat @ data
    
    # store modified data
    item.resize(number_of_samples=data.shape[-1], active_channels=data.shape[0])
    item.data[:,:] = data


def apply_cc(item, cc_matrix):
    if cc_matrix is None:
        return
    data = cc_matrix @ item.data
    
    # store modified data
    item.resize(number_of_samples=data.shape[-1], active_channels=data.shape[0])
    item.data[:,:] = data


def calibrate_prewhitening(noise, scale_factor=1.0, normalize=True):
    # input noise: [ncha, nsamples]
    # output: [ncha, ncha]
    R = np.cov(noise)   #R = (noise @ np.conj(noise).T)/noise.shape[-1]
    if normalize:
        R /= np.mean(abs(np.diag(R)))
    R[np.diag_indices_from(R)] = abs(R[np.diag_indices_from(R)])
    R = np.linalg.inv(np.linalg.cholesky(R))
    return R.T


def calibrate_scc(data):
    # input data: [ncha, nsamples]
    # output: [ncha, ncha]
    U, s, _ = np.linalg.svd(data, full_matrices=False)
    mtx = np.conj(U.T)
    return mtx, s


def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier

    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

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

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Check for GPU availability
    global use_gpu
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        use_gpu = True
    else:
        use_gpu = False

    # Continuously parse incoming data parsed from MRD messages
    acqGroup = []
    noiseGroup = []
    waveformGroup = []

    # hard-coded limit of 256 slices (better: use Nslice from protocol)
    max_no_of_slices = 256
    acsGroup = [[] for _ in range(max_no_of_slices)]
    sensmaps = [None] * max_no_of_slices

    _, corr_factors = getCoilInfo()
    corr_factors = corr_factors[:, np.newaxis]
    apply_prewhitening = True
    ncc = 16  # number of compressed coils
    optmat = None  # for pre-whitening
    cc_matrix = [None] * max_no_of_slices  # for coil compression
    
    if use_multiprocessing:
        pool = multiprocessing.Pool(processes=4)

    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # skip some data fields
                if item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # skope sync scans
                    continue

                if item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA):
                   item.data[:,:] = item.data[:,:] * corr_factors

                # wip: run noise decorrelation
                if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                    noiseGroup.append(item)
                    continue
                elif apply_prewhitening and len(noiseGroup) > 0 and optmat is None:
                    pass
                    noise_data = []
                    for acq in noiseGroup:
                        noise_data.append(acq.data)
                    noise_data = np.concatenate(noise_data, axis=1)  # [ncha, nsamples]
                    # calculate pre-whitening matrix
                    optmat = calibrate_prewhitening(noise_data)
                    del noise_data

                # resize colums; remove os; apply pre-whitening
                adjust_columns(item, optmat, metadata)

                # Accumulate all imaging readouts in a group
                if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    continue
                elif sensmaps[item.idx.slice] is None:
                    if ncc>0:  # calibrate coil compression
                        data = [acq.data for acq in acsGroup[item.idx.slice]]
                        data = np.asarray(data)
                        nc = data.shape[1]
                        cc_matrix[item.idx.slice], s = calibrate_scc(data.transpose((1, 0, 2)).reshape([nc, -1]))
                        cc_matrix[item.idx.slice] = cc_matrix[item.idx.slice][:ncc, :]
                        
                        # apply coil compression
                        data = cc_matrix[item.idx.slice] @ data

                        # write data back to acsGroup:
                        for acq, dat in zip(acsGroup[item.idx.slice], data):
                            acq.resize(number_of_samples=dat.shape[-1], active_channels=dat.shape[0])
                            acq.data[:,:] = dat
                    
                    # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata)

                # apply coil-compression
                apply_cc(item, cc_matrix[item.idx.slice])
                
                acqGroup.append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.                
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    # logging.info("Processing a group of k-space data")
                    # images = process_raw(acqGroup, config, metadata, sensmaps[item.idx.slice])
                    # logging.debug("Sending images to client:\n%s", images)
                    # connection.send_image(images)
                    if use_multiprocessing:
                        pool.apply_async(process_and_send, (connection, acqGroup, config, metadata, sensmaps[item.idx.slice]))
                    else:
                        process_and_send(connection, acqGroup, config, metadata, sensmaps[item.idx.slice])
                    acqGroup = []

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
        if len(acqGroup) > 0:
            logging.info("Processing a group of k-space data (untriggered)")
            if sensmaps[acqGroup[-1].idx.slice] is None:
                # run parallel imaging calibration
                sensmaps[acqGroup[-1].idx.slice] = process_acs(acsGroup[acqGroup[-1].idx.slice], config, metadata)
            # image = process_raw(acqGroup, config, metadata, sensmaps[acqGroup[-1].idx.slice])
            # connection.send_image(image)
            if use_multiprocessing:
                pool.apply_async(process_and_send, (connection, acqGroup, config, metadata, sensmaps[item.idx.slice]))
            else:
                process_and_send(connection, acqGroup, config, metadata, sensmaps[item.idx.slice])
            acqGroup = []

    finally:
        if use_multiprocessing:
            pool.close()
            pool.join()
        connection.send_close()



# wip: this may in the future help with multiprocessing
def process_and_send(connection, acqGroup, config, metadata, sensmap):
    logging.info("Processing a group of k-space data")
    images = process_raw(acqGroup, config, metadata, sensmap)
    logging.debug("Sending images to client:\n%s", images)
    connection.send_image(images)


# Sorting of k-space data
def sort_into_kspace(group, metadata, reduced_input_matrix=False, tight_output=False):
    # initialize k-space
    # nc = metadata.acquisitionSystemInformation.receiverChannels
    nc = group[0].active_channels
    ncol = group[0].number_of_samples

    nx = ncol  # should already be accounted for by now
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z

    if reduced_input_matrix or tight_output:
        # acs data is acquired on a smaller k-space grid
        enc1_min, enc1_max = int(999), int(0)
        enc2_min, enc2_max = int(999), int(0)
        for cnt, acq in enumerate(group):
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
    
    if tight_output:
        offset_enc1 = 0
        offset_enc2 = 0
        nx = ncol
        ny = enc1_max + 1
        nz = enc2_max + 1
        col = slice(0, ncol)
    elif reduced_input_matrix:
        offset_enc1 = ny // 2 - (enc1_max+1) // 2
        offset_enc2 = nz // 2 - (enc2_max+1) // 2
        ccol = ncol // 2
        col = slice(nx // 2 - ccol, nx // 2 + ccol)
    else:
        col = slice(None)
        offset_enc1 = 0
        offset_enc2 = 0
        # support partial Fourier:
        cenc1 = metadata.encoding[0].encodingLimits.kspace_encoding_step_1.center
        cenc2 = metadata.encoding[0].encodingLimits.kspace_encoding_step_2.center
        if ny//2 > cenc1:
            offset_enc1 = ny//2 - cenc1
        if nz//2 > cenc2:
            offset_enc2 = nz//2 - cenc2

    logging.debug('sort_into_kspace: ncol = %d; nx = %d; ny = %d; nz = %d; nc = %d'%(ncol, nx, ny, nz, nc))

    acq_key = [None] * nz
    kspace = np.zeros([ny, nz, nc, nx], dtype=group[0].data.dtype)
    counter = np.zeros([ny, nz], dtype=np.uint16)

    for key, acq in enumerate(group):
        enc1 = acq.idx.kspace_encode_step_1 + offset_enc1
        enc2 = acq.idx.kspace_encode_step_2 + offset_enc2
        
        kspace[enc1, enc2, :, col] += acq.data
        counter[enc1, enc2] += 1

        # save one header per partition
        if acq_key[enc2] is None:
            acq_key[enc2] = key

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc)
    kspace = np.transpose(kspace, [3, 0, 1, 2])

    return kspace, acq_key


def process_acs(group, config, metadata, n_maps=2, chunksz=16):
    
    gpu_str = "-g" if use_gpu else ""

    if len(group)>0:
        acs, _ = sort_into_kspace(group, metadata, True)
        nx, ny, nz, _ = acs.shape

        logging.debug(f'acs.shape = {acs.shape}')

        # '-I': Intensity correction useful? -> Marten
        # ESPIRiT calibration
        if chunksz>0:
            # test: possibly much faster (system.os bart w. /dev/shm instead of wrapper? )
            # name_in = os.path.join('/dev/shm', tempfile.NamedTemporaryFile().name)
            # name_out = os.path.join('/dev/shm', tempfile.NamedTemporaryFile().name)
            # cfl.writecfl(name_in, acs)
            # ERR = os.system('/scratch/ehsesp/bin/espirit_econ.sh -g -c 8 %s %s'%(name_in, name_out))
            # sensmaps = cfl.readcfl(name_out)
            # return sensmaps

            # espirit_econ: reduce memory footprint by chunking
            eon = bart(1, 'ecalib %s -m %d -1'%(gpu_str, n_maps), acs)

            # fix scaling for bart:
            eon *= np.sqrt(np.prod(acs.shape[:3]))

            eon = cifftn(eon, [2])
            dsz = nz-eon.shape[2]
            eon = np.pad(eon, ((0, 0), (0, 0), (dsz//2, dsz//2), (0, 0)))
            eon = cfftn(eon, [2])

            sensmaps = np.zeros(acs.shape + ((n_maps,) if n_maps>1 else ()), dtype=acs.dtype)
            for i in range(0, nz, chunksz):
                sl = slice(i, i+chunksz)
                sl_len = len(range(*sl.indices(nz)))
                sensmaps[:,:,sl] = bart(1, 'ecaltwo %s -m %d %d %d %d'%(gpu_str, n_maps, nx, ny, sl_len), eon[:,:,sl])
        else:
            sensmaps = bart(1, 'ecalib %s-m %d'%(gpu_str, n_maps), acs)

        np.save(os.path.join(debugFolder, "acs.npy"), acs)
        np.save(os.path.join(debugFolder, "sensmaps.npy"), sensmaps)

        return sensmaps
    else:
        return None


def process_raw(group, config, metadata, sensmaps=None, chunksz=48, max_iter=20): #76

    data, acq_key = sort_into_kspace(group, metadata)

    logging.debug("Raw data is size %s" % (data.shape,))
    # np.save(os.path.join(debugFolder, "raw.npy"), data)
    
    gpu_str = "-g" if use_gpu else ""
    pics_str = f'pics -l1 -r0.003 -S -i {max_iter} -d5 {gpu_str}'
    logging.debug(pics_str)
    logging.debug(f'data.shape = {data.shape}; sensmaps.shape = {sensmaps.shape}')
    if chunksz==0:
        data = bart(1, pics_str, data, sensmaps)
        data = data[[slice(None)] * 3 + (data.ndim-3) * [0]]  # select first 3 dims
    else:
        data = cifftn(data, [0])
        nx = data.shape[0]
        for i in range(0, nx, chunksz):
            sl = slice(i, i+chunksz)
            sl_len = len(range(*sl.indices(nx)))
            block = cfftn(data[sl], [0])
            logging.debug('block.shape = %s; sensmaps[sl].shape = %s'%(block.shape, sensmaps[sl].shape))
            block = bart(1, pics_str, block, sensmaps[sl])
            # store reconstructed data in first coil element
            data[sl,:,:,0] = block[[slice(None)] * 3 + (block.ndim-3) * [0]]  # select first 3 dims
        data = data[:,:,:,0]  # reduce coil dim
    
    data = np.abs(data)


    logging.debug("Image data is size %s" % (data.shape,))
    # np.save(os.path.join(debugFolder, "img.npy"), data)

    # export nifti
    rotmat = np.zeros((4,4))
    rotmat[0,2] = 1
    rotmat[1,1] = 1
    rotmat[2,0] = -1
    rotmat[3,3] = 1
    import nibabel as nib
    # data /= 0.00046
    nii = nib.Nifti1Image(data, rotmat)    
    nii.header['pixdim'][1] = metadata.encoding[0].reconSpace.fieldOfView_mm.x / metadata.encoding[0].reconSpace.matrixSize.x
    nii.header['pixdim'][2] = metadata.encoding[0].reconSpace.fieldOfView_mm.y / metadata.encoding[0].reconSpace.matrixSize.y
    nii.header['pixdim'][3] = metadata.encoding[0].reconSpace.fieldOfView_mm.z / metadata.encoding[0].reconSpace.matrixSize.z
    nib.save(nii, os.path.join(debugFolder, 'img.nii.gz'))

    # Normalize and convert to int16
    # save one scaling in 'static' variable
    try:
        process_raw.imascale
    except:
        process_raw.imascale = 0.8 / data.max()
    data *= 32767 * process_raw.imascale
    data = np.around(data)
    data = data.astype(np.int16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768'})
    xml = meta.serialize()
    
    images = []
    n_par = data.shape[-1]
    logging.debug("data.shape %s" % (data.shape,))

    # Wrong Sequence or Protocol detected, aborting...
    # Number of slices announced via protocol: 1
    # Delivered slice index of current MDH: 1 (allowed: 0..0)
    # Fazit:
    # image.slice auf partition index setzen bringt nichts

    for par in range(n_par):
        # Format as ISMRMRD image data
        image = ismrmrd.Image.from_array(data[...,par], group[acq_key[par]])

        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                               ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                               ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

        if n_par>1:
            image.image_index = 1 + par
            image.image_series_index = 1 + group[acq_key[par]].idx.repetition
            image.user_int[0] = par # is this correct??
            # if vol_pos is None:
            #     vol_pos = image.position[-1]
            # image.position[-1] = vol_pos + (par - n_par//2) * FOVz / rNz # funktioniert, muss aber noch richtig angepasst werden (+vorzeichen check!!!)
        else:
            image.image_index = 1 + group[acq_key[par]].idx.repetition
        
        image.attribute_string = xml
        images.append(image)

    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images
