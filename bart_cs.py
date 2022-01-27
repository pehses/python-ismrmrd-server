
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import base64
import ctypes
import multiprocessing

from cfft import cfftn, cifftn

from reco_helper import calculate_prewhitening, remove_os

try:
    from bartpy.wrapper import bart
except:
    from bart import bart


# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

use_multiprocessing = False


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


# WIP: these all need to be tested
def scc_calibrate_mtx(data):
    # input data: [nsamples, ncoils]
    # output: [ncoils, ncoils]
    nc = data.shape[1]
    data = np.moveaxis(data, 1, 0)
    data = data.flatten().reshape((nc, -1))
    U, s, _ = np.linalg.svd(data, full_matrices=False)
    mtx = np.conj(U.T)
    return mtx, s

def apply_mtx(data, inv_mtx):
    return inv_mtx @ data


# Folder for debug output files
debugFolder = "/tmp/share/debug"

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
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        use_gpu = True
    else:
        use_gpu = False

    # Continuously parse incoming data parsed from MRD messages
    acqGroup = []
    noiseGroup = []
    waveformGroup = []

    # hard-coded limit of 256 slices (better: use Nslice from protocol)
    acsGroup = [[] for _ in range(256)]
    sensmaps = [None] * 256

#    _, corr_factors = getCoilInfo()
#    corr_factors = corr_factors[:, np.newaxis]
    ncc = 16  # number of compressed coils
    dmtx = None

    if use_multiprocessing:
        pool = multiprocessing.Pool(processes=4)

    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
#                if item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA):
#                    item.data *= corr_factors

                # wip: run noise decorrelation
                if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                    noiseGroup.append(item)
                    continue
                elif len(noiseGroup) > 0 and dmtx is None:
                    noise_data = []
                    for acq in noiseGroup:
                        noise_data.append(acq.data)
                    noise_data = np.concatenate(noise_data, axis=1)  # [ncha, nsamples]
                    # calculate pre-whitening matrix
                    dmtx = calculate_prewhitening(noise_data)

                # Accumulate all imaging readouts in a group
                if item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # skope sync scans
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    continue
                elif sensmaps[item.idx.slice] is None:
                    if ncc>0:
                        # calibrate coil compression
                        caldata = list()

                        for item2 in acsGroup[item.idx.slice]:
                            caldata.append(apply_mtx(item2.data, dmtx))
                        caldata = np.moveaxis(np.asarray(caldata), 1, 0)
                        caldata = caldata.reshape([caldata.shape[0], -1]).T  # final order: [nsamples, ncoils]

                        M, s = scc_calibrate_mtx(caldata)
                        dmtx = (np.conj(dmtx) @ M).T  # combine pre-whitening and coil-compression
                        dmtx = dmtx[:ncc, :]

                    # dmtx = None  # test

                    # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx, use_gpu)


                acqGroup.append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.                
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    # logging.info("Processing a group of k-space data")
                    # images = process_raw(acqGroup, config, metadata, dmtx, sensmaps[item.idx.slice], use_gpu)
                    # logging.debug("Sending images to client:\n%s", images)
                    # connection.send_image(images)
                    if use_multiprocessing:
                        pool.apply_async(process_and_send, (connection, acqGroup, config, metadata, dmtx, sensmaps[item.idx.slice], use_gpu))
                    else:
                        process_and_send(connection, acqGroup, config, metadata, dmtx, sensmaps[item.idx.slice], use_gpu)
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
                sensmaps[acqGroup[-1].idx.slice] = process_acs(acsGroup[acqGroup[-1].idx.slice], config, metadata, dmtx, use_gpu)
            # image = process_raw(acqGroup, config, metadata, dmtx, sensmaps[acqGroup[-1].idx.slice], use_gpu)
            # connection.send_image(image)
            if use_multiprocessing:
                pool.apply_async(process_and_send, (connection, acqGroup, config, metadata, dmtx, sensmaps[item.idx.slice], use_gpu))
            else:
                process_and_send(connection, acqGroup, config, metadata, dmtx, sensmaps[item.idx.slice], use_gpu)
            acqGroup = []

    finally:
        if use_multiprocessing:
            pool.close()
            pool.join()
        connection.send_close()



# wip: this may in the future help with multiprocessing
def process_and_send(connection, acqGroup, config, metadata, dmtx, sensmap, use_gpu):
    logging.info("Processing a group of k-space data")
    images = process_raw(acqGroup, config, metadata, dmtx, sensmap, use_gpu)
    logging.debug("Sending images to client:\n%s", images)
    connection.send_image(images)


#######
# Sorting of k-space data
#######

def sort_into_kspace(group, metadata, dmtx=None, zf_around_center=True, os_removal=True):
    # initialize k-space
    nc = metadata.acquisitionSystemInformation.receiverChannels
    
    ncc = nc
    if dmtx is not None:
        ncc = dmtx.shape[0]

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

    if zf_around_center:
        nx = metadata.encoding[0].encodedSpace.matrixSize.x
        ny = metadata.encoding[0].encodedSpace.matrixSize.y
        nz = metadata.encoding[0].encodedSpace.matrixSize.z
    else:
        nx = group[0].data.shape[-1]
        ny = enc1_max+1
        nz = enc2_max+1
    if os_removal:
        nx //= 2

    acq_key = [None] * nz
    kspace = np.zeros([ny, nz, ncc, nx], dtype=group[0].data.dtype)
    counter = np.zeros([ny, nz], dtype=np.uint16)

    logging.debug("nx/ny/nz: %s/%s/%s; enc1 min/max: %s/%s; enc2 min/max:%s/%s, ncol: %s" % (nx, ny, nz, enc1_min, enc1_max, enc2_min, enc2_max, group[0].data.shape[-1]))

    for key, acq in enumerate(group):
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        
        # save one header per partition
        if acq_key[enc2] is None:
            acq_key[enc2] = key

        col = slice(None)
        if zf_around_center:
            ncol = acq.data.shape[-1]
            if os_removal:
                ncol //= 2
            cx = nx // 2
            ccol = ncol // 2
            col = slice(cx - ccol, cx + ccol)

            cy = ny // 2
            cz = nz // 2

            cenc1 = (enc1_max+1) // 2
            cenc2 = (enc2_max+1) // 2

            # sort data into center k-space (assuming a symmetric acquisition)
            enc1 += cy - cenc1
            enc2 += cz - cenc2
        
        data = acq.data
        if os_removal:
            data = remove_os(data, -1)

        if dmtx is None:
            kspace[enc1, enc2, :, col] += data
        else:
            kspace[enc1, enc2, :, col] += apply_mtx(data, dmtx)
        counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc)
    kspace = np.transpose(kspace, [3, 0, 1, 2])

    return kspace, acq_key


def process_acs(group, config, metadata, dmtx=None, use_gpu=False, chunksz=8): # chunksz 8?

    if len(group)>0:
        acs, _ = sort_into_kspace(group, metadata, dmtx)
        nx, ny, nz, nc = acs.shape

        # '-I': Intensity correction useful? -> Marten
        # ESPIRiT calibration
        if chunksz>0:
            # espirit_econ: reduce memory footprint by chunking
            if use_gpu:
                eon = bart(1, 'ecalib -g -m 2 -1', acs)
            else:
                eon = bart(1, 'ecalib -m -2 -1', acs)
            
            if nz%chunksz:
                chunksz_ = chunksz
                while nz%chunksz:
                    chunksz //= 2
                logging.debug('chunksz=%d is not a multiple of nz=%d; New chunksz=%d'%(chunksz_, nz, chunksz))
            
            # zero-pad
            tmp = bart(1, 'fft 4', eon)
            tmp2 = bart(1, 'resize -c 2 %d'%(nz,), tmp)
            eon = bart(1, 'fft -i 4', tmp2)
            del tmp, tmp2

            sensmaps = np.zeros(acs.shape + (2,), dtype=acs.dtype)
            for i in range(0, nz, chunksz):
                sl = eon[:,:,i:i+chunksz,:]
                if use_gpu:
                    sensmaps[:,:,i:i+chunksz,:,:] = bart(1, 'ecaltwo -g -m 2 %d %d %d'%(nx, ny, chunksz), sl)
                else:
                    sensmaps[:,:,i:i+chunksz,:,:] = bart(1, 'ecaltwo -m 2 %d %d %d'%(nx, ny, chunksz), sl)
        else:
            if use_gpu:
                sensmaps = bart(1, 'ecalib -g -m 2', acs)
            else:
                sensmaps = bart(1, 'ecalib -m 2', acs)
        np.save(debugFolder + "/" + "acs.npy", acs)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None


def process_raw(group, config, metadata, dmtx=None, sensmaps=None, use_gpu=False):

    data, acq_key = sort_into_kspace(group, metadata, dmtx)

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    if use_gpu:
        data = bart(1, 'pics -l1 -r0.003 -S -d5 -g', data, sensmaps)
    else:
        data = bart(1, 'pics -l1 -r0.003 -S -d5', data, sensmaps)
    data = np.abs(data)
    
    if data.ndim==5: # reconstruction with two e-spirit maps
        data = data[:,:,:,0,0]

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

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
