import ismrmrd
import os
import sys
import itertools
import logging
from cfft import cfft, cifft, fft  # before numpy import to avoid MKL bug!
import numpy as np
import multiprocessing
import tempfile
import ctypes
import mrdhelper
import constants
from multiprocessing import Pool
from functools import partial
from time import perf_counter

from bart import bart


# Folder for sharing data/debugging
# tempfile.tempdir = "/tmp"  # benchmark 1: 148.6 s
tempfile.tempdir = "/dev/shm"  # slightly faster bart wrapper; benchmark 1: 131.1 s

shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")
use_multiprocessing = False  # not implemented yet (bart & numpy use multiprocessing anyway)

# sane defaults:
use_gpu = True
os_removal = True
reduce_fov_x = False
zf_to_orig_sz = True
apply_prewhitening = True
ncc = 16      # number of compressed coils
n_maps = 1    # set to 2 in case of fold-over / too tight FoV
save_unsigned = True  # not sure whether FIRE supports it (or how)
filter_type = None

# override defaults:
# reduce_fov_x = True
# zf_to_orig_sz = False
ncc = 32  # we have time...
sel_x = None
# filter_type = 'long_component'
# filter_type = 'biexponential'


def export_nifti(data, metadata, filename):
    import nibabel as nib

    rotmat = np.zeros((4,4))
    rotmat[0,2] = 1
    rotmat[1,1] = 1
    rotmat[2,0] = -1
    rotmat[3,3] = 1

    nii = nib.Nifti1Image(data, rotmat)
    nii.header['pixdim'][1] = metadata.encoding[0].reconSpace.fieldOfView_mm.x / metadata.encoding[0].reconSpace.matrixSize.x
    nii.header['pixdim'][2] = metadata.encoding[0].reconSpace.fieldOfView_mm.y / metadata.encoding[0].reconSpace.matrixSize.y
    nii.header['pixdim'][3] = metadata.encoding[0].reconSpace.fieldOfView_mm.z / metadata.encoding[0].reconSpace.matrixSize.z
    nib.save(nii, filename)


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


def apply_column_ops(item, optmat, metadata):
    # pre-whitening & os-removal & slicing & resizing

    ncol = item.data.shape[-1]
    nx = metadata.encoding[0].encodedSpace.matrixSize.x

    # operations to perform:
    resize = nx != ncol
    whiten = optmat is not None

    data = item.data

    if resize or reduce_fov_x or os_removal:
        if resize:
            # first pad refscan to full col size (cutting not yet supported)
            dsz = nx - ncol
            data = np.pad(data, ((0, 0), (dsz//2, dsz//2)))
            ncol = data.shape[-1]

        # the next operations need to be performed in image space
        data = cifft(data, -1)

        if os_removal:
            data = data[:, slice(ncol//4, (ncol*3)//4)]

        if reduce_fov_x:
            data = data[..., sel_x]

        data = cfft(data, -1)

    if whiten:
        data = optmat @ data
    else:
        data *= 2e5  # scale it up

    # store modified data
    item.resize(number_of_samples=data.shape[-1], active_channels=data.shape[0])
    item.data[:] = data


def apply_cc(item, cc_matrix):
    if cc_matrix is None:
        return

    data = cc_matrix @ item.data

    # store modified data
    item.resize(number_of_samples=data.shape[-1], active_channels=data.shape[0])
    item.data[:] = data


def calibrate_prewhitening(noise, scale_factor=1.):
    '''Calculates the noise prewhitening matrix

    :param noise: Input noise data (2D array), ``[coil, nsamples]``
    :scale_factor: Applied on the noise covariance matrix. Used to
                   adjust for effective noise bandwith and difference in
                   sampling rate between noise calibration and actual measurement:
                   scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio

    :returns w: Prewhitening matrix, ``[coil, coil]``, w @ data is prewhitened
    '''
    dmtx = (noise @ np.conj(noise).T)/noise.shape[1]
    dmtx = np.linalg.inv(np.linalg.cholesky(dmtx))
    dmtx *= np.sqrt(2)*np.sqrt(scale_factor)
    return dmtx


def calibrate_scc(data):
    # input data: [ncha, nsamples]
    # output: [ncha, ncha]
    U, s, _ = np.linalg.svd(data, full_matrices=False)
    mtx = np.conj(U.T)
    return mtx, s


def calibrate_cc(items):
    if ncc == 0 or ncc >= items[0].data.shape[0]:
        return None  # nothing to do

    data = np.asarray([acq.data for acq in items])
    nc = data.shape[1]
    cc_matrix, s = calibrate_scc(np.moveaxis(data, 1, 0).reshape([nc, -1]))
    cc_matrix = cc_matrix[:ncc, :]

    # apply coil compression
    data = cc_matrix @ data

    # write data back to acsGroup:
    for acq, dat in zip(items, data):
        acq.resize(number_of_samples=dat.shape[-1], active_channels=dat.shape[0])
        acq.data[:] = dat

    return cc_matrix


def apply_filter(item, kcenter_seg=73):

    if filter_type is None:
        return

    def decay(t, tau, amp):
        return amp * np.exp(-t/tau)

    def bidecay(t, tau1, tau2, amp1, amp2):
        return amp1 * np.exp(-t/tau1) + amp2 * np.exp(-t/tau2)
    
    # these were determind from refvolt=220V data (not quite right)
    tau = [100.07897875, 8.54900408]  # unit: 'refocussing pulses'
    amp = [2.14106668, 4.3746221]

    segment = item.idx.segment
    if filter_type is 'long_component':
        # make sure that scale is one in k-space center
        a = 1/decay(kcenter_seg, tau[0], 1)
        scale = 1 / decay(segment, tau[0], a)
    else:
        # make sure that scale is one in k-space center
        a = bidecay(kcenter_seg, *tau, *amp)
        amp[0] = amp[0]/a
        amp[1] = amp[1]/a
        scale = 1/ bidecay(segment, *tau, *amp)

    # logging.debug(f'lin={item.idx.kspace_encode_step_1}, par={item.idx.kspace_encode_step_2}, seg={segment}, scale={scale}')
    item.data[:] = scale * item.data[:]

    return



def process(connection, config, metadata):

    logging.info("Config: \n%s", config)

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

    # # Check for GPU availability
    # global use_gpu
    # if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
    #     use_gpu = True
    # else:
    #     use_gpu = False

    # Continuously parse incoming data parsed from MRD messages
    acqGroup = []
    noiseGroup = []
    waveformGroup = []

    global sel_x
    if reduce_fov_x and sel_x is None:
        nx = metadata.encoding[0].encodedSpace.matrixSize.x//2
        sel_x = slice((nx*3)//32, (nx*19)//32)  # half of matrix

    # hard-coded limit of 256 slices (better: use Nslice from protocol)
    max_no_of_slices = 256
    acsGroup = [[] for _ in range(max_no_of_slices)]
    sensmaps = [None] * max_no_of_slices

    _, corr_factors = getCoilInfo()
    corr_factors = corr_factors[:, np.newaxis]
    optmat = None  # for pre-whitening
    cc_matrix = [None] * max_no_of_slices  # for coil compression

    if use_multiprocessing:
        pool = multiprocessing.Pool(processes=4)

    # Start timer
    tic = perf_counter()

    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # skip some data fields
                if item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_RTFEEDBACK_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_HPFEEDBACK_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):  # deactivate pc for now
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):  # skope sync scans
                    continue
                # elif item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION):  # not in python-ismrmrd yet (todo)
                elif item.is_flag_set(31):
                    continue
                # elif item.is_flag_set(ismrmrd.ACQ_IS_PHASE_STABILIZATION_REFERENCE):  # not in python-ismrmrd yet (todo)
                elif item.is_flag_set(30):
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

                # resize & reslice colums; remove os; apply pre-whitening
                apply_column_ops(item, optmat, metadata)

                # Accumulate all imaging readouts in a group
                if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    continue
                elif sensmaps[item.idx.slice] is None:
                    cc_matrix[item.idx.slice] = calibrate_cc(acsGroup[item.idx.slice])

                    # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata)

                # apply coil-compression
                apply_cc(item, cc_matrix[item.idx.slice])

                apply_filter(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    push_to_kspace(item, metadata, finalize=True)
                    # logging.info("Processing a group of k-space data")
                    # images = process_raw(acqGroup, config, metadata, sensmaps[item.idx.slice])
                    # logging.debug("Sending images to client:\n%s", images)
                    # connection.send_image(images)
                    if use_multiprocessing:
                        pool.apply_async(process_and_send, (connection, push_to_kspace.kspace, push_to_kspace.rawHead, config, metadata, sensmaps[item.idx.slice]))
                    else:
                        process_and_send(connection, push_to_kspace.kspace, push_to_kspace.rawHead, config, metadata, sensmaps[item.idx.slice])
                    acqGroup = []

                    # Measure processing time
                    toc = perf_counter()
                    strProcessTime = "Total processing time: %.2f s" % (toc-tic)
                    logging.info(strProcessTime)

                    # Send this as a text message back to the client
                    connection.send_logging(constants.MRD_LOGGING_INFO, strProcessTime)

                    tic = perf_counter()
                    del push_to_kspace.kspace
                else:
                    push_to_kspace(item, metadata)

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
            push_to_kspace(finalize=True)
            logging.info("Processing a group of k-space data (untriggered)")
            if sensmaps[acqGroup[-1].idx.slice] is None:
                # run parallel imaging calibration
                sensmaps[acqGroup[-1].idx.slice] = process_acs(acsGroup[acqGroup[-1].idx.slice], config, metadata)
            if use_multiprocessing:
                pool.apply_async(process_and_send, (connection, push_to_kspace.kspace, push_to_kspace.rawHead, config, metadata, sensmaps[item.idx.slice]))
            else:
                process_and_send(connection, push_to_kspace.kspace, push_to_kspace.rawHead, config, metadata, sensmaps[item.idx.slice])
            acqGroup = []
            del push_to_kspace.kspace

    finally:
        if use_multiprocessing:
            pool.close()
            pool.join()
        connection.send_close()


# wip: this may in the future help with multiprocessing
def process_and_send(connection, kspace, rawHead, config, metadata, sensmap):
    logging.info("Processing a group of k-space data")
    images = process_raw(kspace, rawHead, connection, config, metadata, sensmap)
    logging.debug("Sending images to client:\n%s", images)
    connection.send_image(images)


# Sorting of k-space data
def push_to_kspace(acq=None, metadata=None, finalize=False):

    if acq is not None:
        try:
            push_to_kspace.kspace
        except:
            # initialize k-space
            nc = acq.active_channels
            nx = acq.number_of_samples
            ny = metadata.encoding[0].encodedSpace.matrixSize.y
            nz = metadata.encoding[0].encodedSpace.matrixSize.z
            push_to_kspace.cenc1 = metadata.encoding[0].encodingLimits.kspace_encoding_step_1.center
            push_to_kspace.cenc2 = metadata.encoding[0].encodingLimits.kspace_encoding_step_2.center
            push_to_kspace.kspace = np.zeros([ny, nz, nc, nx], dtype=acq.data.dtype)
            push_to_kspace.counter = np.zeros([ny, nz], dtype=np.uint16)
            push_to_kspace.rawHead = None
            push_to_kspace.finalized = False
            logging.debug('push_to_kspace: nx = %d; ny = %d; nz = %d; nc = %d'%(nx, ny, nz, nc))

        ## now sort acq into k-space ##
        offset_enc1 = push_to_kspace.cenc1 - acq.idx.user[5]  # center line is encoded in user[5]
        offset_enc2 = push_to_kspace.cenc2 - acq.idx.user[6]  # center partition is encoded in user[6]

        enc1 = acq.idx.kspace_encode_step_1 + offset_enc1
        enc2 = acq.idx.kspace_encode_step_2 + offset_enc2

        push_to_kspace.kspace[enc1, enc2] += acq.data
        push_to_kspace.counter[enc1, enc2] += 1

        ## save one header
        if (push_to_kspace.rawHead is None) or \
            ((acq.idx.user[5] == push_to_kspace.cenc1) and \
             (acq.idx.user[6] == push_to_kspace.cenc2)):
            push_to_kspace.rawHead = acq.getHead()

    if finalize:
        # support averaging (with or without acquisition weighting)
        push_to_kspace.kspace /= np.maximum(1, push_to_kspace.counter[:,:,np.newaxis,np.newaxis])

        # rearrange kspace for bart - target size: (nx, ny, nz, nc)
        push_to_kspace.kspace = np.moveaxis(push_to_kspace.kspace, 3, 0)
        push_to_kspace.finalized = True


# Sorting of k-space data
def sort_into_kspace(group, metadata):  # used for acs scan
    for acq in group:
        push_to_kspace(acq, metadata)
    if not push_to_kspace.finalized:
        push_to_kspace(finalize=True)
    kspace = push_to_kspace.kspace
    del push_to_kspace.kspace  # make room for imaging scan
    return kspace


def ecaltwo(nx, ny, sig):
    gpu_str = "-g" if use_gpu else ""
    maps = bart(1, f'ecaltwo {gpu_str} -m {n_maps} {nx} {ny} {sig.shape[2]}', sig)
    return np.moveaxis(maps, 2, 0)  # slice dim first since we need to concatenate it in the next step


def process_acs(group, config, metadata, threads=8, chunk_sz=None):

    if len(group)==0:
        # nothing to do
        return None

    gpu_str = "-g" if use_gpu else ""

    acs = sort_into_kspace(group, metadata)
    nx, ny, nz, nc = acs.shape
    
    if chunk_sz is None:
        def next_powerof2(x):
            return 1 << (x.bit_length()-1)
        # this should usually work
        chunk_sz = next_powerof2(int(24*(500*500*384*32) / (nx*ny*nz*nc*n_maps) + 0.5))
        chunk_sz = max(1, chunk_sz // threads)

    # ESPIRiT calibration
    if chunk_sz>0:
        # espirit_econ: reduce memory footprint by chunking
        logging.debug(f"eon = bart(1, 'ecalib {gpu_str} -m {n_maps} -1', acs)")
        eon = bart(1, f'ecalib {gpu_str} -m {n_maps} -1', acs)  # currently, gpu doesn't help here but try anyway

        # use norms 'forward'/'backward' for consistent scaling with bart's espirit_econ.sh
        # scaling is very important for proper masking in ecaltwo!
        eon = fft.ifft(eon, axis=2, norm='forward')
        tmp = np.zeros(eon.shape[:2] + (nz-eon.shape[2],) + eon.shape[-1:], dtype=eon.dtype)
        cutpos = eon.shape[2]//2
        eon = np.concatenate((eon[:,:,:cutpos,:], tmp, eon[:,:,cutpos:,:]), axis=2)
        eon = fft.fft(eon, axis=2, norm='backward')

        tic = perf_counter()

        logging.debug(f"loop: 'bart ecaltwo {gpu_str} -m {n_maps} {nx} {ny} {chunk_sz}' with {threads} threads")
        # sensmaps = np.zeros(acs.shape + ((n_maps,) if n_maps>1 else ()), dtype=acs.dtype)

        slcs = (slice(i, i+chunk_sz) for i in range(0, nz, chunk_sz))
        chunks = (eon[:,:,sl] for sl in slcs)
        
        if threads is None or threads <= 0:
            sensmaps = [partial(ecaltwo, nx, ny) for sig in chunks]
        else:
            with Pool(threads) as p:
                sensmaps = p.map(partial(ecaltwo, nx, ny), chunks)

        sensmaps = np.concatenate(sensmaps, axis=0)
        sensmaps = np.moveaxis(sensmaps, 0, 2)

        logging.debug(f"ecalib with chunk_sz={chunk_sz} and {threads} thread(s): {perf_counter()-tic} s")
    else:
        logging.debug(f"sensmaps = bart(1, 'ecalib {gpu_str} -m {n_maps}', acs)")
        sensmaps = bart(1, f'ecalib {gpu_str} -m {n_maps}', acs)

    # np.save(os.path.join(debugFolder, "acs.npy"), acs)
    # np.save(os.path.join(debugFolder, "sensmaps.npy"), sensmaps)

    return sensmaps


def pics_chunk(pics_str, chunk):
    block = chunk[0]
    sensmap = chunk[1]
    cut_slc = chunk[2]

    block = cfft(block, 0)
    block = bart(1, pics_str, block, sensmap)
    block = block[cut_slc]
    block = block[(slice(None),) * 3 + (block.ndim-3) * (0,)]  # select first 3 dims
    return abs(block)


#def process_raw(data, rawHead, connection, config, metadata, sensmaps=None, chunk_sz=226, chunk_overlap=4, max_iter=30):
# def process_raw(data, rawHead, connection, config, metadata, sensmaps=None, chunk_sz=152, chunk_overlap=4, max_iter=30, threads=1):
def process_raw(data, rawHead, connection, config, metadata, sensmaps=None, chunk_sz=114, chunk_overlap=4, max_iter=30, threads=1):

    if threads is not None and threads > 1:
        chunk_sz = 128 // threads

    logging.debug(f"Raw data is size {data.shape}; Sensmaps shape is {sensmaps.shape}")

    gpu_str = "-g" if use_gpu else ""
    pics_str = f'pics -l1 -r0.002 -S -i {max_iter} -d5 {gpu_str} -w 615'  # hard-code scale (-w) for consistency in chunks

    tic = perf_counter()
    if chunk_sz==0 or chunk_sz>=data.shape[0]:
        logging.debug(f"data = bart(1, '{pics_str}', data, sensmaps)")
        data = bart(1, pics_str, data, sensmaps)
        data = abs(data[(slice(None),) * 3 + (data.ndim-3) * (0,)])  # select first 3 dims
        threads = 1
    else:
        data = cifft(data, 0)
        nx = data.shape[0]

        if threads is not None and threads > 1:
            slcs = (slice(0 if i<chunk_overlap else i-chunk_overlap, i+chunk_sz+chunk_overlap) for i in range(0, nx, chunk_sz))
            cut_start = ((i if i<chunk_overlap else chunk_overlap) for i in range(0, nx, chunk_sz))
            cut_stop = ((a + chunk_sz if i+chunk_sz < nx else -1) for a, i in zip(cut_start, range(0, nx, chunk_sz)))
            cut_slcs = (slice(a, b) for a, b in zip(cut_start, cut_stop))
            blocks = (data[sl] for sl in slcs)
            sensblocks = (sensmaps[sl] for sl in slcs)

            with Pool(threads) as p:
                data_out = p.map(partial(pics_chunk, pics_str), zip(blocks, sensblocks, cut_slcs))
        else:
            data_out = []
            for i in range(0, nx, chunk_sz):
                sl = slice(i, i+chunk_sz)
                sl_len = len(range(*sl.indices(nx)))
                ix0 = max(0, i-chunk_overlap)
                sl_ov = slice(ix0, i+chunk_sz+chunk_overlap)
                block = cfft(data[sl_ov], 0)
                logging.debug(f"data = bart(1, '{pics_str}', block, sensmaps[{sl_ov}])")
                block = bart(1, pics_str, block, sensmaps[sl_ov])
                block = block[slice(i-ix0, i-ix0+sl_len)]

                # noch nicht ganz richtig, nx-1 kommt als groesse heraus:
                # sl = slice(0 if i<chunk_overlap else i-chunk_overlap, i+chunk_sz+chunk_overlap)
                # block = cfft(data[sl], 0)
                # cut_start = i if i<chunk_overlap else chunk_overlap
                # cut_stop = cut_start + chunk_sz if i+chunk_sz < nx else -1
                # logging.debug(f"data = bart(1, '{pics_str}', block, sensmaps[{sl}])")
                # block = bart(1, pics_str, block, sensmaps[sl])
                # block = block[slice(cut_start, cut_stop)]

                block = block[(slice(None),) * 3 + (block.ndim-3) * (0,)]  # select first 3 dims
                data_out.append(abs(block))

        data = np.concatenate(data_out, axis=0)
        del data_out
        logging.debug(f"data.shape = {data.shape}")    

    logging.debug(f"pics with chunk_sz={chunk_sz} and {threads} thread(s): {perf_counter()-tic} s")


    # zero-fill up to full fov in case partial recon. in x
    if reduce_fov_x and zf_to_orig_sz:
        sz_x = metadata.encoding[0].reconSpace.matrixSize.x
        if data.shape[0] < sz_x:
            data_out = data_out = np.zeros((sz_x,) + data.shape[1:], dtype=np.float32)
            data_out[sel_x] = data
            data = data_out

    # Remove phase oversampling
    if data.shape[1] > metadata.encoding[0].reconSpace.matrixSize.y:
        offset = (data.shape[1] - metadata.encoding[0].reconSpace.matrixSize.y)//2
        data = data[:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.y]

    # Remove partition oversampling
    if data.shape[2] > metadata.encoding[0].reconSpace.matrixSize.z:
        offset = (data.shape[2] - metadata.encoding[0].reconSpace.matrixSize.z)//2
        data = data[:,:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.z]

    logging.debug("Image data is size %s" % (data.shape,))
    
    return process_image(data, rawHead, config, metadata)


def process_image(data, rawHead, config, metadata):

    field_of_view = (metadata.encoding[0].reconSpace.fieldOfView_mm.x,
                     metadata.encoding[0].reconSpace.fieldOfView_mm.y,
                     metadata.encoding[0].reconSpace.fieldOfView_mm.z)

    recon_matrix = (metadata.encoding[0].reconSpace.matrixSize.x,
                    metadata.encoding[0].reconSpace.matrixSize.y,
                    metadata.encoding[0].reconSpace.matrixSize.z)

    if save_unsigned:
        int_max = 65535
    else:
        int_max = 32767

    # save one scaling in 'static' variable
    try:
        process_image.imascale
    except:
        empirical_factor = 5e-9
        resolution = tuple(fov/mat for fov, mat in zip(field_of_view, recon_matrix))
        process_image.imascale = int_max * empirical_factor
        process_image.imascale *= np.sqrt(np.prod(data.shape))
        process_image.imascale /= np.prod(resolution)

        #test
        # process_image.imascale = int_max/np.max(data)
        # not sure whether we need to account for chunksz or sel_x (probably)

    data *= process_image.imascale

    export_nifti(data, metadata, os.path.join(debugFolder, 'img.nii.gz'))

    # convert to int
    data = np.minimum(int_max, np.floor(data))
    data = data.astype(np.uint16 if save_unsigned else np.int16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           str((int_max+1)//2),
                         'WindowWidth':            str(int_max+1),
                         'Keep_image_geometry':    1})

    # logging.debug("Image MetaAttributes: %s", xml)

    # Flip matrix in RO/PE/3D to be consistent with ICE
    data = np.flip(data, (0, 1, 2))

    # Format as ISMRMRD image data    
    image = ismrmrd.Image.from_array(data)
    image.setHead(mrdhelper.update_img_header_from_raw(image.getHead(), rawHead))
    image.field_of_view = tuple(ctypes.c_float(fov) for fov in field_of_view)
    image.image_index = 1

    logging.debug("Image data has %d elements", image.data.size)

    # Add image orientation directions to MetaAttributes if not already present
    if meta.get('ImageRowDir') is None:
        meta['ImageRowDir'] = ["{:.18f}".format(image.getHead().read_dir[0]), "{:.18f}".format(image.getHead().read_dir[1]), "{:.18f}".format(image.getHead().read_dir[2])]

    if meta.get('ImageColumnDir') is None:
        meta['ImageColumnDir'] = ["{:.18f}".format(image.getHead().phase_dir[0]), "{:.18f}".format(image.getHead().phase_dir[1]), "{:.18f}".format(image.getHead().phase_dir[2])]

    xml = meta.serialize()
    image.attribute_string = xml

    # logging.debug("Image MetaAttributes: %s", xml)

    return image
