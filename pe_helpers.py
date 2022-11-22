import numpy as np
import ismrmrd
import logging
from bart import bart
from time import perf_counter
from cfft_mkl import cfft, cifft, fft, ifft
import multiprocessing
import multiprocessing.pool
from multiprocessing import Pool

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


def apply_column_ops(item, optmat, nx=None, os_removal=True):
    # pre-whitening & os-removal & slicing & resizing

    ncol = item.data.shape[-1]
    if nx is None:
        nx = ncol

    # operations to perform:
    resize = nx != ncol
    whiten = optmat is not None

    data = item.data

    if resize or os_removal:
        if resize:
            # first pad refscan to full col size (cutting not yet supported)
            dsz = nx - ncol
            data = np.pad(data, ((0, 0), (dsz//2, dsz//2)))
            ncol = data.shape[-1]

        # the next operations need to be performed in image space
        data = cifft(data, -1)

        if os_removal:
            data = data[:, slice(ncol//4, (ncol*3)//4)]

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


def calibrate_cc(items, ncc):
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


# Sorting of k-space data
def sort_into_kspace(group, metadata):  # used for acs scan
    
    initialize = True
    for acq in group:
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2

        if initialize:
            initialize = False
            # initialize k-space
            nc = acq.active_channels
            nx = acq.number_of_samples
            ny = metadata.encoding[0].encodedSpace.matrixSize.y
            nz = metadata.encoding[0].encodedSpace.matrixSize.z

            cenc1 = metadata.encoding[0].encodingLimits.kspace_encoding_step_1.center
            cenc2 = metadata.encoding[0].encodingLimits.kspace_encoding_step_2.center
            logging.debug(f'nx={nx}, ny={ny}, nz={nz}, is_acs_scan={acq.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)}')
            
            kspace = np.zeros([ny, nz, nc, nx], dtype=acq.data.dtype)
            counter = np.zeros([ny, nz], dtype=np.uint16)
            logging.debug('push_to_kspace: nx = %d; ny = %d; nz = %d; nc = %d'%(nx, ny, nz, nc))

        ## now sort acq into k-space ##
        offset_enc1 = cenc1 - acq.idx.user[5]  # center line is encoded in user[5]
        offset_enc2 = cenc2 - acq.idx.user[6]  # center partition is encoded in user[6]
        enc1 += offset_enc1
        enc2 += offset_enc2

        kspace[enc1, enc2] += acq.data
        counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc)
    kspace = np.moveaxis(kspace, 3, 0)

    return kspace


def process_acs(group, config, metadata, cal_mode='espirit', n_maps=1, use_gpu=False, threads=8, chunk_sz=None):

    if len(group)==0:
        # nothing to do
        return None

    gpu_str = "-g" if use_gpu else ""
    acs = sort_into_kspace(group, metadata)
    nx, ny, nz, nc = acs.shape

    def ecaltwo(sig):
        maps = bart(1, f'ecaltwo {gpu_str} -m {n_maps} {nx} {ny} {sig.shape[2]}', sig)
        logging.info(bart.stdout)
        if bart.ERR > 0:
            logging.debug(bart.stderr)
            raise RuntimeError
        return np.moveaxis(maps, 2, 0)  # slice dim first since we need to concatenate it in the next step

    if chunk_sz < 0:
        # this should usually work
        gpu_mem = 48 * 1024**3  # bytes
        chunk_sz = gpu_mem / (8*nx*ny*nc*nc*n_maps)
        if threads is not None and threads>1:
            chunk_sz /= threads
        chunk_sz /= 2  # reserve some memory for overhead
        chunk_sz = 2 * (chunk_sz//2)  # round down to multiple of 2
        chunk_sz = int(max(1, chunk_sz))

    if cal_mode.lower() == 'espirit':
        # ESPIRiT calibration
        if chunk_sz is not None and chunk_sz<nz:
            # espirit_econ: reduce memory footprint by chunking
            eon = bart(1, f'ecalib {gpu_str} -m {n_maps} -1', acs)  # currently, gpu doesn't help here but try anyway
            logging.info(bart.stdout)
            if bart.ERR > 0:
                logging.debug(bart.stderr)
                raise RuntimeError

            # use norms 'forward'/'backward' for consistent scaling with bart's espirit_econ.sh
            # scaling is very important for proper masking in ecaltwo!
            tic = perf_counter()
            eon = ifft(eon, axis=2, norm='forward')
            tmp = np.zeros(eon.shape[:2] + (nz-eon.shape[2],) + eon.shape[-1:], dtype=eon.dtype)
            cutpos = eon.shape[2]//2
            eon = np.concatenate((eon[:,:,:cutpos,:], tmp, eon[:,:,cutpos:,:]), axis=2)
            eon = fft(eon, axis=2, norm='backward')
            toc = perf_counter()
            strProcessTime = "FFT interpolation processing time: %.2f s" % (toc-tic)
            logging.info(strProcessTime)

            tic = perf_counter()

            logging.debug(f"loop: 'bart ecaltwo {gpu_str} -m {n_maps} {nx} {ny} {chunk_sz}' with {threads} threads")
            # sensmaps = np.zeros(acs.shape + ((n_maps,) if n_maps>1 else ()), dtype=acs.dtype)

            slcs = (slice(i, i+chunk_sz) for i in range(0, nz, chunk_sz))
            chunks = (eon[:,:,sl] for sl in slcs)

            if threads is None or threads < 2:
                sensmaps = [ecaltwo(sig) for sig in chunks]
            else:
                with Pool(threads) as p:
                    sensmaps = p.map(ecaltwo, chunks)

            sensmaps = np.concatenate(sensmaps, axis=0)
            sensmaps = np.moveaxis(sensmaps, 0, 2)

            logging.debug(f"ecalib with chunk_sz={chunk_sz} and {threads} thread(s): {perf_counter()-tic} s")
        else:
            sensmaps = bart(1, f'ecalib {gpu_str} -m {n_maps}', acs)
            logging.info(bart.stdout)
            if bart.ERR > 0:
                logging.debug(bart.stderr)
                raise RuntimeError
    else:  # simple 'caldir mode
        sensmaps = bart(1, f'caldir 32', acs)
        logging.info(bart.stdout)
        if bart.ERR > 0:
            logging.debug(bart.stderr)
            raise RuntimeError

    return sensmaps