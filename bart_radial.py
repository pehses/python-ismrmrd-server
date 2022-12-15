import ismrmrd
import os
import logging
import numpy as np
import ctypes
import mrdhelper
import tempfile
from bart import bart
# from cfft import cfft, cifft, fft, ifft
from cfft_mkl import cfft, cifft, fft, ifft, cfftn, cifftn

# Folder for sharing data/debugging
# tempfile.tempdir = "/tmp"  # benchmark 1: 148.6 s
tempfile.tempdir = "/dev/shm"  # slightly faster bart wrapper; benchmark 1: 131.1 s

import multiprocessing
import multiprocessing.pool
from pe_helpers import NestablePool, calibrate_prewhitening, calibrate_cc, apply_column_ops, apply_cc, process_acs


# Folder for debug output files
debugFolder = "/tmp/share/debug"

ncc = 0
apply_prewhitening = False  # already applied in ice prog!?
use_multiprocessing = True
fft_in_par = True  # use normal fft in par dir. instead of gridding
window_sz = 34  # for ga/tga reconstructions
pics_opts = '-g -S -e -l1 -r 0.01 -i50'
rms_recon = False
# pics_opts = '-g -S -e'
perform_fit = True  # only relevant in case of gr sampling
perform_pca = False
perform_moba = False  # mutually exclusive with perform_fit, rms_recon

if perform_moba:
    perform_fit = False
    rms_recon = False


def calc_traj(ncol, nprj, trj_type='lin', center_col_in_center=True, is_ute=False):

    gr = (1+np.sqrt(5)) / 2
    if trj_type == 'ga':
        angles = (np.pi/gr * np.arange(nprj))%(2*np.pi)
    elif trj_type == 'tga':
        angles = (np.pi / (gr + 7 - 1) * np.arange(nprj))%(2*np.pi)
    elif nprj%2 or is_ute:
        angles = 2*np.pi/nprj * np.arange(nprj)
    else:
        angles = np.pi/nprj * np.arange(nprj)

    if is_ute:
        trj_1d = np.arange(ncol, dtype=float)  # does not account for ramp-sampling yet!
    else:
        trj_1d = np.arange(-ncol//2, ncol//2, dtype=float) + (0 if center_col_in_center else 0.5)
    trj_1d /= 2  # account for oversampling
    
    return trj_1d, angles


def calc_ramlak(trj_1d, angles=None, is_ute=False):
    ramlak = 0.25/2 + abs(trj_1d[:, np.newaxis])
    if angles is not None:
        # account for angular sampling anisotropy:
        angle_dist = []
        for angle in angles:
            angle_diff = np.angle(np.exp(1j*angles) * np.exp(-1j*angle))
            if not is_ute:
                # make sure to account for other half of projection
                for key, angle in enumerate(angle_diff):
                    if angle > np.pi / 2:
                        angle_diff[key] = angle - np.pi
                    elif angle < -np.pi / 2:
                        angle_diff[key] = angle + np.pi
            right_dist = angle_diff[angle_diff>0].min()
            left_dist = -angle_diff[angle_diff<0].max()
            angle_dist.append((left_dist+right_dist)/2.)
        angle_dist = np.asarray(angle_dist)
        angle_dist /= angle_dist.mean()
        ramlak = ramlak * angle_dist[np.newaxis]
    return ramlak


def recon_radial_3d(data, trj_1d, angles, metadata, sensmaps=None, fft_in_par=False):
    ncol, nprj, npar, cha = data.shape
    nx = ncol//2
    center_par = metadata.encoding[0].encodingLimits.kspace_encoding_step_2.center

    is_ute = abs(trj_1d[0]) < 1
    ramlak = calc_ramlak(trj_1d, angles, is_ute)
    if sensmaps is None:  # adjoint nufft requires density compensation
        data *= ramlak[:, :, np.newaxis, np.newaxis]

    # create trajectory by rotating trj_1d
    rot = np.asarray([np.sin(angles), np.cos(angles), np.zeros(len(angles))])
    trj = rot[:, np.newaxis] * trj_1d[np.newaxis, :, np.newaxis]
    
    # Flip matrix in RO to be consistent with ICE
    # we could do this after image recon, but this would result in one voxel shift
    trj[:2] *= -1
    # switch RO & PE for consistency with ICE
    trj[[0, 1]] = trj[[1, 0]]

    if npar>1 and fft_in_par:  # Fourier Transform in partition direction
        data = cfft(data, axis=2)
        # wait for sensmaps:
        if isinstance(sensmaps, multiprocessing.pool.AsyncResult):
            sensmaps = sensmaps.get()
        # Remove partition oversampling
        if data.shape[2] > metadata.encoding[0].reconSpace.matrixSize.z:
            offset = (data.shape[2] - metadata.encoding[0].reconSpace.matrixSize.z)//2
            data = data[:,:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.z]
            npar = data.shape[2]
            center_par -= offset
            if sensmaps is not None:
                sensmaps = sensmaps[:,:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.z]
        im = list()
        for p in range(data.shape[2]):
            if sensmaps is None:
                try:
                    im.append(bart(1, f"nufft -a -d {nx}:{nx}:1", trj, data[np.newaxis,:,:,p]))
                    logging.info(f'par={p}: successful recon')
                except:
                    logging.info(bart.stdout)
                    logging.debug(bart.stderr)
                    raise RuntimeError(f'Error running nufft for partition {p}')
            else:
                try:
                    im.append(bart(1, f"pics {pics_opts}", data[np.newaxis,:,:,p], sensmaps[:,:,p:p+1], t=trj, p=np.sqrt(ramlak)[np.newaxis]))
                    logging.info(f'par={p}: successful recon')
                except:
                    logging.info(bart.stdout)
                    logging.debug(bart.stderr)
                    raise RuntimeError(f'Error running pics for partition {p}')
            logging.info(bart.stdout)
        data = np.stack(im, axis=2)
    else:
        if npar>1:
            # bring projections and partitions into one dim
            data = np.moveaxis(data, -1, 1)
            data = data.reshape([*data.shape[:2], -1])
            data = np.moveaxis(data, 1, -1)[np.newaxis]
            # concatenate trj. and add partitions to z-dir.
            trj = np.tile(trj[..., np.newaxis], npar)
            trj[2] = (np.arange(npar) - center_par)[np.newaxis, np.newaxis]
            trj = np.moveaxis(trj, 1, -1)
            trj = trj.reshape(*trj.shape[:2], -1)
            trj = np.moveaxis(trj, -1, 1)
        else:
            # first dim needs to be empty for non-cartesian scans
            data = data[np.newaxis,:,:,0]

        try:
            if sensmaps is None:
                data = bart(1, f"nufft -a -d {nx}:{nx}:{npar}", trj, data)
            else:
                # wait for sensmaps:
                if isinstance(sensmaps, multiprocessing.pool.AsyncResult):
                    sensmaps = sensmaps.get()
                data = bart(1, f'pics {pics_opts}', data, sensmaps, t=trj, p=np.sqrt(ramlak)[np.newaxis])
            logging.info(bart.stdout)
        except:
            logging.info(bart.stdout)
            logging.debug(bart.stderr)
            raise RuntimeError
        data = np.atleast_3d(data)

    if rms_recon:
        if sensmaps is None:
            # Sum of squares coil combination
            # Data will be [PE RO par]
            data = np.sqrt(np.sum(np.square(np.abs(data)), axis=-1))

            # pre-scale data based on radial undersampling:
            data *= nx/nprj
        else:
            data = abs(data)  # coil-combination already done

    return data


def convert_to_image(imdata, metadata, rawHead, frame_num=None, series_num=None, image_scale=None):
    if image_scale is None:
        # Normalize and convert to int16
        image_scale = 32767/imdata.max()
    logging.debug(f"Image scale: {image_scale}")
    imdata = imdata * image_scale
    imdata = abs(imdata)  # optional: remove sign!
    imdata = np.around(imdata).astype(np.int16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768',
                         'Keep_image_geometry':    1,
                         'ImageScale':             image_scale })  # store this for our own future use (e.g. in fitting)
    xml = meta.serialize()

    field_of_view = (metadata.encoding[0].reconSpace.fieldOfView_mm.x,
                     metadata.encoding[0].reconSpace.fieldOfView_mm.x,
                     metadata.encoding[0].reconSpace.fieldOfView_mm.z)

    # Format as ISMRMRD image data
    image = ismrmrd.Image.from_array(imdata)
    image.setHead(mrdhelper.update_img_header_from_raw(image.getHead(), rawHead))
    image.field_of_view = tuple(ctypes.c_float(fov) for fov in field_of_view)
    if frame_num is not None:
        image.image_index = frame_num + 1
    if series_num is not None:
        image.image_series_index = series_num + 1

    # Add image orientation directions to MetaAttributes if not already present
    if meta.get('ImageRowDir') is None:
        meta['ImageRowDir'] = ["{:.18f}".format(image.getHead().read_dir[0]), "{:.18f}".format(image.getHead().read_dir[1]), "{:.18f}".format(image.getHead().read_dir[2])]

    if meta.get('ImageColumnDir') is None:
        meta['ImageColumnDir'] = ["{:.18f}".format(image.getHead().phase_dir[0]), "{:.18f}".format(image.getHead().phase_dir[1]), "{:.18f}".format(image.getHead().phase_dir[2])]

    xml = meta.serialize()
    image.attribute_string = xml

    # logging.debug("Image MetaAttributes: %s", xml)
    return image


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
            metadata.encoding[0].trajectory,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z,
            metadata.encoding[0].encodedSpace.matrixSize.x,
            metadata.encoding[0].encodedSpace.matrixSize.x,
            metadata.encoding[0].encodedSpace.matrixSize.z
        )

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    acqGroup = []
    noiseGroup = []
    # hard-coded limit of 256 slices (better: use Nslice from protocol)
    max_no_of_slices = 256
    acsGroup = [[] for _ in range(max_no_of_slices)]
    sensmaps = [None] * max_no_of_slices
    optmat = None  # for pre-whitening
    cc_matrix = [None] * max_no_of_slices  # for coil compression

    if use_multiprocessing:
        pool = NestablePool(processes=2)

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

                # Accumulate all imaging readouts in a group
                if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    apply_column_ops(item, optmat, nx=metadata.encoding[0].encodedSpace.matrixSize.x, os_removal=True)
                    acsGroup[item.idx.slice].append(item)
                    continue
                elif sensmaps[item.idx.slice] is None and cc_matrix[item.idx.slice] is None and len(acsGroup[item.idx.slice]):
                    cc_matrix[item.idx.slice] = calibrate_cc(acsGroup[item.idx.slice], ncc)

                    # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                    if not perform_moba:
                        if use_multiprocessing:
                            sensmaps[item.idx.slice] = pool.apply_async(process_acs, (acsGroup[item.idx.slice], config, metadata, True))
                        else:
                            sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, is_radial_seq=True)

                apply_column_ops(item, optmat, nx=item.data.shape[-1], os_removal=False)
                apply_cc(item, cc_matrix[item.idx.slice])
                acqGroup.append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup, config, metadata, sensmaps[item.idx.slice])
                    connection.send_image(image)
                    acqGroup = []

            # ----------------------------------------------------------
            # Image and waveform data messages are not supported
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                logging.info("Received an image, but this is not supported -- discarding")
                continue

            elif isinstance(item, ismrmrd.Waveform):
                logging.info("Received a waveform, but this is not supported -- discarding")
                continue

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Process any remaining groups of raw or image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(acqGroup) > 0:
            logging.info("Processing a group of k-space data (untriggered)")
            image = process_raw(acqGroup, config, metadata)
            connection.send_image(image)
            acqGroup = []

    finally:
        connection.send_close()


def process_raw(group, config, metadata, sensmaps=None):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Format data into single [cha PE RO par] array
    prj = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    par = [acquisition.idx.kspace_encode_step_2 for acquisition in group]

    encoding = metadata.encoding[0]
    ncol = group[0].data.shape[-1]
    nprj = encoding.encodedSpace.matrixSize.y
    npar = encoding.encodedSpace.matrixSize.z
    ncha = group[0].data.shape[0]

    wip_params = {item.name: item.value for item in metadata.userParameters.userParameterLong}
    # wip_params.update({item.name: item.value for item in metadata.userParameters.userParameterDouble})
    traj_params = {item.name: item.value for item in encoding.trajectoryDescription.userParameterLong}
    is_ute = traj_params['ute']

    def unpack_bits(input):
        return np.bitwise_and(
            np.int32(input), 2**np.arange(32, dtype=np.uint32)).astype(bool)
    compact_bool = unpack_bits(wip_params['alFree[6]'])
    center_col_in_center = bool(compact_bool[3])
    compact_sels = (ctypes.c_uint8*4).from_buffer(ctypes.c_int32(wip_params['alFree[0]']))
    if compact_sels[1] == 6:
        trj_type = 'ga'
    elif compact_sels[1] == 7:
        trj_type = 'tga'
    else:
        trj_type = 'lin'  # or shuffled traj. (Fibonacci) / unknown
    logging.debug(f'trj_type = {trj_type}, center_col_in_center = {center_col_in_center}')

    # Use the zero-padded matrix size
    data = np.zeros((nprj, 
                     npar, 
                     ncha,
                     ncol), 
                     group[0].data.dtype)

    rawHead = None
    for acq, prj, par in zip(group, prj, par):
        if (prj < nprj) and (par < npar):
            # TODO: Account for asymmetric echo in a better way

            data[prj, par] = acq.data

            # center line of k-space is encoded in user[5]
            if (rawHead is None) or (np.abs(acq.getHead().idx.kspace_encode_step_1 - acq.getHead().idx.user[5]) < np.abs(rawHead.idx.kspace_encode_step_1 - rawHead.idx.user[5])):
                rawHead = acq.getHead()

    # move col to front
    data = np.moveaxis(data, 3, 0)

    logging.debug("Raw data is size %s" % (data.shape,))
    # np.save(debugFolder + "/" + "raw.npy", data)

    trj_1d, angles = calc_traj(ncol, nprj, trj_type, center_col_in_center, is_ute)

    if perform_moba:
        if npar>1:  # and fft_in_par:  # Fourier Transform in partition direction
            data = cfft(data, axis=2)
            # Remove partition oversampling
            if data.shape[2] > metadata.encoding[0].reconSpace.matrixSize.z:
                offset = (data.shape[2] - metadata.encoding[0].reconSpace.matrixSize.z)//2
                data = data[:,:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.z]
            npar = data.shape[2]
        ndummy = 2
        TR = metadata.sequenceParameters.TR[0]
        TI = metadata.sequenceParameters.TI[0]
        nframes = nprj // window_sz
        time_vec = TI + TR * (ndummy + window_sz/2 + window_sz * np.arange(nframes))
        time_vec *= 1e-3 # ms -> s

        if not (trj_type == 'ga' or trj_type == 'tga'):
            raise ValueError('moba recon currently only works with golden-angle sampling')

        time_vec = np.expand_dims(time_vec, [0, 1, 2, 3, 4])
        nx = ncol//2
        res = np.zeros((3, nx, nx, npar), dtype=np.complex64)
        for p in range(npar):
            kspace = np.zeros((1, ncol, window_sz, ncha, 1, nframes), dtype=np.complex64)
            trj = np.zeros((3, ncol, window_sz, 1, 1, nframes), dtype=np.complex64)
            for frame in range(nframes):
                sel = slice(frame*window_sz, (frame+1)*window_sz)
                kspace[..., frame] = data[np.newaxis,:,sel,p,:, np.newaxis]
                # create trajectory by rotating trj_1d
                rot = np.asarray([np.sin(angles[sel]), np.cos(angles[sel]), np.zeros(len(angles[sel]))])
                trj[:,:, :, 0, 0, frame] = rot[:, np.newaxis] * trj_1d[np.newaxis, :, np.newaxis]
            # Flip matrix in RO to be consistent with ICE
            # we could do this after image recon, but this would result in one voxel shift
            trj[:2] *= -1
            # switch RO & PE for consistency with ICE
            trj[[0, 1]] = trj[[1, 0]]

            logging.debug(f'trj.shape = {trj.shape}, kspace.shape = {kspace.shape}, time_vec.shape = {time_vec.shape}, res.shape = {res.shape}')
            
            # tmp = bart(1, f'moba -L -g -d4 -l1 -i8 -C100 -j0.09 -B0.0', kspace, time_vec, t=trj)
            tmp = bart(1, f'moba -L -g -d4 -l1 -i8 -C100 -j0.09 -B0.0 -n', kspace, time_vec, t=trj)
            # tmp = bart(1, f'moba -L -g -d4 -l1 -i8 -C100 -j0.09 -n', kspace, time_vec, t=trj)
            # tmp = bart(1, f'moba -L -g -d4 -l1 -j0.01 -B0.0 -n', kspace, time_vec, t=trj)
            logging.info(bart.stdout)
            logging.info(bart.stderr)
            res[:,:,:,p] = np.moveaxis(tmp.squeeze()[slice(nx//2,(nx*3)//2), slice(nx//2,(nx*3)//2)], -1, 0)
        res = abs(res)
        images = list()
        image_scale = 32767/max(res[0].max(), res[1].max())
        images.append(convert_to_image(res[0], metadata, rawHead, 0, 0, image_scale))  # Mss
        images.append(convert_to_image(res[1], metadata, rawHead, 0, 1,  image_scale))  # M0
        T1s = np.zeros_like(res[2])
        T1s[res[2]!=0] = 1/(res[2][res[2]!=0])
        images.append(convert_to_image(T1s, metadata, rawHead, 0, 2, 10000))  # T1s
        # images.append(convert_to_image(res[2], metadata, rawHead, 0, 2, 1000))  # R1s
        return images
    elif trj_type == 'ga' or trj_type == 'tga':
        nframes = nprj // window_sz
        imdata = list()
        for frame in range(nframes):
            sel = slice(frame*window_sz, (frame+1)*window_sz)
            imdata.append(recon_radial_3d(data[:,sel], trj_1d, angles[sel], metadata, sensmaps, fft_in_par))

        if use_multiprocessing:
            for key,im in enumerate(imdata):
                if isinstance(im, multiprocessing.pool.AsyncResult):
                    imdata[key] = im.get()
        imdata = np.array(imdata)

        if not rms_recon:
            if sensmaps is None:
                # coil-combination has not yet been performed
                logging.debug(f'starting walsh sensitivity calc.')
                tmp = bart(1, 'fft 7', imdata.sum(0))
                sens = bart(1, f'caldir 32', tmp)

                # tmp = bart(1, 'walsh -b 7:7:7', tmp)
                logging.info(bart.stdout)
                # sens = bart(1, f'ecaltwo -m1 {tmp.shape[0]} {tmp.shape[1]} {tmp.shape[2]}', tmp)
                logging.debug(f'applying walsh weights , {sens.shape}')
                imdata = (np.conj(sens[np.newaxis]) * imdata).sum(axis=-1)
                logging.debug(f'imdata.shape = {imdata.shape}')

            # phase correction and transform to real numbers
            imdata *= np.exp(-1j * np.angle(imdata[-1]))
            imdata = np.sign(imdata.real) * np.abs(imdata)

        images = []
        image_scale = 0.5 * 32767/abs(imdata).max()
        for frame, im in enumerate(imdata):
            images.append(convert_to_image(im, metadata, rawHead, frame, 0, image_scale))
        np.save(debugFolder + "/" + "imdata.npy", imdata)

        series_index = 1
        if perform_pca:
            # PCA test           
            shp = list(imdata.shape)
            imdata = imdata.reshape((shp[0], -1))
            training = imdata @ np.conj(imdata).T
            u,s,v = np.linalg.svd(training, full_matrices=False)
            v = v[:3,:]  # only keep first 3 eigenvalues
            pc_red = v.T @ np.conj(v)
            imdata = pc_red @ imdata
            imdata = imdata.reshape(shp)
            for frame, im in enumerate(imdata):
                images.append(convert_to_image(im, metadata, rawHead, frame, series_index, image_scale))
            series_index += 1
        
        if perform_fit:
            ndummy = 2
            TR = metadata.sequenceParameters.TR[0]
            TI = metadata.sequenceParameters.TI[0]
            time_vec = TI + TR * (ndummy + window_sz/2 + window_sz * np.arange(nframes))
            time_vec *= 1e-3 # ms -> s
            logging.debug(f'time_vec = {time_vec}')
            logging.debug(f'imdata.shape = {imdata.shape}')


            mask = abs(imdata).mean(axis=0)
            mask = mask > (0.8 * mask.mean())
            imdata = imdata[:, mask]

            def ir_abs(x, a, b, r1):
                return abs(a - b * np.exp(-r1 * x))

            def ir_real(x, a, b, r1):
                return a - b * np.exp(-r1 * x)

            from scipy.optimize import curve_fit
            a = np.zeros_like(imdata[0])
            b = np.zeros_like(imdata[0])
            r1s = np.zeros_like(imdata[0])
            for key, ydata in enumerate(imdata.T):
                p0 = list()
                p0.append(ydata[-1])
                p0.append(abs(ydata[0]) + abs(ydata[-1]))
                p0.append(1) # 1/seconds
                try:
                    popt, pcov = curve_fit((ir_abs if rms_recon else ir_real), time_vec, ydata, p0=p0)
                except:
                    popt = [0, 0, 0]
                a[key] = popt[0]
                b[key] = popt[1]
                r1s[key] = popt[2]

            t1s = np.zeros(mask.shape)
            t1s[mask] = r1s
            t1s[t1s!=0] = 1/t1s[t1s!=0]
            s0 = np.zeros(mask.shape)
            s0[mask] = b - a

            images.append(convert_to_image(s0, metadata, rawHead, 0,  series_index, image_scale))
            images.append(convert_to_image(t1s, metadata, rawHead, 0, series_index + 1,  10000))
        return images
    else:
        imdata = recon_radial_3d(data, trj_1d, angles, metadata, sensmaps, fft_in_par)
        return convert_to_image(imdata, metadata, rawHead)
