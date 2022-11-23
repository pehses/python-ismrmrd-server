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
apply_prewhitening = True
use_multiprocessing = True


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
                elif sensmaps[item.idx.slice] is None and len(acsGroup[item.idx.slice]):
                    cc_matrix[item.idx.slice] = calibrate_cc(acsGroup[item.idx.slice], ncc)

                    # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
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
    nx = ncol//2

    logging.debug(f"Encoded Space is {ncol}, {nprj}, {npar}, {ncha}")

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

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # calc radial trajectory - center traj. at ncol//2 ('-c')
    center_col_in_center = True
    trj_type = 'lin'
    # trj_type = 'ga'
    # trj_type = 'tga'

    bart_cmd = f'traj -r -x{ncol} -y{nprj}'
    if center_col_in_center:
        bart_cmd += ' -c'
    if trj_type == 'ga':
        bart_cmd += ' -G -D'
    elif trj_type == 'tga':
        bart_cmd += ' -s7 -D'
    elif nprj%2:
        bart_cmd += ' -D'
    trj = bart(1, bart_cmd) / 2
    
    # Flip matrix in RO to be consistent with ICE
    # we could do this after image recon, but this would result in one voxel shift
    trj[:2] *= -1
    # switch RO & PE for consistency with ICE
    trj[[0, 1]] = trj[[1, 0]]

    if sensmaps is None:  # adjoint nufft requires density compensation
        if center_col_in_center:
            ramlak = abs(np.arange(-ncol//2, ncol//2)) + 0.5
        else:
            ramlak = abs(np.arange(-ncol//2, ncol//2) + 0.5) + 0.5
        data *= ramlak

    # move col to front
    data = np.moveaxis(data, 3, 0)

    logging.debug(f'sensmaps {sensmaps}')
    if sensmaps is not None:
        # bring projections and partitions into one dim (need to do same for trj for 3D meas!)
        data = np.moveaxis(data, -1, 1)
        data = data.reshape([*data.shape[:2], -1])
        data = np.moveaxis(data, 1, -1)[np.newaxis]
        # wait for sensmaps:
        if isinstance(sensmaps, multiprocessing.pool.AsyncResult):
            sensmaps = sensmaps.get()
        logging.info(f'trj: {trj.shape}, data: {data.shape}, sensmaps: {sensmaps.shape}')
        try:
            # data = bart(1, 'pics -S -e -l1 -r 0.0001 -i 50 -t', trj, data, sensmaps)
            data = bart(1, 'pics -S -e -l1 -r 0.0001 -t', trj, data, sensmaps)
            logging.info(bart.stdout)
        except:
            logging.info(bart.stdout)
            logging.debug(bart.stderr)
            raise RuntimeError
        data = np.atleast_3d(abs(data))
        logging.debug(f'data.shape = {data.shape}')
    else:  # normal gridding
        # Fourier Transform in partition direction
        if npar>0:
            data = cfft(data, axis=2)
        im = []
        for dat in np.moveaxis(data, 2, 0):
            try:
                im.append(bart(1, f"nufft -a -d {nx}:{nx}:1", trj, dat[np.newaxis]))
                logging.info(bart.stdout)
            except:
                logging.info(bart.stdout)
                logging.debug(bart.stderr)
                raise RuntimeError
        data = np.concatenate(im, axis=2)

        # Sum of squares coil combination
        # Data will be [PE RO par]
        data = np.sqrt(np.sum(np.square(np.abs(data)), axis=-1))

        # pre-scale data based on radial undersampling:
        data *= nx/nprj

    # Remove partition oversampling
    if data.shape[2] > metadata.encoding[0].reconSpace.matrixSize.z:
        offset = (data.shape[2] - metadata.encoding[0].reconSpace.matrixSize.z)//2
        data = data[:,:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.z]

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

    # Normalize and convert to int16
    image_scale = 32767/data.max()
    logging.debug(f"Image scale: {image_scale}")
    data *= image_scale
    data = np.around(data).astype(np.int16)

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
    image = ismrmrd.Image.from_array(data)
    image.setHead(mrdhelper.update_img_header_from_raw(image.getHead(), rawHead))
    image.field_of_view = tuple(ctypes.c_float(fov) for fov in field_of_view)

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
