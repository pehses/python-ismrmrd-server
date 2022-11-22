import ismrmrd
import os
import logging
import numpy as np
import ctypes
import mrdhelper
import tempfile
from bart import bart
# from cfft import cfft, cifft, fft, ifft
from cfft_mkl import cfft, cifft, fft, ifft


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
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # Accumulate all imaging readouts in a group
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA)):
                    acqGroup.append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup, config, metadata)
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

def process_raw(group, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Format data into single [cha PE RO par] array
    prj = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    par = [acquisition.idx.kspace_encode_step_2 for acquisition in group]

    encoding = metadata.encoding[0]
    ncol = 2*encoding.encodedSpace.matrixSize.x
    nprj = encoding.trajectoryDescription.userParameterLonginterleaves
    npar = encoding.encodedSpace.matrixSize.z
    ncha = group[0].data.shape[0]

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

    # Flip matrix in RO to be consistent with ICE
    # we could do this after image recon, but this would result in one voxel shift
    # however, now we need to make sure that trajectory is correct in case of "CenterColInCenter"
    data = np.flip(data, (-1))

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # calc radial trajectory - center traj. at ncol//2 ('-c')
    center_col_in_center = True
    trj_type = 'lin'
    # trj_type = 'ga'
    # trj_type = 'tga'
    # bart_cmd = f'traj -r -c -o2 -x{ncol//2} -y{nprj}'

    if center_col_in_center:
        bart_cmd = f'traj -r -x{ncol+1} -y{nprj}'
    else:
        bart_cmd = f'traj -r -o2 -x{ncol//2} -y{nprj}'
    if trj_type == 'ga':
        bart_cmd += ' -G -D'
    elif trj_type == 'tga':
        bart_cmd += ' -s7 -D'
    elif nprj%2:
        bart_cmd += ' -D'
    trj = bart(1, bart_cmd)
    
    if center_col_in_center:
        trj = trj[:,1:]/2
        ramlak = abs(np.arange(-ncol//2, ncol//2)) + 0.5
    else:
        ramlak = abs(np.arange(-ncol//2, ncol//2) + 0.5) + 0.5
    data *= ramlak

    # move col to front
    data = np.moveaxis(data, 3, 0)

    # Fourier Transform in partition direction
    if npar>0:
        data = cfft(data, axis=2)

    # BART gridding
    im = []
    for dat in np.moveaxis(data, 2, 0):
        im.append(bart(1, 'nufft -i -t -m20 -d %d:%d:%d'%(ncol//2, ncol//2, 1), trj, dat[np.newaxis]))
        # im.append(bart(1, 'nufft -i -m20 -g -d %d:%d:%d'%(ncol//2, ncol//2, 1), trj, dat[np.newaxis]))
    data = np.concatenate(im, axis=2)

    # Sum of squares coil combination
    # Data will be [PE RO par]
    data = np.sqrt(np.sum(np.square(np.abs(data)), axis=-1))

    # Remove partition oversampling
    if data.shape[2] > metadata.encoding[0].reconSpace.matrixSize.z:
        offset = (data.shape[2] - metadata.encoding[0].reconSpace.matrixSize.z)//2
        data = data[:,:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.z]

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

    # switch RO & PE for consistency with ICE
    data = np.moveaxis(data, 0, 1)

    # Normalize and convert to int16
    data *= 32767/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768',
                         'Keep_image_geometry':    1})
    xml = meta.serialize()


    field_of_view = (metadata.encoding[0].reconSpace.fieldOfView_mm.x,
                     metadata.encoding[0].reconSpace.fieldOfView_mm.x,
                     metadata.encoding[0].reconSpace.fieldOfView_mm.z)

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
