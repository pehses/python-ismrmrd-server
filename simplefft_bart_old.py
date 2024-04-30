
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
from datetime import datetime
from bart import bart

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def groups(iterable, predicate):
    group = []
    for item in iterable:
        group.append(item)

        if predicate(item):
            yield group
            group = []


def conditionalGroups(iterable, predicateAccept, predicateFinish):
    group = []
    try:
        for item in iterable:
            if item is None:
                break

            if predicateAccept(item):
                group.append(item)

            if predicateFinish(item):
                yield group
                group = []
    finally:
        iterable.send_close()


def process(connection, config, metadata):
    logging.info("Config: \n%s", config)
    logging.info("Metadata: \n%s", metadata)

    cal_data = []

    # Discard phase correction lines and accumulate lines until "ACQ_LAST_IN_SLICE" is set
    for key, group in enumerate(conditionalGroups(connection, lambda acq: not acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA), lambda acq: acq.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE))):
        cal_data.append([acq.data for acq in group if acq.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)])

        logging.debug('enc step: %d, %d'%(group[0].idx.kspace_encode_step_1, group[0].idx.kspace_encode_step_1))
        # group = [acq for acq in group if not acq.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)]
        # assume that we now have enough calibration data
        logging.debug('groupkey = %d, len(cal_data) = %d, len(cal_data[0]) = %d'%(key, len(cal_data), len(cal_data[0])))
        # logging.debug('len(cal_data) = %d, len(cal_data[0]) = %d, cal_data[0][0].shape = %d'%(len(cal_data), len(cal_data[0]), cal_data[0][0].shape))
        # sens = bart(1, 'ecalib -m 1 -I ', ksp)  # ESPIRiT calibration
        #reco = bart(1, 'pics', und2x2, sens)
        image = process_group(group, config, metadata)

        logging.debug("Sending image to client:\n%s", image)
        connection.send_image(image)


def process_group(group, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder, mode=0o774)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Format data into single [cha RO PE] array
    data = [acquisition.data for acquisition in group]
    data = np.stack(data, axis=-1)

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # Fourier Transform
    # data = fft.fftshift(data, axes=(1, 2))
    # data = fft.ifft2(data)
    # data = fft.ifftshift(data, axes=(1, 2))

    data = np.moveaxis(data, 0, -1)[:,:,np.newaxis,:]
    sens = bart(1, 'ecalib -m 1 -I ', data)  # ESPIRiT calibration

    data[:,1::2] = 0
    data[::4,:] = 0
    data[1::4,:] = 0

    data = bart(1, 'pics', data, sens)
    
    logging.debug("pics size %s" % (data.shape,))

    # data = bart(1, 'fft -i 6', data)
    

    # Sum of squares coil combination
    data = np.abs(data)
    # data = np.square(data)
    # data = np.sum(data, axis=0)
    # data = np.sqrt(data)

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

    # Normalize and convert to int16
    data *= 32767/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Remove phase oversampling
    nRO = np.size(data,0)
    data = data[int(nRO/4):int(nRO*3/4),:]
    logging.debug("Image without oversampling is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgCrop.npy", data)

    # Format as ISMRMRD image data
    image = ismrmrd.Image.from_array(data, acquisition=group[0])
    image.image_index = 1

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768'})
    xml = meta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has %d elements", image.data.size)

    image.attribute_string = xml
    return image



