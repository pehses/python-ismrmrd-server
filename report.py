import ismrmrd
import os
import itertools
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper
import constants
from time import perf_counter
import matplotlib.pyplot as plt

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, mrdHeader):
    logging.info("Config: \n%s", config)

    # mrdHeader should be xml formatted MRD header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("MRD header: \n%s", mrdHeader.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(mrdHeader.encoding))
        logging.info("First encoding is of type '%s', with a matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3", 
            mrdHeader.encoding[0].trajectory, 
            mrdHeader.encoding[0].encodedSpace.matrixSize.x, 
            mrdHeader.encoding[0].encodedSpace.matrixSize.y, 
            mrdHeader.encoding[0].encodedSpace.matrixSize.z, 
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.x, 
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.y, 
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted MRD header: \n%s", mrdHeader)

    # Continuously parse incoming data parsed from MRD messages
    currentSeries = 0
    acqGroup = []
    imgGroup = []
    waveformGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # Accumulate all imaging readouts in a group
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)):
                    acqGroup.append(item)

                # When this criteria is met, run process_data() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                    logging.info("Processing a group of k-space data")
                    image = process_data(acqGroup, connection, config, mrdHeader)
                    connection.send_image(image)
                    acqGroup = []

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # When this criteria is met, run process_group() on the accumulated
                # data, which returns images that are sent back to the client.
                # e.g. when the series number changes:
                if item.image_series_index != currentSeries:
                    logging.info("Processing a group of images because series index changed to %d", item.image_series_index)
                    currentSeries = item.image_series_index
                    image = process_data(imgGroup, connection, config, mrdHeader)
                    connection.send_image(image)
                    imgGroup = []

                # Only process magnitude images -- send phase images back without modification (fallback for images with unknown type)
                if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()

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
            if len(ecgData) > 0:
                ecgData = np.concatenate(ecgData,1)

        # Process any remaining groups of raw or image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(acqGroup) > 0:
            logging.info("Processing a group of k-space data (untriggered)")
            image = process_data(acqGroup, connection, config, mrdHeader)
            connection.send_image(image)
            acqGroup = []

        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_data(imgGroup, connection, config, mrdHeader)
            connection.send_image(image)
            imgGroup = []

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()

def process_data(group, connection, config, mrdHeader):
    if len(group) == 0:
        return []

    # Start timer
    tic = perf_counter()

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Create a dictionary of values to report
    data = {}
    data['protocolName'] = mrdHeader.measurementInformation.protocolName
    data['scanner']      = f'{mrdHeader.acquisitionSystemInformation.systemVendor} {mrdHeader.acquisitionSystemInformation.systemModel} {mrdHeader.acquisitionSystemInformation.systemFieldStrength_T:{".1f" if mrdHeader.acquisitionSystemInformation.systemFieldStrength_T > 1 else ".2f"}}T'
    data['fieldOfView']  = f'{mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.x:.1f} x {mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.y:.1f} x {mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.z:.1f} mm^3'
    data['matrixSize']   = f'{mrdHeader.encoding[0].encodedSpace.matrixSize.x} x {mrdHeader.encoding[0].encodedSpace.matrixSize.y} x {mrdHeader.encoding[0].encodedSpace.matrixSize.z}'

    if isinstance(group[0], ismrmrd.acquisition.Acquisition):
        data['inputData'] = f'{len(group)} readouts'
    elif isinstance(group[0], ismrmrd.image.Image):
        data['inputData'] = f'{len(group)} images'

    # Properties of report image to create
    width       = 512  # pixels
    height      = 512  # pixels
    fontSize    = 12   # points
    lineSpacing = 1.75  # scale relative to fontSize

    # Create a blank image with white background
    plt.style.use('dark_background')
    dpi = 100
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi, frameon=False)

    # Display the blank image to set image size
    ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Starting position (relative to top left corner)
    x0 = width *0.05
    y0 = height*0.05

    # Iterate through dict and print each item
    maxKeyLen   = max([len(key) for key in data.keys()])
    for index, (key, value) in enumerate(data.items()):
        ax.text(x0, y0+index*fontSize*lineSpacing, f'{key:{maxKeyLen}}  {value}',  va='center', fontsize=fontSize, color='white', fontfamily='monospace')

    # Invoke a draw to create a buffer we can copy the pixel data from
    fig.canvas.draw()

    imgReport = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    w, h = fig.canvas.get_width_height()
    imgReport = imgReport.reshape((int(h), int(w), -1))

    plt.imsave(os.path.join(debugFolder, 'report.png'), imgReport)

    # Conversion as per CCIR 601 (https://en.wikipedia.org/wiki/Luma_(video)
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    # Must convert to grayscale float32 or uint16 (float64s are not supported)
    imgGray = rgb2gray(np.asarray(imgReport))
    imgGray = imgGray.astype(np.float32)

    imagesOut = []

    # Create new MRD instance for the report image
    # data has shape [PE RO phs], i.e. [y x].
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    mrdImg = ismrmrd.Image.from_array(imgGray, transpose=False)

    # Set the minimum appropriate ImageHeader information without using a reference acquisition/image as a starting point
    # Note mrdImg.getHead().matrix_size should be used instead of the convenience mrdImg.matrix_size because the latter
    # returns a transposed result.  See: https://github.com/ismrmrd/ismrmrd-python/pull/54
    mrdImg.field_of_view = (mrdImg.getHead().matrix_size[0], mrdImg.getHead().matrix_size[1], mrdImg.getHead().matrix_size[2])

    # Set image orientation dimensions.  Note that the default initialized values (0,0,0) are invalid
    # because they are not unit vectors
    mrdImg.read_dir  = (1, 0, 0)
    mrdImg.phase_dir = (0, 1, 0)
    mrdImg.slice_dir = (0, 0, 1)

    # mrdImg.position is optional, but is relative to the patient_table_position
    # Setting patient_table_position is recommended, otherwise the report may
    # significantly shifted from other images in the series_
    mrdImg.patient_table_position = group[0].patient_table_position

    # Optional, but recommended.  Default value (0) corresponds to midnight
    mrdImg.acquisition_time_stamp = group[0].acquisition_time_stamp

    # Default value of image_type (0) is invalid
    mrdImg.image_type = ismrmrd.IMTYPE_MAGNITUDE

    # Use a different image_series_index to have a separate series than the main
    # images.  Absolute value does not matter, but images with the same
    # image_series_index are grouped together in the same DICOM SeriesNumber
    mrdImg.image_series_index = 0

    # DICOM InstanceNumber. Should be incremented if multiple images in a series
    mrdImg.image_index = 0

    # Set MRD MetaAttributes
    tmpMeta = ismrmrd.Meta()
    tmpMeta['DataRole']               = 'Image'
    tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
    tmpMeta['Keep_image_geometry']    = 1

    # Add image orientation directions to MetaAttributes
    # Note that DICOM image orientation is in LPS coordinates, so if another set of image directional
    # cosines are chosen, they may be flipped/rotated to bring them into LPS coordinate space
    tmpMeta['ImageRowDir']    = ["{:.18f}".format(mrdImg.read_dir[0]),  "{:.18f}".format(mrdImg.read_dir[1]),  "{:.18f}".format(mrdImg.read_dir[2])]
    tmpMeta['ImageColumnDir'] = ["{:.18f}".format(mrdImg.phase_dir[0]), "{:.18f}".format(mrdImg.phase_dir[1]), "{:.18f}".format(mrdImg.phase_dir[2])]

    # Add all of the report data to the MetaAttributes so they can be parsed from the resulting images
    tmpMeta.update(data)

    xml = tmpMeta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    mrdImg.attribute_string = xml
    imagesOut.append(mrdImg)

    return imagesOut