import ismrmrd
import os
import logging
import traceback
import numpy as np
import constants
import mrdhelper
import ctypes
import xml.dom.minidom
from scipy.optimize import curve_fit


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
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    imgGroup = []
    waveformGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Image):
                # Only process phase images
                if item.image_type is ismrmrd.IMTYPE_MAGNITUDE:
                    imgGroup.append(item)
                else:
                    connection.send_image(item)
                    continue

            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)

            # ----------------------------------------------------------
            # Ignore raw k-space data
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Acquisition):
                strWarn = "Received an ismrmrd.Acquisition which is ignored by this analysis"
                logging.warning(strWarn)
                connection.send_logging(constants.MRD_LOGGING_INFO, strWarn)

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Extract raw ECG waveform data. Basic sorting to make sure that data 
        # is time-ordered, but no additional checking for missing data.
        # ecgData has shape (5 x timepoints)
        # if len(waveformGroup) > 0:
        #     waveformGroup.sort(key = lambda item: item.time_stamp)
        #     ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
        #     ecgData = np.concatenate(ecgData,1)

        # Process any remaining groups of raw or image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, connection, config, metadata)
            logging.debug("Sending images to client")
            connection.send_image(image)
            imgGroup = []

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()


def ir_abs(x, a, b, r1):
    return abs(a - b * np.exp(-r1 * x))


def process_image(images, connection, config, metadata):

    if not os.path.exists(debugFolder):  # Create debug folder if necessary
        os.makedirs(debugFolder)

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Extract some indices for the images
    slice = [img.slice for img in images]; nsli = max(slice) + 1
    set   = [img.set for img in images]; nset = max(set) + 1

    # calculate time vector for sets
    ndummy = 2
    TR = 1e-3 * metadata.sequenceParameters.TR[0]
    TI = 1e-3 * metadata.sequenceParameters.TI[0]
    nprj = metadata.encoding[0].encodedSpace.matrixSize.y
    time_vec = TI + TR * (ndummy + nprj/2 + nprj * np.arange(nset))

    imagesOut = images  # pass along original images
    for s in range(nsli):
        imdata = np.asarray([img.data[0,0] for img in images if img.slice==s])
        mask = abs(imdata).mean(axis=0)
        mask = mask > (0.8 * mask.mean())
        imdata = imdata[:, mask]
        a = np.zeros_like(imdata[0])
        b = np.zeros_like(imdata[0])
        r1s = np.zeros_like(imdata[0])
        for key, ydata in enumerate(imdata.T):
            p0 = [ydata[-1], abs(ydata[0]) + abs(ydata[-1]), 1]
            try:
                popt, pcov = curve_fit(ir_abs, time_vec, ydata, p0=p0)
            except:
                popt = [0, 0, 0]
            a[key], b[key], r1s[key] = popt
             
        t1s = np.zeros(mask.shape)
        t1s[mask] = np.divide(1, r1s, np.zeros_like(r1s), where=r1s!=0)
        s0 = np.zeros(mask.shape)
        s0[mask] = b - a

        for img in images:  # get header for first set
            if img.slice==s and img.set==0:
                head = img.getHead()
                meta = ismrmrd.Meta.deserialize(img.attribute_string)
                break
        imagesOut.append(export_image(s0, meta, head, frame_num=s, series_num=1, image_scale=1))
        imagesOut.append(export_image(t1s, meta, head, frame_num=s, series_num=2, image_scale=1000))

    return imagesOut



def export_image(imdata, meta, head, frame_num=None, series_num=None, image_scale=None):
    if image_scale is None:
        # Normalize and convert to int16
        image_scale = 32767/imdata.max()
    logging.debug(f"Image scale: {image_scale}")
    imdata = imdata * image_scale
    imdata = abs(imdata)  # optional: remove sign!
    imdata = np.around(imdata).astype(np.int16)

    # Create new MRD instance for the inverted image
    # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    imageOut = ismrmrd.Image.from_array(imdata, transpose=False)
    data_type = imageOut.data_type

    # Create a copy of the original fixed header and update the data_type
    # (we changed it to int16 from all other types)
    oldHeader = head
    oldHeader.data_type = data_type
    imageOut.setHead(oldHeader)

    # Create a copy of the original ISMRMRD Meta attributes and update
    tmpMeta = meta
    tmpMeta['DataRole']                       = 'Image'
    tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'INVERT']
    tmpMeta['WindowCenter']                   = '16384'
    tmpMeta['WindowWidth']                    = '32768'
    tmpMeta['SequenceDescriptionAdditional']  = 'FIRE'
    tmpMeta['Keep_image_geometry']            = 1
    # tmpMeta['ImageScale']                     = image_scale  # store this for our own future use (e.g. in fitting)

    if tmpMeta.get('ImageRowDir') is None:
        tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]

    if tmpMeta.get('ImageColumnDir') is None:
        tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]

    metaXml = tmpMeta.serialize()
    # logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
    logging.debug("Image data has %d elements", imageOut.data.size)

    imageOut.attribute_string = metaXml
    
    if frame_num is not None:
        imageOut.image_index = frame_num + 1
    if series_num is not None:
        imageOut.image_series_index = series_num + 1
    
    return imageOut
