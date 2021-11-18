
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import base64

from reco_helper import calculate_prewhitening, apply_prewhitening
# from bart import bart
from bartpy.wrapper import bart


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
    

def combine_DM(D, M):
    # the combined matrix can be applied to pre-whiten and
    # coil compress the data simultaneously. Example:
    # sig_white_cc = bart(1, 'ccapply -p 12', sig, DM)
    return np.conj(D) @ M


# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier

    try:
        # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))
        # logging.info("Metadata: \n%s", metadata.serialize())

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

    # Continuously parse incoming data parsed from MRD messages
    acqGroup = []
    noiseGroup = []
    waveformGroup = []

    # hard-coded limit of 256 slices (better: use Nslice from protocol)
    acsGroup = [[] for _ in range(256)]
    sensmaps = [None] * 256
    dmtx = None

    _, corr_factors = getCoilInfo()
    corr_factors = corr_factors[:, np.newaxis]
    ncc = 10

    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                if item.is_flag_set(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA):
                    item.data *= corr_factors

                # wip: run noise decorrelation
                if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                    noiseGroup.append(item)
                    continue
                elif len(noiseGroup) > 0 and dmtx is None:
                    noise_data = []
                    for acq in noiseGroup:
                        noise_data.append(acq.data)
                    noise_data = np.concatenate(noise_data, axis=1)
                    # calculate pre-whitening matrix
                    dmtx = calculate_prewhitening(noise_data)
                    del(noise_data)
                
                
                # Accumulate all imaging readouts in a group
                if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                
                if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    continue
                elif len(acsGroup[item.idx.slice]) > 0 and sensmaps[item.idx.slice] is None:
                    # warning: multi-slice 2D slices may be skipped
                    if ncc>0:
                        # calibrate coil compression
                        caldata = list()
                        for caldata in acsGroup[item.idx.slice]:
                            caldata.append(apply_prewhitening(item.data, dmtx))
                        caldata = np.moveaxis(np.asarray(caldata), 1, 0)
                        caldata = caldata.reshape([caldata.shape[0], -1])


                acqGroup.append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                

                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    logging.info("Processing a group of k-space data")
                    if sensmaps[item.idx.slice] is None:
                        # run parallel imaging calibration
                        sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx)
                    image = process_raw(acqGroup, config, metadata, dmtx, sensmaps[item.idx.slice])
                    logging.debug("Sending image to client:\n%s", image)
                    connection.send_image(image)
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
            if sensmaps[item.idx.slice] is None:
                # run parallel imaging calibration
                sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx) 
            image = process_raw(acqGroup, config, metadata, dmtx, sensmaps[item.idx.slice])
            logging.debug("Sending image to client:\n%s", image)
            connection.send_image(image)
            acqGroup = []

    finally:
        connection.send_close()


def sort_into_kspace(group, metadata, dmtx=None, zf_around_center=False):
    # initialize k-space
    nc = metadata.acquisitionSystemInformation.receiverChannels
    nx = group[0].number_of_samples

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

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z

    kspace = np.zeros([ny, nz, nc, nx], dtype=group[0].data.dtype)
    counter = np.zeros([ny, nz], dtype=np.uint16)

    logging.debug("nx/ny/nz: %s/%s/%s; enc1 min/max: %s/%s; enc2 min/max:%s/%s, ncol: %s" % (nx, ny, nz, enc1_min, enc1_max, enc2_min, enc2_max, group[0].data.shape[-1]))

    for acq in group:
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2

        # in case dim sizes smaller than expected, sort data into k-space center (e.g. for reference scans)
        ncol = acq.data.shape[-1]
        cx = nx // 2
        ccol = ncol // 2
        col = slice(cx - ccol, cx + ccol)

        if zf_around_center:
            cy = ny // 2
            cz = nz // 2

            cenc1 = (enc1_max+1) // 2
            cenc2 = (enc2_max+1) // 2

            # sort data into center k-space (assuming a symmetric acquisition)
            enc1 += cy - cenc1
            enc2 += cz - cenc2
        
        if dmtx is None:
            kspace[enc1, enc2, :, col] += acq.data
        else:
            kspace[enc1, enc2, :, col] += apply_prewhitening(acq.data, dmtx)
        counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc)
    kspace = np.transpose(kspace, [3, 0, 1, 2])

    return kspace


def process_acs(group, config, metadata, dmtx=None):
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)

        sensmaps = bart(1, 'ecalib -m 1 -I ', data)  # ESPIRiT calibration
        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None


def process_raw(group, config, metadata, dmtx=None, sensmaps=None):

    data = sort_into_kspace(group, metadata, dmtx)

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)
    
    # if sensmaps is None: # assume that this is a fully sampled scan (wip: only use autocalibration region in center k-space)
        # sensmaps = bart(1, 'ecalib -m 1 -I ', data)  # ESPIRiT calibration

    if sensmaps is None:
        logging.debug("no pics necessary, just do standard fft")
        # Fourier Transform
        data = fft.fftshift(data, axes=(0, 1, 2))
        data = fft.ifftn(data, axes=(0, 1, 2))
        data = fft.ifftshift(data, axes=(0, 1, 2))

        # Sum of squares coil combination
        data = np.sqrt(np.sum(np.abs(data)**2, axis=-1))
    else:
        data = bart(1, 'pics -r0.01', data, sensmaps)
        data = np.abs(data)

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

    # Remove readout oversampling
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
