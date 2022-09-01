
import ismrmrd
import os
import logging
import numpy as np
import ctypes
import copy
import xml.dom.minidom
import tempfile
import psutil
from time import perf_counter

from bart import bart
import subprocess
from cfft import cfftn, cifftn
import mrdhelper

from scipy.ndimage import  median_filter, gaussian_filter, binary_fill_holes, binary_dilation
from skimage.transform import resize
from skimage.restoration import unwrap_phase
from dipy.segment.mask import median_otsu

from pulseq_helper import insert_hdr, insert_acq, get_ismrmrd_arrays, check_signature, read_acqs
import reco_helper as rh

""" Reconstruction of imaging data acquired with the Pulseq Sequence via the FIRE framework
    Reconstruction is done with the BART toolbox and the PowerGrid toolbox
"""

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

# tempfile.tempdir = "/dev/shm"  # slightly faster bart wrapper

read_ecalib = False

########################
# Main Function
########################

def process(connection, config, metadata, prot_file):
    
    # -- Some manual parameters --- #

    # Select a slice (only for debugging purposes) - if "None" reconstruct all slices
    process_raw.slc_sel = None

    # Reconstruct only first contrast after that data was acquired
    process_raw.first_contrast = False
    if len(metadata.userParameters.userParameterLong) > 0:
        online_slc = metadata.userParameters.userParameterLong[0].value # Only online reco can send single slice number (different xml)
        if online_slc >= 0:
            logging.debug(f"Dataset is processed online. Only slice {online_slc} is reconstructed.")
            process_raw.slc_sel = int(online_slc)
        else:
            logging.debug(f"Dataset is processed online. Only first contrast is reconstructed.")
            process_raw.first_contrast = True # reconstruct only first contrast, if complete volume is processed online

    # Coil Compression: Compress number of coils by n_compr coils
    n_compr = 0
    n_cha = metadata.acquisitionSystemInformation.receiverChannels
    if n_compr > 0 and n_compr<n_cha:
        process_acs.cc_cha = n_cha - n_compr
        logging.debug(f'Coil Compression from {n_cha} to {process_acs.cc_cha} channels.')
    elif n_compr<0 or n_compr>=n_cha:
        process_acs.cc_cha = n_cha
        logging.debug('Invalid number of compressed coils.')
    else:
        process_acs.cc_cha = n_cha

    # ----------------------------- #

    # Create folder, if necessary
    if len(metadata.userParameters.userParameterString) > 1:
        seq_signature = metadata.userParameters.userParameterString[1].value
        global debugFolder 
        debugFolder += f"/{seq_signature}"
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Check if ecalib maps calculated
    global read_ecalib
    if read_ecalib and not os.path.isfile(debugFolder + "/sensmaps.npy"):
        read_ecalib = False

    # Insert protocol header
    insert_hdr(prot_file, metadata)

    # Read user parameters
    up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}

    # Check SMS, in the 3D case we can have an acceleration factor, but its not SMS
    sms_factor = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2) if metadata.encoding[0].encodingLimits.slice.maximum > 0 else 1
    if sms_factor > 1 and process_raw.slc_sel is not None:
        process_raw.slc_sel = None
        logging.debug("SMS reconstruction is not possible for single slices. Reconstruct whole volume.")

    # Get additional arrays from protocol file for diffusion imaging
    prot_arrays = get_ismrmrd_arrays(prot_file)

    # parameters for reapplying FOV shift
    nsegments = metadata.encoding[0].encodingLimits.segment.maximum + 1
    matr_sz = np.array([metadata.encoding[0].encodedSpace.matrixSize.x, metadata.encoding[0].encodedSpace.matrixSize.y])
    res = np.array([metadata.encoding[0].encodedSpace.fieldOfView_mm.x / matr_sz[0], metadata.encoding[0].encodedSpace.fieldOfView_mm.y / matr_sz[1], 1])

    # parameters for B0 correction
    dwelltime = 1e-6*up_double["dwellTime_us"] # [s]
    t_min = up_double["t_min"] # [s]
    spiral_delay =  up_double["traj_delay"] # [s]
    t_min += int(spiral_delay/dwelltime) * dwelltime # account for removing possibly corrupted ADCs at the start (insert_acq)

    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3", 
            metadata.encoding[0].trajectory.value, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)
    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Log some measurement parameters
    freq = metadata.experimentalConditions.H1resonanceFrequency_Hz
    shim_currents = [v for k,v in up_double.items() if "ShimCurrent" in k]
    ref_volt = up_double["RefVoltage"]
    logging.info(f"Measurement Frequency: {freq}")
    logging.info(f"Shim Currents: {shim_currents}")
    logging.info(f"Reference Voltage: {ref_volt}")

    # Initialize lists for datasets
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1 # all slices acquired (not reduced by sms factor)
    n_contr = 1 if process_raw.first_contrast else metadata.encoding[0].encodingLimits.contrast.maximum + 1
    n_intl = metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1
    half_refscan = True if metadata.encoding[0].encodingLimits.segment.center else False # segment center is misused as indicator for halved number of refscan slices
    if half_refscan and n_slc%2:
        raise ValueError("Odd number of slices with halved number of refscan slices is invalid.")

    acqGroup = [[[] for _ in range(n_contr)] for _ in range(n_slc//sms_factor)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * n_slc
    sensmaps_shots = [None] * n_slc
    img_coord = [None] * (n_slc//sms_factor)
    dmtx = None
    shotimgs = None
    sens_shots = False
    base_trj = None
    skope = False
    process_acs.cc_mat = [None] * n_slc # compression matrix
    process_acs.refimg = [None] * n_slc # save one reference image for DTI analysis (BET masking) 

    if "b_values" in prot_arrays and n_intl > 1:
        # we use the contrast index here to get the PhaseMaps into the correct order
        # PowerGrid reconstructs with ascending contrast index, so the phase maps should be ordered like that
        # WIP: This does not work with multiple repetitions or averages atm (needs more list dimensions)
        shotimgs = [[[] for _ in range(n_contr)] for _ in range(n_slc//sms_factor)]
        sens_shots = True

    # field map, if it was acquired - needs at least 2 reference contrasts
    if 'echo_times' in prot_arrays:
        process_acs.fmap = {'fmap': [None] * n_slc, 'mask': [None] * n_slc, 'name': 'Field Map from reference scan'}
        echo_times = prot_arrays['echo_times']
        te_diff = echo_times[1] - echo_times[0] # [s]
    else:
        process_acs.fmap = None
        te_diff = None

    # read protocol acquisitions - faster than using read_acquisition
    logging.debug("Reading in protocol acquisitions.")
    acqs = read_acqs(prot_file)

    try:
        for item in connection:

            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                if item.traj.size > 0:
                    skope = True # Skope data inserted?

                # insert acquisition protocol
                # base_trj is used to correct FOV shift (see below)
                base_traj = insert_acq(acqs[0], item, metadata)
                acqs.pop(0)
                if base_traj is not None:
                    base_trj = base_traj

                # run noise decorrelation
                if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                    noiseGroup.append(item)
                    continue
                elif len(noiseGroup) > 0 and dmtx is None:
                    noise_data = []
                    for acq in noiseGroup:
                        noise_data.append(acq.data)
                    noise_data = np.concatenate(noise_data, axis=1)
                    # calculate pre-whitening matrix
                    dmtx = rh.calculate_prewhitening(noise_data)
                    del(noise_data)
                    noiseGroup.clear()
                               
                # Skope sync scans
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # skope sync scans
                    continue

                # skip slices in single slice reconstruction
                if process_raw.slc_sel is None or item.idx.slice == process_raw.slc_sel:

                    # Process reference scans
                    if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                        if half_refscan:
                            item.idx.slice *= 2
                            acsGroup[item.idx.slice].append(item)
                            item2 = copy.deepcopy(item)
                            item2.idx.slice += 1
                            acsGroup[item2.idx.slice].append(item2)
                        else:
                            acsGroup[item.idx.slice].append(item)
                        continue

                    # Coil Compression calibration
                    if process_acs.cc_cha < n_cha and process_acs.cc_mat[item.idx.slice] is None:
                        cc_data = [acsGroup[item.idx.slice + n_slc//sms_factor*k] for k in range(sms_factor)]
                        cc_data = [item for sublist in cc_data for item in sublist] # flatten list
                        cc_mat = rh.calibrate_cc(cc_data, process_acs.cc_cha, apply_cc=False)
                        for k in range(sms_factor):
                            process_acs.cc_mat[item.idx.slice + n_slc//sms_factor*k] = cc_mat

                    for k in range(sms_factor):
                        slc_ix = item.idx.slice + n_slc//sms_factor*k
                        # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                        if sensmaps[slc_ix] is None:
                            sensmaps[slc_ix], sensmaps_shots[slc_ix] = process_acs(acsGroup[slc_ix], metadata, dmtx, te_diff, sens_shots) # [nx,ny,nz,nc]
                            acsGroup[slc_ix].clear()
                            # copy data
                            if half_refscan:
                                if slc_ix%2==0:
                                    sensmaps[slc_ix+1] = sensmaps[slc_ix]
                                    sensmaps_shots[slc_ix+1] = sensmaps_shots[slc_ix]
                                    process_acs.refimg[slc_ix+1] = process_acs.refimg[slc_ix]
                                    if process_acs.fmap is not None:
                                        process_acs.fmap['fmap'][slc_ix+1] = process_acs.fmap['fmap'][slc_ix]
                                        process_acs.fmap['mask'][slc_ix+1] = process_acs.fmap['mask'][slc_ix]
                                if slc_ix%2==1:
                                    sensmaps[slc_ix-1] = sensmaps[slc_ix]
                                    sensmaps_shots[slc_ix-1] = sensmaps_shots[slc_ix]
                                    process_acs.refimg[slc_ix-1] = process_acs.refimg[slc_ix]
                                    if process_acs.fmap is not None:
                                        process_acs.fmap['fmap'][slc_ix-1] = process_acs.fmap['fmap'][slc_ix]
                                        process_acs.fmap['mask'][slc_ix-1] = process_acs.fmap['mask'][slc_ix]
                                    

                    # trigger online first-contrast-recon early
                    if process_raw.first_contrast and item.idx.contrast > 0:
                        if len(acqGroup) > 0:
                            process_and_send(connection, acqGroup, metadata, sensmaps, shotimgs, prot_arrays, img_coord)
                        continue

                    # Process imaging scans - deal with ADC segments 
                    # (not needed in newer spiral sequence versions, but kept for compatibility also with other scanners)
                    if item.idx.segment == 0:
                        nsamples = item.number_of_samples
                        t_vec = t_min + dwelltime * np.arange(nsamples) # time vector for B0 correction
                        item.traj[:,3] = t_vec.copy()
                        acqGroup[item.idx.slice][item.idx.contrast].append(item)

                        img_coord[item.idx.slice] = rh.calc_img_coord(metadata, item)
                    else:
                        # append data to first segment of ADC group
                        idx_lower = item.idx.segment * item.number_of_samples
                        idx_upper = (item.idx.segment+1) * item.number_of_samples
                        acqGroup[item.idx.slice][item.idx.contrast][-1].data[:,idx_lower:idx_upper] = item.data[:]

                    if item.idx.segment == nsegments - 1:

                        last_item = acqGroup[item.idx.slice][item.idx.contrast][-1]

                        # Noise whitening
                        if dmtx is not None:
                            last_item.data[:] = rh.apply_prewhitening(last_item.data[:], dmtx)

                        # Apply coil compression to spiral data (CC for ACS data in sort_into_kspace)
                        if process_acs.cc_mat[last_item.idx.slice] is not None:
                            rh.apply_cc(last_item, process_acs.cc_mat[last_item.idx.slice])

                        # Reapply FOV Shift with predicted trajectory
                        rotmat = rh.calc_rotmat(item)
                        shift = rh.pcs_to_gcs(np.asarray(last_item.position), rotmat) # shift [mm] in GCS, as traj is in GCS
                        shift_px = shift / res # shift in pixel
                        last_item.data[:] = rh.fov_shift_spiral_reapply(last_item.data[:], last_item.traj[:], base_trj, shift_px, matr_sz)

                        # filter signal to avoid Gibbs Ringing
                        traj = np.swapaxes(last_item.traj[:,:3],0,1) # traj to [dim, samples]
                        last_item.data[:] = rh.filt_ksp(last_item.data[:], traj, filt_fac=0.95)

                        # Correct the global phase
                        if skope:
                            k0 = last_item.traj[:,4]
                            last_item.data[:] *= np.exp(-1j*k0)

                        # T2* filter
                        t2_star = 40e-3
                        last_item.data[:] *= 1/np.exp(-t_vec/t2_star)

                        # remove ADC oversampling
                        os_factor = up_double["os_factor"] if "os_factor" in up_double else 1
                        if os_factor == 2:
                            rh.remove_os_spiral(last_item)
                        
                    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) and shotimgs is not None:
                        # Reconstruct shot images for phase maps in multishot diffusion imaging
                        # WIP: Repetitions or Averages not possible atm
                        sensmaps_shots_stack = []
                        for k in range(sms_factor):
                            slc_ix = item.idx.slice + n_slc//sms_factor*k
                            sensmaps_shots_stack.append(sensmaps_shots[slc_ix])
                        shotimgs[item.idx.slice][item.idx.contrast] = process_shots(acqGroup[item.idx.slice][item.idx.contrast], metadata, sensmaps_shots_stack)

                # Process acquisitions with PowerGrid - full recon
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
                    process_and_send(connection, acqGroup, metadata, sensmaps, shotimgs, prot_arrays, img_coord)

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
        if item is not None:
            logging.info("There was untriggered k-space data that will not get processed.")
            acqGroup = []

    finally:
        connection.send_close()

# %%
#########################
# Process Data
#########################

def process_and_send(connection, acqGroup, metadata, sensmaps, shotimgs, prot_arrays, img_coord):
    # Start data processing
    logging.info("Processing a group of k-space data")
    images = process_raw(acqGroup, metadata, sensmaps, shotimgs, prot_arrays, img_coord)
    logging.debug("Sending images to client.")
    connection.send_image(images)
    acqGroup.clear()

def process_raw(acqGroup, metadata, sensmaps, shotimgs, prot_arrays, img_coord):

    # average acquisitions before reco
    avg_before = True 
    if metadata.encoding[0].encodingLimits.contrast.maximum > 0:
        avg_before = False # do not average before reco in diffusion imaging as this could introduce phase errors

    # Write ISMRMRD file for PowerGrid
    tmp_file = dependencyFolder+"/PowerGrid_tmpfile.h5"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    dset_tmp = ismrmrd.Dataset(tmp_file, create_if_needed=True)

    # Write header
    sms_factor = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2) if metadata.encoding[0].encodingLimits.slice.maximum > 0 else 1
    if sms_factor > 1:
        metadata.encoding[0].encodedSpace.matrixSize.z = sms_factor
        metadata.encoding[0].encodingLimits.slice.maximum = int((metadata.encoding[0].encodingLimits.slice.maximum + 1) / sms_factor + 0.5) - 1
    if process_raw.slc_sel is not None:
        metadata.encoding[0].encodingLimits.slice.maximum = 0
    if process_raw.first_contrast:
        metadata.encoding[0].encodingLimits.contrast.maximum = 0
        metadata.encoding[0].encodingLimits.repetition.maximum = 0
    if avg_before:
        n_avg = metadata.encoding[0].encodingLimits.average.maximum + 1
        metadata.encoding[0].encodingLimits.average.maximum = 0
    dset_tmp.write_xml_header(metadata.toXML())

    # Insert Coordinates
    img_coord = np.asarray(img_coord) # [n_slc, 3, nx, ny, nz]
    img_coord = np.transpose(img_coord, [1,0,4,3,2]) # [3, n_slc, nz, ny, nx]
    dset_tmp.append_array("ImgCoord", img_coord)

    # Insert Sensitivity Maps
    if read_ecalib:
        sens = np.load(debugFolder + "/sensmaps.npy")
    elif process_raw.slc_sel is not None:
        sens = np.transpose(sensmaps[process_raw.slc_sel][np.newaxis], [0,4,3,2,1])
    else:
        sens = np.transpose(np.stack(sensmaps), [0,4,3,2,1]) # [slices,nc,nz,ny,nx]
        if sms_factor > 1:
            sens = reshape_sens_sms(sens, sms_factor)
    np.save(debugFolder + "/sensmaps.npy", sens)
    dset_tmp.append_array("SENSEMap", sens.astype(np.complex128))

    # Insert Field Map
    # process_acs.fmap = None # just for debugging
    if process_acs.fmap is not None:
        fmap = process_acs.fmap
    else: # external field map
        fmap_path = dependencyFolder+"/fmap.npz"
        fmap_shape = [sens.shape[0]*sens.shape[2], sens.shape[3], sens.shape[4]] # shape to check field map dims
        fmap = load_external_fmap(fmap_path, fmap_shape)

    fmap_data = fmap['fmap']
    fmap_mask = fmap['mask']
    fmap_name = fmap['name']
    if process_raw.slc_sel is not None:
        fmap_data = fmap_data[process_raw.slc_sel][np.newaxis]
        fmap_mask = fmap_mask[process_raw.slc_sel][np.newaxis]
    fmap_data = np.asarray(fmap_data)
    fmap_mask = np.asarray(fmap_mask)
    if fmap_data.ndim == 4: # remove slice dimension, if 3D dataset 
        fmap_data = fmap_data[0]
        fmap_mask = fmap_mask[0]
    np.save(debugFolder+"/fmap_data.npy", fmap_data)
    np.save(debugFolder+"/fmap_mask.npy", fmap_mask)
    if sms_factor > 1:
        fmap_data = reshape_fmap_sms(fmap_data, sms_factor) # reshape for SMS imaging

    dset_tmp.append_array('FieldMap', fmap_data) # [slices,nz,ny,nx] normally collapses to [slices/nz,ny,nx], 4 dims are only used in SMS case
    logging.debug("Field Map name: %s", fmap_name)

    # Calculate phase maps from shot images and append if necessary
    pcSENSE = False
    if shotimgs is not None:
        pcSENSE = True
        if process_raw.slc_sel is not None:
            shotimgs = np.stack(shotimgs[process_raw.slc_sel])[np.newaxis]
        else:
            shotimgs = np.stack(shotimgs) # [slice, contrast, shot, nz, ny, nx] , nz is used for SMS
        shotimgs = np.swapaxes(shotimgs, 0, 1) # to [contrast, slice, shot, nz, ny, nx] - WIP: expand to [rep, avg, contrast, slice, shot, nz, ny, nx]
        if sms_factor > 1:
            mask = reshape_fmap_sms(fmap_mask.copy(), sms_factor) # to [slice,nz,ny,nx]
        else:
            mask = fmap_mask.copy()[:,np.newaxis]
        phasemaps = calc_phasemaps(shotimgs, mask, metadata)
        dset_tmp.append_array("PhaseMaps", phasemaps)

    # Average acquisition data before reco
    # Assume that averages are acquired in the same order for every slice, contrast, ...
    if avg_before:
        avgData = [[] for _ in range(n_avg)]
        for slc in acqGroup:
            for contr in slc:
                for acq in contr:
                    avgData[acq.idx.average].append(acq.data[:])
        avgData = np.mean(avgData, axis=0)

    # Insert acquisitions
    avg_ix = 0
    bvals = []
    dirs = []
    contr_ctr = -1
    for slc in acqGroup:
        for contr in slc:
            for acq in contr:
                slc_ix = acq.idx.slice
                if avg_before:
                    if acq.idx.average == 0:
                        acq.data[:] = avgData[avg_ix]
                        avg_ix += 1
                    else:
                        continue
                if process_raw.slc_sel is not None:
                    if slc_ix != process_raw.slc_sel:
                        continue
                    else:
                        acq.idx.slice = 0
                if process_raw.first_contrast and acq.idx.repetition > 0:
                    continue
                if acq.idx.contrast > contr_ctr:
                    contr_ctr += 1
                    bvals.append(acq.user_int[0])
                    dirs.append(acq.user_float[:3])

                # get rid of k0 in 5th dim, we dont need it in PowerGrid
                # Trajectory is shaped as: kx,ky,kz,time // WIP: five 2nd order terms in the following dimensions
                save_trj = acq.traj[:,:4].copy()
                acq.resize(trajectory_dimensions=4, number_of_samples=acq.number_of_samples, active_channels=acq.active_channels)
                acq.traj[:] = save_trj.copy()
                dset_tmp.append_acquisition(acq)

    bvals = np.asarray(bvals)
    dirs = np.asarray(dirs)

    readout_dur = acq.traj[-1,3] - acq.traj[0,3]
    ts_time = int((acq.traj[-1,3] - acq.traj[0,3]) / 1e-3) # 1 time segment per ms readout
    ts_fmap = int(np.max(abs(fmap_data)) * (acq.traj[-1,3] - acq.traj[0,3]) / (np.pi/2)) # 1 time segment per pi/2 maximum phase evolution
    ts = min(ts_time, ts_fmap)
    dset_tmp.close()

    # Define in- and output for PowerGrid
    pg_dir = dependencyFolder+"/powergrid_results"
    if not os.path.exists(pg_dir):
        os.makedirs(pg_dir)
    if os.path.exists(pg_dir+"/images_pg.npy"):
        os.remove(pg_dir+"/images_pg.npy")
    n_shots = metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1

    """ PowerGrid reconstruction
    # Comment from Alex Cerjanic, who developed PowerGrid: 'histo' option can generate a bad set of interpolators in edge cases
    # He recommends using the Hanning interpolator with ~1 time segment per ms of readout (which is based on experience @3T)
    # However, histo lead to quite nice results so far & does not need as many time segments
    """
 
    temp_intp = 'hanning' # hanning / histo / minmax
    if temp_intp == 'histo' or temp_intp == 'minmax': ts = int(ts/1.5 + 0.5)
    if sms_factor > 1:
        logging.debug(f'Readout is {1e3*readout_dur} ms. Use DFT for reconstruction as SMS was used.')
    else:
        logging.debug(f'Readout is {1e3*readout_dur} ms. Use {ts} time segments.')

    # MPI and hyperthreading
    mpi = True
    hyperthreading = False # seems to slow down recon in some cases
    if hyperthreading:
        cores = psutil.cpu_count(logical = True)
        mpi_cmd = 'mpirun --use-hwthread-cpus'
    else:
        cores = psutil.cpu_count(logical = False)
        mpi_cmd = 'mpirun'

    # Source modules to use module load - module load sets correct LD_LIBRARY_PATH for MPI
    # the LD_LIBRARY_PATH is causing problems with BART though, so it has to be done here
    pre_cmd = 'source /etc/profile.d/modules.sh && module load /opt/nvidia/hpc_sdk/modulefiles/nvhpc/22.1 && '

    mps_server = False
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all' and mpi:
        # Start an MPS Server for faster MPI on GPU
        # See: https://stackoverflow.com/questions/34709749/how-do-i-use-nvidia-multi-process-service-mps-to-run-multiple-non-mpi-cuda-app
        # and https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
        mps_server = True
        try:
            subprocess.run('nvidia-cuda-mps-control -d', shell=True, check=True, text=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logging.debug("MPS Server not started. See error messages below.")
            logging.debug(e.stdout)

    # Define PowerGrid options
    higher_order = True # WIP: use trajectory size to determine if possible
    if higher_order:
        pg_opts = f'-i {tmp_file} -o {pg_dir} -B 500 -n 20 -D 2'
        subproc = pre_cmd + f'{mpi_cmd} -n {cores} PowerGridSenseMPI_ho ' + pg_opts
    else:
        pg_opts = f'-i {tmp_file} -o {pg_dir} -s {n_shots} -B 500 -n 20 -D 2' # -w option writes intermediate results as niftis in pg_dir folder
        if pcSENSE: # Multishot
            if sms_factor > 1: # use discrete Fourier transform as 3D gridding has bug
                if mpi:
                    subproc = pre_cmd + f'{mpi_cmd} -n {cores} PowerGridPcSenseMPI ' + pg_opts
                else:
                    subproc = 'PowerGridPcSense ' + pg_opts
            else: # nufft with time segmentation
                pg_opts += f' -I {temp_intp} -t {ts}'
                if mpi:
                    subproc = pre_cmd + f'{mpi_cmd} -n {cores} PowerGridPcSenseMPI_TS ' + pg_opts
                else:
                    subproc = 'PowerGridPcSenseTimeSeg ' + pg_opts
        else: # Singleshot
            pg_opts += f' -I {temp_intp} -t {ts}' # these are added also for the DFT, even though they are not used (required option in "PowerGridSenseMPI")
            if sms_factor > 1:
                pg_opts += ' -F DFT' # use discrete Fourier transform as 3D gridding has bug
            else:
                pg_opts += ' -F NUFFT' # nufft with time segmentation
            if mpi:
                subproc = pre_cmd + f'{mpi_cmd} -n {cores} PowerGridSenseMPI ' + pg_opts
            else:
                subproc = 'PowerGridIsmrmrd ' + pg_opts
    # Run in bash
    logging.debug("PowerGrid Reconstruction cmdline: %s",  subproc)
    try:
        tic = perf_counter()
        process = subprocess.run(subproc, shell=True, check=True, text=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        toc = perf_counter()
        logging.debug(f"PowerGrid Reconstruction time: {toc-tic}.")
        # logging.debug(process.stdout)
        if mps_server:
            subprocess.run('echo quit | nvidia-cuda-mps-control', shell=True) 
    except subprocess.CalledProcessError as e:
        if mps_server:
            subprocess.run('echo quit | nvidia-cuda-mps-control', shell=True) 
        logging.debug(e.stdout)
        raise RuntimeError("PowerGrid Reconstruction failed. See logfiles for errors.")

    # Image data is saved as .npy
    data = np.load(pg_dir + "/images_pg.npy")
    data = np.abs(data)

    """
    """

    # data should have output [Slice, Phase, Contrast/Echo, Avg, Rep, Nz, Ny, Nx]
    # change to [Avg, Rep, Contrast/Echo, Phase, Slice, Nz, Ny, Nx] and average
    data = np.transpose(data, [3,4,2,1,0,5,6,7]).mean(axis=0)
    # reorder sms slices
    newshape = [_ for _ in data.shape]
    newshape[3:5] = [newshape[3]*newshape[4], 1]
    data = data.reshape(newshape, order='f')

    logging.debug("Image data is size %s" % (data.shape,))
   
    images = []
    dsets = []

    # If we have a diffusion dataset, b-value and direction contrasts are stored in contrast index
    # as otherwise we run into problems with the PowerGrid acquisition tracking.
    # We now (in case of diffusion imaging) split the b=0 image from other images and reshape to b-values (contrast) and directions (phase)
    if "b_values" in prot_arrays and not process_raw.first_contrast:
        # Append data
        dsets.append(data)

        # Reshape Arrays
        shp = data.shape
        n_b0 = len(prot_arrays['b_values']) - np.count_nonzero(prot_arrays['b_values'])
        n_bval = metadata.encoding[0].encodingLimits.contrast.center # number of b-values (incl b=0)
        n_dirs = metadata.encoding[0].encodingLimits.phase.center # number of directions
        b0 = data[:,np.nonzero(bvals==0)[0]]
        diffw_imgs = data[:,np.nonzero(bvals)[0]].reshape(shp[0], n_bval-n_b0, n_dirs, shp[3], shp[4], shp[5], shp[6])

        # Calculate ADC maps
        mask = fmap_mask.copy()
        b0 = np.expand_dims(b0.mean(1), 1) # averages b=0 scans
        adc_maps = process_diffusion_images(b0, diffw_imgs, prot_arrays, mask)
        adc_maps = adc_maps[:,np.newaxis] # add empty nz dimension for correct flip
        dsets.append(adc_maps)
    else:
        dsets.append(data)
        n_b0 = 0

    # Append reference image
    np.save(debugFolder + "/refimg.npy", process_acs.refimg)
    if process_raw.slc_sel is not None:
        dsets.append(process_acs.refimg[process_raw.slc_sel][np.newaxis])
    else:
        dsets.append(np.asarray(process_acs.refimg))

    # Correct orientation, normalize and convert to int16 for online recon
    int_max = np.iinfo(np.uint16).max
    imascale = dsets[0].max() # use max of T2 weighted, should be also global max
    for k in range(len(dsets)):
        dsets[k] = np.swapaxes(dsets[k], -1, -2)
        dsets[k] = np.flip(dsets[k], (-4,-3,-2,-1))
        if dsets[k].ndim>4:
            dsets[k] *= int_max / imascale # images from PowerGrid (T2 and diff images)
        else:
            dsets[k] *= int_max / dsets[k].max() # other images (ADC maps, Refimage)
        dsets[k] = np.around(dsets[k])
        dsets[k] = dsets[k].astype(np.uint16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                        'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                        'WindowCenter':           str((int_max+1)//2),
                        'WindowWidth':            str(int_max+1),
                        'Keep_image_geometry':    '1',
                        'PG_Options':              subproc,
                        'Field Map':               fmap_name})

    series_ix = 0
    img_ix = 0
    n_slc = data.shape[3]
    slc_res = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    rotmat = rh.calc_rotmat(acqGroup[0][0][0]) if process_raw.slc_sel is None else rh.calc_rotmat(acqGroup[process_raw.slc_sel][0][0]) # rotmat is always the same
    for data_ix,data in enumerate(dsets):
        series_ix += 1
        # Format as 2D ISMRMRD image data [nx,ny]
        if data.ndim > 4:
            for contr in range(data.shape[1]):
                for phs in range(data.shape[2]):
                    for rep in range(data.shape[0]): # save one repetition after another
                        for slc in range(data.shape[3]):
                            img_ix += 1
                            for nz in range(data.shape[4]):
                                if process_raw.slc_sel is None:
                                    image = ismrmrd.Image.from_array(data[rep,contr,phs,slc,nz], acquisition=acqGroup[0][contr][0])
                                    meta['ImageRowDir'] = ["{:.18f}".format(acqGroup[0][0][0].read_dir[0]), "{:.18f}".format(acqGroup[0][0][0].read_dir[1]), "{:.18f}".format(acqGroup[0][0][0].read_dir[2])]
                                    meta['ImageColumnDir'] = ["{:.18f}".format(acqGroup[0][0][0].phase_dir[0]), "{:.18f}".format(acqGroup[0][0][0].phase_dir[1]), "{:.18f}".format(acqGroup[0][0][0].phase_dir[2])]
                                else:
                                    image = ismrmrd.Image.from_array(data[rep,contr,phs,slc,nz], acquisition=acqGroup[process_raw.slc_sel][contr][0])
                                    meta['ImageRowDir'] = ["{:.18f}".format(acqGroup[process_raw.slc_sel][0][0].read_dir[0]), "{:.18f}".format(acqGroup[process_raw.slc_sel][0][0].read_dir[1]), "{:.18f}".format(acqGroup[process_raw.slc_sel][0][0].read_dir[2])]
                                    meta['ImageColumnDir'] = ["{:.18f}".format(acqGroup[process_raw.slc_sel][0][0].phase_dir[0]), "{:.18f}".format(acqGroup[process_raw.slc_sel][0][0].phase_dir[1]), "{:.18f}".format(acqGroup[process_raw.slc_sel][0][0].phase_dir[2])]
                                image.image_index = img_ix
                                image.image_series_index = series_ix
                                image.slice = slc
                                image.repetition = rep
                                image.phase = phs
                                image.contrast = contr
                                if 'b_values' in prot_arrays:
                                    image.user_int[0] = bvals[contr]
                                if 'Directions' in prot_arrays and data_ix>0:
                                    image.user_float[:3] = dirs[contr]
                                image.attribute_string = meta.serialize()
                                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                                       ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                                       ctypes.c_float(slc_res))
                                offset = [0, 0, -1*slc_res*(slc-(n_slc-1)/2)] # slice offset in GCS
                                image.position[:] += rh.gcs_to_pcs(offset, rotmat) # correct image position in PCS
                                images.append(image)
        else:
            # ADC maps and Refimg
            series_ix += 1
            for slc, img in enumerate(data):
                if process_raw.slc_sel is None:
                    image = ismrmrd.Image.from_array(img[0], acquisition=acqGroup[0][0][0])
                else:
                    image = ismrmrd.Image.from_array(img[0], acquisition=acqGroup[process_raw.slc_sel][0][0])
                image.image_index = slc + 1
                image.image_series_index = series_ix
                image.slice = slc
                image.attribute_string = meta.serialize()
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                       ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                       ctypes.c_float(slc_res))
                offset = [0, 0, -1*slc_res*(slc-(n_slc-1)/2)]    
                image.position[:] += rh.gcs_to_pcs(offset, rotmat)
                images.append(image)

    logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(meta.serialize()).toprettyxml())
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images

def process_acs(group, metadata, dmtx=None, te_diff=None, sens_shots=False):
    """ Process reference scans for parallel imaging calibration
    """

    if len(group)==0:
        raise ValueError("Process ACS was triggered for empty acquisition group.")

    data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
    data = np.swapaxes(data,0,1) # for correct orientation in PowerGrid

    slc_ix = group[0].idx.slice

    # ESPIRiT calibration - use only first contrast
    gpu = False
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        gpu = True
    if read_ecalib:
        sensmaps = np.zeros(1)
    else:
        if gpu and data.shape[2] > 1: # only for 3D data, otherwise the overhead makes it slower than CPU
            logging.debug("Run Espirit on GPU.")
            sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I', data[...,0]) # c: crop value ~0.9, t: threshold ~0.005, r: radius (default is 24)
        else:
            logging.debug("Run Espirit on CPU.")
            sensmaps = bart(1, 'ecalib -m 1 -k 6 -I', data[...,0])

    # Field Map calculation - if acquired
    refimg = cifftn(data, [0,1,2])
    process_acs.refimg[slc_ix] = rh.rss(refimg[...,0], axis=-1).T
    if te_diff is not None and data.shape[-1] > 1:
        process_acs.fmap['fmap'][slc_ix], process_acs.fmap['mask'][slc_ix] = calc_fmap(refimg, te_diff, metadata)

    # calculate low resolution sensmaps for shot images
    if sens_shots:
        up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}
        os_region = up_double["os_region"]
        if np.allclose(os_region,0):
            os_region = 0.25 # use default if no region provided
        nx = metadata.encoding[0].encodedSpace.matrixSize.x
        data = bart(1,f'resize -c 0 {int(nx*os_region)} 1 {int(nx*os_region)}', data)
        if gpu and data.shape[2] > 1:
            sensmaps_shots = bart(1, 'ecalib -g -m 1 -k 6', data[...,0])
        else:
            sensmaps_shots = bart(1, 'ecalib -m 1 -k 6', data[...,0])
    else:
        sensmaps_shots = None

    np.save(debugFolder + "/" + "acs.npy", data)

    return sensmaps, sensmaps_shots

def calc_fmap(imgs, te_diff, metadata):
    """ Calculate field maps from reference images with two different contrasts

        imgs: [nx,ny,nz,nc,n_contr] - atm: n_contr=2 mandatory
        te_diff: TE difference [s]
    """
    
    mc_fmaps = True # calculate multi-coil field maps to remove outliers
    filtering = False # apply Gaussian and median filtering

    phasediff = imgs[...,1] * np.conj(imgs[...,0]) # phase difference
    if phasediff.shape[2] == 1:
        phasediff = phasediff[:,:,0]

    if mc_fmaps:
        fmap_shape = imgs.shape[:3]
        phasediff_uw = np.zeros_like(phasediff,dtype=np.float64)
        for k in range(phasediff.shape[-1]):
            phasediff_uw[...,k] = unwrap_phase(np.angle(phasediff[...,k]))
        nc = phasediff_uw.shape[-1]
        phasediff_uw = phasediff_uw.reshape([-1,nc])
        img_mag = abs(imgs[...,0]).reshape([-1,nc])
        fmap = np.zeros([phasediff_uw.shape[0]])
        for i in range(phasediff_uw.shape[0]):
            ix = np.argsort(phasediff_uw[i])[nc//4:-nc//4] # remove lowest & highest quartile
            weights = img_mag[i,ix] / np.sum(img_mag[i,ix])
            fmap[i] = np.sum(weights * phasediff_uw[i,ix])
        fmap = fmap.reshape(fmap_shape)
    else:
        fmap = np.sum(phasediff, axis=-1) # coil combination
        fmap = unwrap_phase(np.angle(fmap))
        
    fmap = np.atleast_3d(fmap)
    fmap = -1 * fmap/te_diff # for some reason the sign in Powergrid is different

    # mask image with median otsu from dipy
    img = rh.rss(imgs[...,0], axis=-1)
    img_masked, mask_otsu = median_otsu(img, median_radius=1, numpass=20)

    # simple threshold mask
    thresh = 0.13
    mask_thresh = img/np.max(img)
    mask_thresh[mask_thresh<thresh] = 0
    mask_thresh[mask_thresh>=thresh] = 1

    # combine masks
    mask = mask_thresh + mask_otsu
    mask[mask>0] = 1
    if mask.shape[-1] == 1:
        mask = binary_fill_holes(mask[...,0])[...,np.newaxis]
    else:
        mask = binary_fill_holes(mask)
    mask = binary_dilation(mask, iterations=2) # some extrapolation

    # apply masking and some regularization
    fmap *= mask
    if filtering:
        # WIP: standard deviation denoising - s. Paper Robinson 2009
        fmap = gaussian_filter(fmap, sigma=0.5)
        fmap = median_filter(fmap, size=2)

    # interpolate if necessary
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    newshape = [nx,ny,nz]
    fmap = resize(fmap, newshape, anti_aliasing=True)

    # if fmap is 2D, remove nz
    if nz == 1:
        fmap = fmap[...,0]
        mask = mask[...,0]

    return fmap.T, mask.T # to [nz,ny,nx]

def process_shots(group, metadata, sensmaps_shots):
    """ Reconstruct images from single shots for calculation of phase maps

    WIP: maybe use PowerGrid for B0-correction? If recon without B0 correction is sufficient, BART is more time efficient
    """

    # sort data
    data, traj = sort_spiral_data(group)

    # stack SMS dimension
    sensmaps_shots = np.stack(sensmaps_shots)

    sms_factor = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2) if metadata.encoding[0].encodingLimits.slice.maximum > 0 else 1
    if sms_factor > 1:
        sms = True
        sms_dim = 13
        data = rh.add_naxes(data, sms_dim+1-data.ndim)
        for s in range(sms_factor):
            data = np.concatenate((data, np.zeros_like(data[...,0,np.newaxis])),axis=sms_dim) # WIP: this might not work for stacked spiral
        sensmaps_shots = rh.add_naxes(sensmaps_shots, sms_dim+1-sensmaps_shots.ndim)
        sensmaps_shots = np.moveaxis(sensmaps_shots,0,sms_dim)
    else:
        sms = False
        sensmaps_shots = sensmaps_shots[0]

    # undo the swap in process_acs as BART needs different orientation  
    sensmaps = np.swapaxes(sensmaps_shots, 0, 1) 

    # Reconstruct low resolution images
    # dont use GPU as it creates a lot of overhead, which causes longer recon times
    imgs = []
    for k in range(data.shape[2]):
        traj_shot = traj[:,:,k,np.newaxis]
        data_shot = data[:,:,k,np.newaxis]
        pat = bart(1, 'pattern', data_shot) if sms else None # k-space pattern - needed for SMS recon
        if sms:
            img = bart(1, 'pics -S -e -l1 -r 0.001 -i 15 -M', data_shot, sensmaps, t=traj_shot, p=pat)
            img = np.moveaxis(img,-1,0)[...,0,0,0,0,0,0,0,0,0,0,0]
        else:
            img = bart(1, 'pics -S -e -l1 -r 0.001 -i 15', data_shot, sensmaps, t=traj_shot)
            img = img[np.newaxis]
        imgs.append(img) # shot images in list with [nz,ny,nx]
    
    return imgs

def calc_phasemaps(shotimgs, mask, metadata):
    """ Calculate phase maps for phase corrected reconstruction
        WIP: still artifacts in the images
             also it might make sense to set phasemaps to zero for b=0 images
             as no phase correction is needed
    """

    nx = metadata.encoding[0].encodedSpace.matrixSize.x

    phasemaps = np.conj(shotimgs[:,:,0,np.newaxis]) * shotimgs # 1st shot is taken as reference phase
    phasemaps = np.angle(phasemaps)

    np.save(debugFolder + "/" + "shotimgs.npy", shotimgs)
    np.save(debugFolder + "/" + "phsmaps_wrapped.npy", phasemaps)

    phasemaps = np.swapaxes(phasemaps, 1, 2) # to [contrast, shot, slice, nz, ny, nx]
    shape = [s for s in phasemaps.shape[:-2]]
    phasemaps = phasemaps.reshape([-1]+[s for s in phasemaps.shape[2:]]) # to [-1, slice, nz, ny, nx]

    # phase unwrapping, interpolation to higher resolution, filtering
    unwrapped_phasemaps = np.zeros([s for s in phasemaps.shape[:-2]] + [nx, nx])
    for i,item in enumerate(phasemaps): # contrast * shots
        for j, slc in enumerate(item): # slice
            for k, phsmap in enumerate(slc): # nz/sms-stack
                phsmap = unwrap_phase(phsmap, wrap_around=(False, False))
                phsmap = resize(phsmap, [nx,nx])
                phsmap *= mask[j,k]
                unwrapped_phasemaps[i,j,k] = median_filter(phsmap, size=9) # median filter seems to be better than Gaussian
    
    phasemaps = unwrapped_phasemaps.reshape(shape + [nx,nx]) # back to [contrast, shot, slice, nz, ny, nx]
    phasemaps = np.swapaxes(phasemaps, 1, 2) # back to [contrast, slice, shot, nz, ny, nx]

    np.save(debugFolder + "/phsmaps.npy", phasemaps)
    
    return phasemaps

def process_diffusion_images(b0, diffw_imgs, prot_arrays, mask):
    """ Calculate ADC maps from diffusion images
    """

    from scipy.stats.mstats import gmean

    b_val = prot_arrays['b_values']
    n_bval = np.count_nonzero(b_val)
    bval = b_val[np.nonzero(b_val)[0]]

    # reshape images - atm: just average repetitions and Nz is not used (no 3D imaging for diffusion)
    b0 = b0[:,0,0,:,0,:,:].mean(0) # [slices, Ny, Nx]
    imgshape = [s for s in b0.shape]
    diff = np.transpose(diffw_imgs[:,:,:,:,0].mean(0), [2,3,4,1,0]) # from [Rep, b_val, Direction, Slice, Nz, Ny, Nx] to [Slice, Ny, Nx, Direction, b_val]

    # calculate trace images (geometric mean)
    trace = gmean(diff, axis=-2)

    # calculate trace ADC map with LLS
    trace_norm = np.divide(trace.T, b0.T, out=np.zeros_like(trace.T), where=b0.T!=0).T
    trace_log  = -np.log(trace_norm, out=np.zeros_like(trace_norm), where=trace_norm!=0)

    # calculate trace diffusion coefficient - WIP: Is the fitting function working correctly?
    if n_bval<3:
        adc_map = (trace_log / bval).mean(-1)
    else:
        adc_map = np.polynomial.polynomial.polyfit(bval, trace_log.reshape([-1,n_bval]).T, 1)[1,].T.reshape(imgshape)

    adc_map *= mask

    return adc_map
    
# %%
#########################
# Sort Data
#########################

def sort_spiral_data(group):

    sig = list()
    trj = list()
    for acq in group:
        # signal
        sig.append(acq.data)

        # trajectory
        traj = np.swapaxes(acq.traj,0,1)[:3] # [dims, samples]
        trj.append(traj)
  
    # convert lists to numpy arrays
    trj = np.asarray(trj) # current size: (nacq, 3, ncol)
    sig = np.asarray(sig) # current size: (nacq, ncha, ncol)

    # rearrange trj & sig for bart
    trj = np.transpose(trj, [1, 2, 0]) # [3, ncol, nacq]
    sig = np.transpose(sig, [2, 0, 1])[np.newaxis]
   
    return sig, trj

def sort_into_kspace(group, metadata, dmtx=None, zf_around_center=False):
    # initialize k-space
    enc1_min, enc1_max = int(999), int(0)
    enc2_min, enc2_max = int(999), int(0)
    contr_max = 0
    for acq in group:
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        contr = acq.idx.contrast
        if enc1 < enc1_min:
            enc1_min = enc1
        if enc1 > enc1_max:
            enc1_max = enc1
        if enc2 < enc2_min:
            enc2_min = enc2
        if enc2 > enc2_max:
            enc2_max = enc2
        if contr > contr_max:
            contr_max = contr
            
        # Oversampling removal - WIP: assumes 2x oversampling at the moment
        data = rh.remove_os(acq.data[:], axis=-1)
        acq.resize(number_of_samples=data.shape[-1], active_channels=data.shape[0])
        acq.data[:] = data

    nc = process_acs.cc_cha
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    n_contr = contr_max + 1

    kspace = np.zeros([ny, nz, nc, nx, n_contr], dtype=group[0].data.dtype)
    counter = np.zeros([ny, nz], dtype=np.uint16)

    logging.debug("nx/ny/nz: %s/%s/%s; enc1 min/max: %s/%s; enc2 min/max:%s/%s, ncol: %s" % (nx, ny, nz, enc1_min, enc1_max, enc2_min, enc2_max, group[0].data.shape[-1]))

    for acq in group:

        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        contr = acq.idx.contrast

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
        
        # Noise-whitening
        if dmtx is not None:
            acq.data[:] = rh.apply_prewhitening(acq.data, dmtx)
        
        # Apply coil compression
        if process_acs.cc_mat[acq.idx.slice] is not None:
            rh.apply_cc(acq, process_acs.cc_mat[acq.idx.slice])

        kspace[enc1, enc2, :, col, contr] += acq.data   
        
        if contr==0:
            counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc, n_contr)
    kspace = np.transpose(kspace, [3, 0, 1, 2, 4])

    return kspace

def reshape_sens_sms(sens, sms_factor):
    # reshape sensmaps array for sms imaging, sensmaps for one acquisition are stored at nz
    sens_cpy = sens.copy() # [slices, coils, nz, ny, nx]
    slices_eff = sens_cpy.shape[0]//sms_factor
    sens = np.zeros([slices_eff, sens_cpy.shape[1], sms_factor, sens_cpy.shape[3], sens_cpy.shape[4]], dtype=sens_cpy.dtype)
    for slc in range(sens_cpy.shape[0]):
        sens[slc%slices_eff,:,slc//slices_eff] = sens_cpy[slc,:,0] 
    return sens

def reshape_fmap_sms(fmap, sms_factor):
    # reshape field map array for sms imaging
    fmap_cpy = fmap.copy()
    slices_eff = fmap_cpy.shape[0]//sms_factor
    fmap = np.zeros([slices_eff, sms_factor, fmap_cpy.shape[1], fmap_cpy.shape[2]], dtype=fmap_cpy.dtype) # [slices, ny, nx] to [slices, nz, ny, nx]
    for slc in range(fmap_cpy.shape[0]):
        fmap[slc%slices_eff, slc//slices_eff] = fmap_cpy[slc] 
    return fmap

def load_external_fmap(path, shape):
    # Load an external field map (has to be a .npz file)
    if not os.path.exists(path):
        fmap = {'fmap': np.zeros(shape), 'mask': np.ones(shape), 'name': 'No Field Map'}
        logging.debug("No field map file in dependency folder. Use zeros array instead. Field map should be .npz file.")
    else:
        fmap = np.load(path, allow_pickle=True)
        if 'name' not in fmap:
            fmap['name'] = 'No name.'
    if shape != list(fmap['fmap'].shape):
        logging.debug(f"Field Map dimensions do not fit. Fmap shape: {list(fmap['fmap'].shape)}, Img Shape: {shape}. Dont use field map in recon.")
        fmap = {'fmap': np.zeros(shape), 'mask': np.ones(shape), 'name': 'No Field Map'}
    if 'params' in fmap:
        logging.debug("Field Map regularisation parameters: %s",  fmap['params'].item())

    return fmap
