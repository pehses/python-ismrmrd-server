
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

from scipy.ndimage import  median_filter
from skimage.transform import resize
from skimage.restoration import unwrap_phase

from pulseq_helper import insert_hdr, insert_acq, get_ismrmrd_arrays, read_acqs
import reco_helper as rh

""" Reconstruction of imaging data acquired with the Pulseq Sequence via the FIRE framework
    Reconstruction is done with the BART toolbox and the PowerGrid toolbox

    The B0 field corrected recon is done in the logical coordinate system (GCS)
    Trajectories therefore have to be int the logical coordinate system and scaled with "FOV[m] / (2*pi)" to make them dimensionless and suitable for BART/PowerGrid.

    The 0th order term k0 will be directly applied in this script and afterwards replaced by a time vector (for B0 correction), which is also calculated in this script                    
    
    If multishot spirals were used, a phase-corrected reconstruction is done.
"""

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

# tempfile.tempdir = "/dev/shm"  # slightly faster bart wrapper

read_ecalib = False
save_cmplx = True # save images as complex data
online_recon = False

########################
# Main Function
########################

def process(connection, config, metadata, prot_file):
    
    # -- Some manual parameters --- #

    # if >0 only the volumes up to the specified number will be reconstructed
    process_raw.reco_n_contr = 0
    up_long = {item.name: item.value for item in metadata.userParameters.userParameterLong}
    if 'recon_slice' in up_long:
        logging.debug(f"Dataset is processed online. Only first contrast is reconstructed.")
        # parameter recon_slice defined in "IsmrmrdParameterMap_Siemens_pulseq_online.xsl"
        n_vol = up_long['recon_slice'] # number of volumes to be reconstructed
        process_raw.reco_n_contr = n_vol if n_vol > 0 else 0 # reconstruct n contrasts, if data is processed online
        global save_cmplx
        save_cmplx = False
        global online_recon
        online_recon = True

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
        up_string = {item.name: item.value for item in metadata.userParameters.userParameterString}
        seq_signature = up_string['seq_signature']
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

    # Get additional arrays from protocol file for diffusion imaging
    prot_arrays = get_ismrmrd_arrays(prot_file)

    # parameters for reapplying FOV shift
    nsegments = metadata.encoding[0].encodingLimits.segment.maximum + 1
    matr_sz = np.array([metadata.encoding[0].encodedSpace.matrixSize.x, metadata.encoding[0].encodedSpace.matrixSize.y, metadata.encoding[0].encodedSpace.matrixSize.z])
    res = np.array([metadata.encoding[0].encodedSpace.fieldOfView_mm.x / matr_sz[0], metadata.encoding[0].encodedSpace.fieldOfView_mm.y / matr_sz[1], metadata.encoding[0].encodedSpace.fieldOfView_mm.z / matr_sz[2]])

    # parameters for B0 correction
    dwelltime = 1e-6*up_double["dwellTime_us"] # [s]
    t_min = up_double["t_min"] # [s]
    spiral_delay =  up_double["traj_delay"] # [s]
    trajtype = metadata.encoding[0].trajectory.value
    if spiral_delay > 0 and trajtype=='spiral':
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
    n_contr = process_raw.reco_n_contr if process_raw.reco_n_contr else metadata.encoding[0].encodingLimits.contrast.maximum + 1
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
        shotimgs = [[[] for _ in range(n_contr)] for _ in range(n_slc//sms_factor)]
        sens_shots = True

    # field map, if it was acquired - needs at least 2 reference contrasts
    if 'echo_times' in prot_arrays:
        echo_times = prot_arrays['echo_times']
        process_acs.fmap = {'fmap': [None] * n_slc, 'mask': None, 'TE': echo_times, 'name': 'Field Map from reference scan'}
    else:
        process_acs.fmap = None

    # read protocol acquisitions - faster than using read_acquisition
    logging.debug("Reading in protocol acquisitions.")
    acqs = read_acqs(prot_file)

    try:
        for item in connection:

            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                if item.traj.size > 0 and not skope:
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
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):
                    continue

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
                        sensmaps[slc_ix], sensmaps_shots[slc_ix] = process_acs(acsGroup[slc_ix], metadata, dmtx, sens_shots) # [nx,ny,nz,nc]
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
                                

                # trigger recon early
                if process_raw.reco_n_contr and item.idx.contrast > process_raw.reco_n_contr - 1:
                    if len(acqGroup) > 0:
                        process_and_send(connection, acqGroup, metadata, sensmaps, shotimgs, prot_arrays)
                    continue

                # Process imaging scans - deal with ADC segments 
                # (not needed in newer spiral sequence versions, but kept for compatibility also with other scanners)
                if item.idx.segment == 0:
                    nsamples = item.number_of_samples
                    t_vec = t_min + dwelltime * np.arange(nsamples) # time vector for B0 correction
                    acqGroup[item.idx.slice][item.idx.contrast].append(item)
                else:
                    # append data to first segment of ADC group
                    last_idx = acqGroup[item.idx.slice][item.idx.contrast][-1].number_of_samples - 1
                    idx_lower = last_idx - (nsegments-item.idx.segment) * item.number_of_samples
                    idx_upper = last_idx - (nsegments-item.idx.segment-1) * item.number_of_samples
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
                    traj = last_item.traj[:,:3]
                    last_item.data[:] = rh.fov_shift_spiral_reapply(last_item.data[:], traj, base_trj, shift_px, matr_sz)

                    # filter signal to avoid Gibbs Ringing
                    traj = np.swapaxes(last_item.traj[:,:3],0,1) # traj to [dim, samples]
                    last_item.data[:] = rh.filt_ksp(last_item.data[:], traj, filt_fac=0.95)

                    # Correct the global phase
                    if skope:
                        k0 = last_item.traj[:,3]
                        last_item.data[:] *= np.exp(-1j*k0)

                    # replace k0 with time vector
                    last_item.traj[:,3] = t_vec.copy()

                    # T2* filter
                    if freq < 2e8:
                        t2_star = 70e-3 # 3T
                    else:
                        t2_star = 40e-3 # 7T
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
                    process_and_send(connection, acqGroup, metadata, sensmaps, shotimgs, prot_arrays)

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

def process_and_send(connection, acqGroup, metadata, sensmaps, shotimgs, prot_arrays):
    # Start data processing
    logging.info("Processing a group of k-space data")
    images = process_raw(acqGroup, metadata, sensmaps, shotimgs, prot_arrays)
    logging.debug("Sending images to client.")
    connection.send_image(images)
    acqGroup.clear()

def process_raw(acqGroup, metadata, sensmaps, shotimgs, prot_arrays):

    # Multiband factor
    sms_factor = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2) if metadata.encoding[0].encodingLimits.slice.maximum > 0 else 1

    # average acquisitions before reco
    avg_before = True 
    if metadata.encoding[0].encodingLimits.contrast.maximum > 0:
        avg_before = False # do not average before reco in diffusion imaging as this could introduce phase errors

    # Write ISMRMRD file for PowerGrid
    tmp_file = dependencyFolder+"/PowerGrid_tmpfile.h5"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    dset_tmp = ismrmrd.Dataset(tmp_file, create_if_needed=True)

    # Insert Sensitivity Maps
    if read_ecalib:
        sens = np.load(debugFolder + "/sensmaps.npy")
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
        refimgs = np.asarray(fmap['fmap'])
        echo_times = fmap['TE']
        fmap['fmap'], fmap['mask'] = rh.calc_fmap(refimgs, echo_times, metadata)
    else: # external field map
        fmap_path = dependencyFolder+"/fmap.npz"
        fmap_shape = [sens.shape[0]*sens.shape[2], sens.shape[3], sens.shape[4]] # shape to check for correct dimensions
        fmap = rh.load_external_fmap(fmap_path, fmap_shape)

    fmap_data = fmap['fmap']
    fmap_mask = fmap['mask']
    fmap_name = fmap['name']
    np.save(debugFolder+"/fmap_data.npy", fmap_data)
    np.save(debugFolder+"/fmap_mask.npy", fmap_mask)
    if sms_factor > 1:
        fmap_data = reshape_fmap_sms(fmap_data, sms_factor) # reshape for SMS imaging

    dset_tmp.append_array('FieldMap', fmap_data.astype(np.float64)) # [slices,nz,ny,nx] normally collapses to [slices/nz,ny,nx], 4 dims are only used in SMS case
    logging.debug("Field Map name: %s", fmap_name)

    # Calculate phase maps from shot images and append if necessary
    pcSENSE = False
    if shotimgs is not None:
        pcSENSE = True
        shotimgs = np.stack(shotimgs) # [slice, contrast, shot, nz, ny, nx] , nz is used for SMS
        shotimgs = np.swapaxes(shotimgs, 0, 1) # to [contrast, slice, shot, nz, ny, nx] - WIP: expand to [rep, avg, contrast, slice, shot, nz, ny, nx]
        if sms_factor > 1:
            mask = reshape_fmap_sms(fmap_mask.copy(), sms_factor) # to [slice,nz,ny,nx]
        else:
            mask = fmap_mask.copy()[:,np.newaxis]
        phasemaps = calc_phasemaps(shotimgs, mask, metadata)
        dset_tmp.append_array("PhaseMaps", phasemaps.astype(np.float64))

    # Write header
    if sms_factor > 1:
        metadata.encoding[0].encodedSpace.matrixSize.z = sms_factor
        metadata.encoding[0].encodingLimits.slice.maximum = int((metadata.encoding[0].encodingLimits.slice.maximum + 1) / sms_factor + 0.5) - 1
    if process_raw.reco_n_contr:
        metadata.encoding[0].encodingLimits.contrast.maximum = process_raw.reco_n_contr - 1
        metadata.encoding[0].encodingLimits.repetition.maximum = 0
    if avg_before:
        n_avg = metadata.encoding[0].encodingLimits.average.maximum + 1
        metadata.encoding[0].encodingLimits.average.maximum = 0
    dset_tmp.write_xml_header(metadata.toXML())

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
                if avg_before:
                    if acq.idx.average == 0:
                        acq.data[:] = avgData[avg_ix]
                        avg_ix += 1
                    else:
                        continue
                if process_raw.reco_n_contr and acq.idx.repetition > 0:
                    continue
                if acq.idx.contrast > contr_ctr:
                    contr_ctr += 1
                    bvals.append(acq.user_int[0])
                    dirs.append(acq.user_float[:3])

                dset_tmp.append_acquisition(acq)

    bvals = np.asarray(bvals)
    dirs = np.asarray(dirs)

    readout_dur = acq.traj[-1,3] - acq.traj[0,3]
    ts_time = int((acq.traj[-1,3] - acq.traj[0,3]) / 1e-3 + 0.5) # 1 time segment per ms readout
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
        logging.debug(f'Readout is {1e3*readout_dur} ms. DFT reconstruction as this is a multiband acquisition.')
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
        # On some GPUs, the reconstruction seems to fail with an MPS server activated. In this case, comment out this "if"-block.
        # See: https://stackoverflow.com/questions/34709749/how-do-i-use-nvidia-multi-process-service-mps-to-run-multiple-non-mpi-cuda-app
        # and https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
        mps_server = True
        try:
            subprocess.run('nvidia-cuda-mps-control -d', shell=True, check=True, text=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logging.debug("MPS Server not started. See error messages below.")
            logging.debug(e.stdout)

    # Define PowerGrid options
    pg_opts = f'-i {tmp_file} -o {pg_dir} -s {n_shots} -n 20 -B 500 -D 2' # -w option writes intermediate results as niftis in pg_dir folder
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
    if not save_cmplx:
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
    dsets.append(data.copy())

    # If we have a diffusion dataset, b-value and direction contrasts are stored in contrast index
    # as otherwise we run into problems with the PowerGrid acquisition tracking.
    # We now (in case of diffusion imaging) split the b=0 image from other images and reshape to b-values (contrast) and directions (phase)
    if "b_values" in prot_arrays and not process_raw.reco_n_contr:
        try:
            data_eval = abs(data)

            # Calculate ADC maps
            mask = fmap_mask.copy()
            adc_maps = process_diffusion_images(data_eval, bvals, mask)
            adc_maps = adc_maps[:,np.newaxis] # add empty nz dimension for correct flip

            # Append data
            dsets.append(adc_maps)
        except:
            logging.debug("ADC map calculation failed.")

    # Append reference image & field map
    np.save(debugFolder + "/refimg.npy", process_acs.refimg)
    dsets.append(np.asarray(process_acs.refimg))
    dsets.append(fmap["fmap"][:,np.newaxis] /2/np.pi) # add axis for [slc,z,y,x], save in [Hz]

    # Correct orientation, normalize and convert to int16 for online recon
    uint_max = np.iinfo(np.uint16).max
    for k in range(len(dsets)):
        dsets[k] = np.swapaxes(dsets[k], -1, -2)
        dsets[k] = np.flip(dsets[k], (-4,-3,-2,-1))
        if k==0 and save_cmplx:
            dsets[k] /= abs(dsets[k]).max()
        elif k<len(dsets)-1:
            dsets[k] *= uint_max / abs(dsets[k]).max()
            dsets[k] = np.around(dsets[k])
            dsets[k] = dsets[k].astype(np.uint16)
        else:
            dsets[k] = np.around(dsets[k])
            dsets[k] = dsets[k].astype(np.int16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                        'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                        'WindowCenter':           str((uint_max+1)//2),
                        'WindowWidth':            str(uint_max+1),
                        'Keep_image_geometry':    1,
                        'PG_Options':              subproc,
                        'Field Map':               fmap_name})
    # Set ISMRMRD Meta Attributes
    meta2 = ismrmrd.Meta({'DataRole':              'Quantitative',
                        'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                        'WindowCenter':           '0',
                        'WindowWidth':            '8192',
                        'Keep_image_geometry':    1,
                        'PG_Options':              subproc,
                        'Field Map':               fmap_name})

    img_ix = 0
    n_slc = data.shape[3]
    slc_res = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    rotmat = rh.calc_rotmat(acqGroup[0][0][0])
    for series_ix, imgs in enumerate(dsets):
        # Format as 2D ISMRMRD image data [nx,ny]
        if imgs.ndim > 4:
            for contr in range(imgs.shape[1]):
                for phs in range(imgs.shape[2]):
                    for rep in range(imgs.shape[0]): # save one repetition after another
                        for slc in range(imgs.shape[3]):
                            img_ix += 1
                            for nz in range(imgs.shape[4]):
                                image = ismrmrd.Image.from_array(imgs[rep,contr,phs,slc,nz], acquisition=acqGroup[0][contr][0])
                                meta['ImageRowDir'] = ["{:.18f}".format(acqGroup[0][0][0].read_dir[0]), "{:.18f}".format(acqGroup[0][0][0].read_dir[1]), "{:.18f}".format(acqGroup[0][0][0].read_dir[2])]
                                meta['ImageColumnDir'] = ["{:.18f}".format(acqGroup[0][0][0].phase_dir[0]), "{:.18f}".format(acqGroup[0][0][0].phase_dir[1]), "{:.18f}".format(acqGroup[0][0][0].phase_dir[2])]
                                image.image_index = img_ix
                                image.image_series_index = series_ix
                                image.slice = slc
                                image.repetition = rep
                                image.phase = phs
                                image.contrast = contr
                                image.user_int[1] = sms_factor
                                # b-values and directions should already be correct, as they are in the acquisition header
                                # but we set them here explicitly again
                                if 'b_values' in prot_arrays:
                                    image.user_int[0] = bvals[contr]
                                if 'Directions' in prot_arrays:
                                    image.user_float[:3] = dirs[contr]
                                image.attribute_string = meta.serialize()
                                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                                       ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                                       ctypes.c_float(slc_res))
                                offset = [0, 0, -1*slc_res*(slc-(n_slc-1)/2)] # slice offset in GCS
                                image.position[:] += rh.gcs_to_pcs(offset, rotmat) # correct image position in PCS
                                images.append(image)
        else:
            # ADC maps, Refimg & Fmap
            for slc, img in enumerate(imgs):
                image = ismrmrd.Image.from_array(img[0], acquisition=acqGroup[0][0][0])
                image.image_index = slc + 1
                image.image_series_index = series_ix
                image.slice = slc
                if series_ix != len(dsets) - 1:
                    image.attribute_string = meta.serialize()
                else:
                    image.attribute_string = meta2.serialize()
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                       ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                       ctypes.c_float(slc_res))
                offset = [0, 0, -1*slc_res*(slc-(n_slc-1)/2)]    
                image.position[:] += rh.gcs_to_pcs(offset, rotmat)
                images.append(image)

    logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(meta.serialize()).toprettyxml())
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images

def process_acs(group, metadata, dmtx=None, sens_shots=False):
    """ Process reference scans for parallel imaging calibration
    """

    if len(group)==0:
        raise ValueError("Process ACS was triggered for empty acquisition group.")

    data = sort_into_kspace(group, metadata, dmtx)
    data = np.swapaxes(data,0,1) # for correct orientation in PowerGrid

    # cut matrix to correct size for sensitivity maps
    # for field maps, we use the whole size, but interpolate in the end of calc_fmap
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    data_sens = bart(1,f'resize -c 0 {nx} 1 {ny} 2 {nz}', data)
    data_sens = data_sens.reshape([nx,ny,nz,data.shape[-2],data.shape[-1]]) # if number of contrasts is 1, BART will remove the last dimension

    slc_ix = group[0].idx.slice

    # ESPIRiT calibration - use only first contrast
    gpu = False
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        gpu = True
    if read_ecalib:
        sensmaps = np.zeros(1)
    else:
        logging.debug(f"Sensmap calibration for slice {slc_ix}.")
        if online_recon:
            sensmaps = bart(1, 'caldir 40', data_sens[...,0])
        elif gpu and data_sens.shape[2] > 1: # only for 3D data, otherwise the overhead makes it slower than CPU
            logging.debug("Run Espirit on GPU.")
            sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I -c 0.92 -t 0.003', data_sens[...,0]) # c: crop value ~0.9, t: threshold ~0.005, r: radius (default is 24)
        else:
            logging.debug("Run Espirit on CPU.")
            sensmaps = bart(1, 'ecalib -m 1 -k 6 -I -c 0.92 -t 0.003', data_sens[...,0])

    # Save reference data for masking and field mapping
    process_acs.refimg[slc_ix] = rh.rss(cifftn(data_sens[...,0], [0,1,2]), axis=-1).T # save at spiral matrix size
    if data.shape[-1] > 1 and process_acs.fmap is not None:
        process_acs.fmap['fmap'][slc_ix] = cifftn(data, [0,1,2]) # save at refscan matrix size (will get interpolated in fmap calculation)

    # calculate low resolution sensmaps for shot images
    if sens_shots:
        up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}
        os_region = up_double["os_region"]
        if np.allclose(os_region,0):
            os_region = 0.25 # use default if no region provided
        nx = metadata.encoding[0].encodedSpace.matrixSize.x
        data_sens = bart(1,f'resize -c 0 {int(nx*os_region)} 1 {int(nx*os_region)}', data_sens)
        if gpu and data_sens.shape[2] > 1:
            sensmaps_shots = bart(1, 'ecalib -g -m 1 -k 6', data_sens[...,0])
        else:
            sensmaps_shots = bart(1, 'ecalib -m 1 -k 6', data_sens[...,0])
    else:
        sensmaps_shots = None

    np.save(debugFolder + "/" + "acs.npy", data)

    return sensmaps, sensmaps_shots

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
            data = np.concatenate((data, np.zeros_like(data[...,0,np.newaxis])),axis=sms_dim)
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

def process_diffusion_images(data, bvals, mask):
    """ Calculate ADC maps from diffusion images
    """

    from scipy.stats.mstats import gmean

    data = data[0,:,0,:,0] # [contrast, slice, ny, nx]
    b0 = data[np.where(bvals==0)] # b0 images
    b0 = b0.mean(0)

    trace_log = []
    b_val_nz = np.unique(bvals[np.nonzero(bvals)]) # nonzero b-values
    for bval in b_val_nz:
        diffimgs = data[np.where(bvals==bval)]

        # calculate trace images (geometric mean)
        trace = gmean(diffimgs, axis=0)

        # calculate trace log
        trace_norm = np.divide(trace.T, b0.T, out=np.zeros_like(trace.T), where=b0.T!=0).T
        trace_log.append(-np.log(trace_norm, out=np.zeros_like(trace_norm), where=trace_norm!=0))

    trace_log = np.asarray(trace_log) # [n_bval, slice, ny, nx]

    if len(b_val_nz) < 3:
        adc_map = (trace_log.T / b_val_nz).T.mean(0)
    else:
        imgshape = b0.shape
        adc_map = np.polynomial.polynomial.polyfit(b_val_nz, trace_log.reshape([trace_log.shape[0],-1]), 1)[1]
        adc_map = adc_map.reshape(imgshape)

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

def sort_into_kspace(group, metadata, dmtx=None):
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
    # if reference scan has bigger matrix than spiral scan (e.g. because of higher resolution), use the bigger matrix
    nx = max(metadata.encoding[0].encodedSpace.matrixSize.x, acq.number_of_samples)
    ny = max(metadata.encoding[0].encodedSpace.matrixSize.y, enc1_max+1)
    nz = max(metadata.encoding[0].encodedSpace.matrixSize.z, enc2_max+1)
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

        # sort data into center k-space (assuming a symmetric acquisition)
        cy = ny // 2
        cz = nz // 2
        cenc1 = (enc1_max+1) // 2
        cenc2 = (enc2_max+1) // 2
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
