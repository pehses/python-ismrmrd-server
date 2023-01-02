
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

from scipy.ndimage import  median_filter, gaussian_filter, binary_fill_holes, binary_dilation
from skimage.transform import resize
from skimage.restoration import unwrap_phase
from dipy.segment.mask import median_otsu

from pulseq_helper import insert_hdr, insert_acq, get_ismrmrd_arrays, read_acqs
import reco_helper as rh

""" Higher order reconstruction of imaging data acquired with the Pulseq Sequence via the FIRE framework
    Reconstruction is done with the BART toolbox and the PowerGrid toolbox

    The higher order reconstruction is done in the physical coordinate system (DCS) and currently only possible, when Skope data is already inserted.
    The k-space coefficients from the Skope system have to stay in the physical coordinate system without rescaling, when they are inserted into the MRD file. 
    Units are: 1st order: [rad/m], 2nd order: [rad/m^2], ...

    The MRD trajectory field should have the following trajectory dimensions
    0: kx, 1: ky, 2: kz, 3: k0, 4-8: 2nd order, 9-15: 3rd order, 16-19: concomitant fields

    If 2nd or 3rd order was not measured, the concomitant field terms move to lower dimensions (no zero-filling), to reduce the required space

    The 0th order term k0 will be directly applied in this script and afterwards replaced by a time vector (for B0 correction), which is also calculated in this script                    
    
    !!! This recon does not support multishot diffusion imaging !!!

"""

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

# tempfile.tempdir = "/dev/shm"  # slightly faster bart wrapper

read_ecalib = False
save_cmplx = True # save images as complex data

########################
# Main Function
########################

def process(connection, config, metadata, prot_file):
    
    # -- Some manual parameters --- #

    # if >0 only the volumes up to the specified number will be reconstructed
    process_raw.reco_n_contr = 0

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

    # Get additional arrays from protocol file for diffusion imaging
    prot_arrays = get_ismrmrd_arrays(prot_file)

    # number of ADC segments
    nsegments = metadata.encoding[0].encodingLimits.segment.maximum + 1

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
    half_refscan = True if metadata.encoding[0].encodingLimits.segment.center else False # segment center is misused as indicator for halved number of refscan slices
    if half_refscan and n_slc%2:
        raise ValueError("Odd number of slices with halved number of refscan slices is invalid.")

    acqGroup = [[[] for _ in range(n_contr)] for _ in range(n_slc//sms_factor)]
    noiseGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * n_slc
    img_coord = [None] * (n_slc//sms_factor)
    dmtx = None
    base_trj = None
    process_acs.cc_mat = [None] * n_slc # compression matrix
    process_acs.refimg = [None] * n_slc # save one reference image for DTI analysis (BET masking) 

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

                # insert acquisition protocol
                # base_trj is used to correct FOV shift (see below)
                base_traj = insert_acq(acqs[0], item, metadata, noncartesian=True, return_basetrj=True, traj_phys=True)
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
                        sensmaps[slc_ix] = process_acs(acsGroup[slc_ix], metadata, dmtx) # [nx,ny,nz,nc]
                        acsGroup[slc_ix].clear()
                        # copy data
                        if half_refscan:
                            if slc_ix%2==0:
                                sensmaps[slc_ix+1] = sensmaps[slc_ix]
                                process_acs.refimg[slc_ix+1] = process_acs.refimg[slc_ix]
                                if process_acs.fmap is not None:
                                    process_acs.fmap['fmap'][slc_ix+1] = process_acs.fmap['fmap'][slc_ix]
                                    process_acs.fmap['mask'][slc_ix+1] = process_acs.fmap['mask'][slc_ix]
                            if slc_ix%2==1:
                                sensmaps[slc_ix-1] = sensmaps[slc_ix]
                                process_acs.refimg[slc_ix-1] = process_acs.refimg[slc_ix]
                                if process_acs.fmap is not None:
                                    process_acs.fmap['fmap'][slc_ix-1] = process_acs.fmap['fmap'][slc_ix]
                                    process_acs.fmap['mask'][slc_ix-1] = process_acs.fmap['mask'][slc_ix]
                                

                # trigger recon early
                if process_raw.reco_n_contr and item.idx.contrast > process_raw.reco_n_contr - 1:
                    if len(acqGroup) > 0:
                        process_and_send(connection, acqGroup, metadata, sensmaps, prot_arrays, img_coord)
                    continue

                # Process imaging scans - deal with ADC segments 
                # (not needed in newer spiral sequence versions, but kept for compatibility also with other scanners)
                if item.idx.segment == 0:
                    nsamples = item.number_of_samples
                    t_vec = t_min + dwelltime * np.arange(nsamples) # time vector for B0 correction
                    acqGroup[item.idx.slice][item.idx.contrast].append(item)
                    img_coord[item.idx.slice] = rh.calc_img_coord(metadata, item) # image coordinates in DCS
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

                    # Undo FOV Shift with base trajectory
                    shift = rh.pcs_to_dcs(np.asarray(last_item.position)) * 1e-3 # shift [m] in DCS
                    last_item.data[:] *= np.exp(1j*(shift*base_trj).sum(axis=-1)) # base_trj in [rad/m]

                    # filter signal to avoid Gibbs Ringing
                    traj = np.swapaxes(last_item.traj[:,:3],0,1) # traj to [dim, samples]
                    last_item.data[:] = rh.filt_ksp(last_item.data[:], traj, filt_fac=0.95)

                    # Correct the global phase
                    k0 = last_item.traj[:,3]
                    last_item.data[:] *= np.exp(-1j*k0)

                    # invert trajectory sign (is necessary as field map and k0 also need sign change)
                    # WIP: for some unknown reason, not inverting the concomitant field terms yields better results
                    if last_item.traj.shape[1] > 4:
                        last_item.traj[:,:-4] *= -1
                    else:
                        last_item.traj *= -1 # no concomitant fields

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
                    
                # Process acquisitions with PowerGrid - full recon
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
                    process_and_send(connection, acqGroup, metadata, sensmaps, prot_arrays, img_coord)

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

def process_and_send(connection, acqGroup, metadata, sensmaps, prot_arrays, img_coord):
    # Start data processing
    logging.info("Processing a group of k-space data")
    images = process_raw(acqGroup, metadata, sensmaps, prot_arrays, img_coord)
    logging.debug("Sending images to client.")
    connection.send_image(images)
    acqGroup.clear()

def process_raw(acqGroup, metadata, sensmaps, prot_arrays, img_coord):

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

    # Insert Coordinates
    img_coord = np.asarray(img_coord) # [n_slc, 3, nx, ny, nz]
    img_coord = np.transpose(img_coord, [1,0,4,3,2]) # [3, n_slc, nz, ny, nx]
    dset_tmp.append_array("ImgCoord", img_coord.astype(np.float64))

    # Calculate and insert DCF
    traj = acqGroup[0][0][0].traj[:,:2]
    dcf = rh.calc_dcf(traj)
    dcf /= np.max(dcf)
    dcf2 = np.tile(dcf, acqGroup[0][0][0].active_channels)
    dset_tmp.append_array("DCF", dcf2.astype(np.float64))

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
        fmap['fmap'], fmap['mask'] = calc_fmap(refimgs, echo_times, metadata)
    else: # external field map
        fmap_path = dependencyFolder+"/fmap.npz"
        fmap_shape = [sens.shape[0]*sens.shape[2], sens.shape[3], sens.shape[4]] # shape to check for correct dimensions
        fmap = load_external_fmap(fmap_path, fmap_shape)

    fmap_data = fmap['fmap']
    fmap_mask = fmap['mask']
    fmap_name = fmap['name']
    np.save(debugFolder+"/fmap_data.npy", fmap_data)
    np.save(debugFolder+"/fmap_mask.npy", fmap_mask)
    if sms_factor > 1:
        fmap_data = reshape_fmap_sms(fmap_data, sms_factor) # reshape for SMS imaging

    dset_tmp.append_array('FieldMap', fmap_data.astype(np.float64)) # [slices,nz,ny,nx] normally collapses to [slices/nz,ny,nx], 4 dims are only used in SMS case
    logging.debug("Field Map name: %s", fmap_name)

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
    pg_opts = f'-i {tmp_file} -o {pg_dir} -n 20 -B 500 -D 2 -e 0.0005'
    subproc = pre_cmd + f'{mpi_cmd} -n {cores} PowerGridSenseMPI_ho ' + pg_opts
    
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
                        'Keep_image_geometry':    '1',
                        'PG_Options':              subproc,
                        'Field Map':               fmap_name})
    # Set ISMRMRD Meta Attributes
    meta2 = ismrmrd.Meta({'DataRole':              'Quantitative',
                        'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                        'WindowCenter':           '0',
                        'WindowWidth':            '8192',
                        'Keep_image_geometry':    '1',
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

def process_acs(group, metadata, dmtx=None):
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

    slc_ix = group[0].idx.slice

    # ESPIRiT calibration - use only first contrast
    gpu = False
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        gpu = True
    if read_ecalib:
        sensmaps = np.zeros(1)
    else:
        logging.debug(f"Sensmap calibration for slice {slc_ix}.")
        if gpu and data_sens.shape[2] > 1: # only for 3D data, otherwise the overhead makes it slower than CPU
            logging.debug("Run Espirit on GPU.")
            sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I -c 0.92 -t 0.003', data_sens[...,0]) # c: crop value ~0.9, t: threshold ~0.005, r: radius (default is 24)
        else:
            logging.debug("Run Espirit on CPU.")
            sensmaps = bart(1, 'ecalib -m 1 -k 6 -I -c 0.92 -t 0.003', data_sens[...,0])

    # Save reference data for masking and field mapping
    process_acs.refimg[slc_ix] = rh.rss(cifftn(data_sens[...,0], [0,1,2]), axis=-1).T # save at spiral matrix size
    if data.shape[-1] > 1 and process_acs.fmap is not None:
        process_acs.fmap['fmap'][slc_ix] = cifftn(data, [0,1,2]) # save at refscan matrix size (will get interpolated in fmap calculation)

    np.save(debugFolder + "/" + "acs.npy", data)

    return sensmaps

def calc_fmap(imgs, echo_times, metadata):
    """ Calculate field maps from reference images with two different contrasts

        imgs: [slices,nx,ny,nz,nc,n_contr] - atm: n_contr=2 mandatory
        echo_times: list of echo times [s]
    """
    
    mc_fmap = True # calculate multi-coil field maps to remove outliers (Robinson, MRM. 2011) - recommended
    std_filter = True # apply standard deviation filter (only if mc_fmap selected) - recommended
    std_fac = 1.1 # factor for standard deviation denoising (see below)
    romeo_fmap = False # use the ROMEO toolbox for field map calculation
    romeo_uw = False # use ROMEO only for unwrapping (slower than unwrapping with skimage)
    filtering = False # apply Gaussian and median filtering (not recommended for mc_fmap)

    if len(echo_times) > 2:
        romeo_fmap = True # more than one echo time only possible with ROMEO

    if romeo_fmap or romeo_uw:
        result_path = dependencyFolder+"/romeo_results"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    n_slc = imgs.shape[0]

    # from here on either [slices,nx,ny,coils,echoes] or [nz,nx,ny,coils,echoes]
    if nz == 1:
        imgs = imgs[:,:,:,0] # 2D field map acquisition
    elif nz > 1:
        imgs = np.moveaxis(imgs[0],2,0) # 3D field map acquisition
    elif nz > 1 and n_slc > 1:
        raise ValueError("Multi-slab is not supported.")

    if romeo_fmap:
        # ROMEO unwrapping and field map calculation (Dymerska, MRM, 2020)
        fmap = rh.romeo_unwrap(imgs, echo_times, result_path, mc_unwrap=False, return_b0=True)
    elif mc_fmap:
        # Multi-coil field map calculation (Robinson, MRM, 2011)
        phasediff = imgs[...,1] * np.conj(imgs[...,0])
        if romeo_uw:
            phasediff_uw = rh.romeo_unwrap(phasediff,[], result_path, mc_unwrap=True, return_b0=False)
        else:
            phasediff_uw = np.zeros_like(phasediff,dtype=np.float64)
            for k in range(phasediff.shape[-1]):
                phasediff_uw[...,k] = unwrap_phase(np.angle(phasediff[...,k])) # unwrap phase for each coil

        nc = phasediff_uw.shape[-1]
        phasediff_uw_rs = phasediff_uw.reshape([-1,nc])
        img_mag = abs(imgs[...,0]).reshape([-1,nc])
        ix = np.argsort(phasediff_uw_rs, axis=-1)[:,nc//4:-nc//4] # remove lowest & highest quartile
        weights = np.take_along_axis(img_mag, ix, axis=-1) / np.sum(np.take_along_axis(img_mag, ix, axis=-1), axis=-1)[:,np.newaxis]
        fmap = np.sum(weights * np.take_along_axis(phasediff_uw_rs, ix, axis=-1), axis=-1)
        fmap = fmap.reshape(imgs.shape[:3])
        te_diff = echo_times[1] - echo_times[0]
        fmap = -1 * fmap/te_diff # the sign in Powergrid is inverted
    else:
        # Standard field mapping approach (Hermitian product & SOS coil combination)
        phasediff = imgs[...,1] * np.conj(imgs[...,0]) 
        phasediff = np.sum(phasediff, axis=-1) # coil combination
        if romeo_uw:
            phasediff_uw = rh.romeo_unwrap(phasediff, [], result_path, mc_unwrap=False, return_b0=False)
        else:
            phasediff_uw = unwrap_phase(np.angle(phasediff))
        te_diff = echo_times[1] - echo_times[0]
        fmap = -1 * phasediff_uw/te_diff # the sign in Powergrid is inverted

    # mask with threshold and median otsu from dipy
    # do it in 2D as it works better and hole filling is easier
    img_mask = rh.rss(imgs[...,0], axis=-1)
    mask = np.zeros_like(img_mask)
    for k,img in enumerate(img_mask):
        _, mask_otsu = median_otsu(img, median_radius=1, numpass=20)

        # simple threshold mask
        thresh = 0.13
        mask_thresh = img/np.max(img)
        mask_thresh[mask_thresh<thresh] = 0
        mask_thresh[mask_thresh>=thresh] = 1

        # combine masks
        mask[k] = mask_thresh + mask_otsu
        mask[k][mask[k]>0] = 1
        mask[k] = binary_fill_holes(mask[k])
        mask[k] = binary_dilation(mask[k], iterations=2) # some extrapolation

    # Standard deviation filter (Robinson, MRM, 2011)
    if std_filter and mc_fmap:
        phasediff_uw *= mask[...,np.newaxis]
        std = phasediff_uw.std(axis=-1)
        median_std = np.median(std[std!=0])
        idx = np.argwhere(std>std_fac*median_std)
        n_cb = [2,2,1]
        for ix in idx:
            voxel_cb = tuple([slice(max(0,ix[k]-n_cb[k]), ix[k]+n_cb[k]+1) for k in range(3)])
            fmap[tuple(ix)] = np.median((fmap[voxel_cb])[np.nonzero(fmap[voxel_cb])])

    # Gauss/median filter
    if filtering:
        fmap *= mask
        fmap = gaussian_filter(fmap, sigma=0.5)
        fmap = median_filter(fmap, size=2)

    # interpolate to correct matrix size
    if nz == 1:
        newshape = [n_slc,ny,nx]
    else:
        newshape = [nz,ny,nx]
    fmap = resize(np.transpose(fmap,[0,2,1]), newshape, anti_aliasing=True)
    mask = resize(np.transpose(mask,[0,2,1]), newshape, anti_aliasing=False)
    mask[mask>0] = 1 # fix interpolation artifacts in binary mask

    fmap *= mask

    return fmap, mask

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
