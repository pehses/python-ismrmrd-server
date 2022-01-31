
import ismrmrd
import os
import itertools
import logging
import numpy as np
import base64
import ctypes

from bart import bart
import subprocess
from cfft import cfftn, cifftn
import mrdhelper

from pulseq_prot import insert_hdr, insert_acq, get_ismrmrd_arrays, check_signature
from reco_helper import calculate_prewhitening, apply_prewhitening, calc_rotmat, pcs_to_gcs, remove_os, filt_ksp
from reco_helper import fov_shift_spiral_reapply #, fov_shift_spiral, fov_shift 


""" Reconstruction of imaging data acquired with the Pulseq Sequence via the FIRE framework
    Reconstruction is done with the BART toolbox and the PowerGrid toolbox
"""

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

########################
# Main Function
########################

def process(connection, config, metadata):
    
    # -- Some manual parameters --- #
    
    # Select a slice (only for debugging purposes) - if "None" reconstruct all slices
    slc_sel = None
    if len(metadata.userParameters.userParameterLong) > 0:
        online_recon = True
        online_slc = metadata.userParameters.userParameterLong[0].value # Only online reco can send single slice number (different xml)
        if online_slc >= 0:
            slc_sel = int(online_slc)
            fast_recon = True
        else:
            fast_recon = False
    else:
        online_recon = False
        fast_recon = False

    # Coil Compression: Compress number of coils by n_compr coils
    n_compr = 0
    n_cha = metadata.acquisitionSystemInformation.receiverChannels
    if n_compr > 0 and n_compr<n_cha:
        cc_cha = n_cha - n_compr
        logging.debug(f'Coil Compression from {n_cha} to {cc_cha} channels.')
    elif n_compr<0 or n_compr>=n_cha:
        cc_cha = n_cha
        logging.debug('Invalid number of compressed coils.')
    else:
        cc_cha = n_cha

    # ----------------------------- #

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # ISMRMRD protocol file
    protFolder = os.path.join(dependencyFolder, "pulseq_protocols")
    prot_filename = os.path.splitext(metadata.userParameters.userParameterString[0].value)[0] # protocol filename from Siemens protocol parameter tFree, remove .seq ending in Pulseq version 1.4
    prot_file = protFolder + "/" + prot_filename + ".h5"

    # Check if local protocol folder is available, if protocol is not in dependency protocol folder
    if not os.path.isfile(prot_file):
        protFolder_local = "/tmp/local/pulseq_protocols" # optional local protocol mountpoint (via -v option)
        date = prot_filename.split('_')[0] # folder in Protocols (=date of seqfile)
        protFolder_loc = os.path.join(protFolder_local, date)
        prot_file_loc = protFolder_loc + "/" + prot_filename + ".h5"
        if os.path.isfile(prot_file_loc):
            prot_file = prot_file_loc
        else:
            raise ValueError("No protocol file available.")

    # check signature
    prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
    prot_hdr = ismrmrd.xsd.CreateFromDocument(prot.read_xml_header())
    check_signature(metadata, prot_hdr) # check MD5 signature
    prot.close()

    # Insert protocol header
    insert_hdr(prot_file, metadata)

    sms_factor = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2)
    if sms_factor > 1 and slc_sel is not None:
        raise ValueError("SMS reconstruction is not possible for single slices.")

    # Get additional arrays from protocol file - e.g. for diffusion imaging
    prot_arrays = get_ismrmrd_arrays(prot_file)

    # parameters for reapplying FOV shift
    nsegments = metadata.encoding[0].encodingLimits.segment.maximum + 1
    matr_sz = np.array([metadata.encoding[0].encodedSpace.matrixSize.x, metadata.encoding[0].encodedSpace.matrixSize.y])
    res = np.array([metadata.encoding[0].encodedSpace.fieldOfView_mm.x / matr_sz[0], metadata.encoding[0].encodedSpace.fieldOfView_mm.y / matr_sz[1], 1])

    # parameters for B0 correction
    dwelltime = 1e-6*metadata.userParameters.userParameterDouble[0].value # [s]
    t_min = metadata.userParameters.userParameterDouble[3].value # [s]

    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier

    try:
        # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))
        # logging.info("Metadata: \n%s", metadata.serialize())

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
    shim_currents = [k.value for k in metadata.userParameters.userParameterDouble[6:15]]
    ref_volt = metadata.userParameters.userParameterDouble[5].value
    logging.info(f"Measurement Frequency: {freq}")
    logging.info(f"Shim Currents: {shim_currents}")
    logging.info(f"Reference Voltage: {ref_volt}")

    # Initialize lists for datasets
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    n_intl = metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1

    acqGroup = [[[] for _ in range(n_contr)] for _ in range(n_slc)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * n_slc
    sensmaps_shots = [None] * n_slc
    dmtx = None
    offres = None 
    shotimgs = None
    sens_shots = False
    phs = None
    phs_ref = [None] * n_slc
    base_trj = None
    skope = False

    if "b_values" in prot_arrays and n_intl > 1:
        # we use the contrast index here to get the PhaseMaps into the correct order
        # PowerGrid reconstructs with ascending contrast index, so the phase maps should be ordered like that
        shotimgs = [[[] for _ in range(n_contr)] for _ in range(n_slc)]
        sens_shots = True

    # field map, if it was acquired - needs at least 2 reference contrasts
    max_refcontr = 0
    if 'echo_times' in prot_arrays:
        echo_times = prot_arrays['echo_times']
        te_diff = echo_times[1] - echo_times[0] # [s]
        process_raw.fmap = {'fmap': [None] * n_slc, 'mask': [None] * n_slc, 'name': 'Field Map from reference scan'}
    else:
        te_diff = None
        process_raw.fmap = None

    if cc_cha != n_cha:
        process_raw.cc_mat = [None] * n_slc # compression matrix
    else:
        process_raw.cc_mat = None
    process_raw.img_ix = 1 # img idx counter for single slice recos

    # read protocol acquisitions - faster than doing it one by one
    logging.debug("Reading in protocol acquisitions.")
    acqs = []
    prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
    for n in range(prot.number_of_acquisitions()):
        acqs.append(prot.read_acquisition(n))
    prot.close()

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
                    dmtx = calculate_prewhitening(noise_data)
                    del(noise_data)
                    noiseGroup.clear()
                
                # Phase correction scans (WIP: phase navigators not working correctly atm)
                if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    if item.idx.contrast == 0:
                        phs_ref[item.idx.slice] = item.data[:]
                    else:
                        data = item.data[:] * np.conj(phs_ref[item.idx.slice]) # subtract reference phase
                        data_sum = np.sum(data, axis=0) # sum weights coils by signal magnitude
                        slope = np.sum(data_sum[:-1] * np.conj(data_sum[1:]))
                        phs_slope = np.angle(slope)
                        offres = phs_slope / 1e-6 # 1 us dwelltime of phase correction scans
                    continue
                
                # Skope sync scans
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # skope sync scans
                    continue

                # skip slices in single slice reconstruction
                if slc_sel is None or item.idx.slice == slc_sel:

                    # Process reference scans
                    if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                        acsGroup[item.idx.slice].append(item)
                        if item.idx.contrast > max_refcontr:
                            max_refcontr = item.idx.contrast
                        if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) and item.idx.contrast==max_refcontr:
                            # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                            sensmaps[item.idx.slice], sensmaps_shots[item.idx.slice] = process_acs(acsGroup[item.idx.slice], metadata, cc_cha, dmtx, te_diff, sens_shots) # [nx,ny,nz,nc]
                            acsGroup[item.idx.slice].clear()
                        continue

                    # Process imaging scans - deal with ADC segments
                    if item.idx.segment == 0:
                        nsamples = item.number_of_samples
                        t_vec = t_min + dwelltime * np.arange(nsamples) # time vector for B0 correction
                        item.traj[:,3] = t_vec.copy()
                        acqGroup[item.idx.slice][item.idx.contrast].append(item)

                        # variables for reapplying FOV shift (see below)
                        pred_trj = item.traj[:]
                        rotmat = calc_rotmat(item)
                        shift = pcs_to_gcs(np.asarray(item.position), rotmat) / res
                    else:
                        # append data to first segment of ADC group
                        idx_lower = item.idx.segment * item.number_of_samples
                        idx_upper = (item.idx.segment+1) * item.number_of_samples
                        acqGroup[item.idx.slice][item.idx.contrast][-1].data[:,idx_lower:idx_upper] = item.data[:]

                    if item.idx.segment == nsegments - 1:
                        # Noise whitening
                        if dmtx is None:
                            data = acqGroup[item.idx.slice][-1].data[:]
                        else:
                            data = apply_prewhitening(acqGroup[item.idx.slice][item.idx.contrast][-1].data[:], dmtx)

                        # Reapply FOV Shift with predicted trajectory
                        data = fov_shift_spiral_reapply(data, pred_trj, base_trj, shift, matr_sz)
                        #--- FOV shift is done in the Pulseq sequence by tuning the ADC frequency   ---#
                        #--- However leave this code to fall back to reco shifts, if problems occur ---#
                        #--- and for reconstruction of old data                                     ---#
                        # rotmat = calc_rotmat(item)
                        # shift = pcs_to_gcs(np.asarray(item.position), rotmat) / res
                        # data = fov_shift_spiral(data, np.swapaxes(pred_trj,0,1), shift, matr_sz[0])

                        # filter signal to avoid Gibbs Ringing
                        traj_filt = np.swapaxes(acqGroup[item.idx.slice][item.idx.contrast][-1].traj[:,:3],0,1) # traj to [dim, samples]
                        acqGroup[item.idx.slice][item.idx.contrast][-1].data[:] = filt_ksp(data, traj_filt, filt_fac=0.95)
                        
                        # Correct the global phase
                        k0 = acqGroup[item.idx.slice][item.idx.contrast][-1].traj[:,4]
                        if skope:
                            acqGroup[item.idx.slice][item.idx.contrast][-1].data[:] *= np.exp(-1j*k0)
                        # WIP: phase navigators not working correctly atm
                        if not skope and offres is not None:
                            t_vec = acqGroup[item.idx.slice][item.idx.contrast][-1].traj[:,3]
                            global_phs = offres * t_vec + k0 # add up linear and GIRF predicted phase
                            acqGroup[item.idx.slice][item.idx.contrast][-1].data[:] *= np.exp(-1j*global_phs)
                            offres = None

                    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                        # Reconstruct shot images for phase maps in multishot diffusion imaging
                        if shotimgs is not None:
                            shotimgs[item.idx.slice][item.idx.contrast] = process_shots(acqGroup[item.idx.slice][item.idx.contrast], metadata, sensmaps_shots[item.idx.slice])

                    # Process acquisitions with PowerGrid - fast single slice online reco
                    if (item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) and fast_recon):
                        logging.info("Processing a group of k-space data")
                        group = acqGroup[item.idx.slice][item.idx.contrast]
                        if shotimgs is not None:
                            shotgroup = shotimgs[item.idx.slice][item.idx.contrast]
                        else:
                            shotgroup = None
                        images = process_raw_online(group, metadata, sensmaps, shotgroup, cc_cha, slc_sel)
                        logging.debug("Sending images to client:\n%s", images)
                        connection.send_image(images)

                # Process acquisitions with PowerGrid - full recon
                if (item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT) and not fast_recon):
                    logging.info("Processing a group of k-space data")
                    images = process_raw(acqGroup, metadata, sensmaps, shotimgs, prot_arrays, cc_cha, slc_sel, online_recon)
                    logging.debug("Sending images to client:\n%s", images)
                    connection.send_image(images)
                    acqGroup.clear()

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

def process_raw(acqGroup, metadata, sensmaps, shotimgs, prot_arrays, cc_cha, slc_sel=None, online_recon=False):

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
    sms_factor = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2)
    if sms_factor > 1:
        metadata.encoding[0].encodedSpace.matrixSize.z = sms_factor
        metadata.encoding[0].encodingLimits.slice.maximum = int((metadata.encoding[0].encodingLimits.slice.maximum + 1) / sms_factor + 0.5) - 1
    if slc_sel is not None:
        metadata.encoding[0].encodingLimits.slice.maximum = 0
    if avg_before:
        n_avg = metadata.encoding[0].encodingLimits.average.maximum + 1
        metadata.encoding[0].encodingLimits.average.maximum = 0
    dset_tmp.write_xml_header(metadata.toXML())

    # Insert Sensitivity Maps
    if slc_sel is not None:
        sens = np.transpose(sensmaps[slc_sel], [3,2,1,0])
        fmap_shape = [len(sensmaps)*sens.shape[1], sens.shape[2], sens.shape[3]]
    else:
        sens = np.transpose(np.stack(sensmaps), [0,4,3,2,1]) # [slices,nc,nz,ny,nx]
        fmap_shape = [sens.shape[0]*sens.shape[2], sens.shape[3], sens.shape[4]] # shape to check field map dims
        if sms_factor > 1:
            sens = reshape_sens_sms(sens, sms_factor)
    dset_tmp.append_array("SENSEMap", sens.astype(np.complex128))

    # Insert Field Map
    # process_raw.fmap = None
    if process_raw.fmap is not None:
        fmap = process_raw.fmap
        # 3D filtering of field map
        from scipy.ndimage import gaussian_filter
        if slc_sel is not None:
            fmap['fmap'][slc_sel] = gaussian_filter(fmap['fmap'][slc_sel], sigma=1)
        else:
            fmap['fmap'] = gaussian_filter(fmap['fmap'], sigma=1)
    else:
        fmap_path = dependencyFolder+"/fmap.npz"
        if not os.path.exists(fmap_path):
            fmap = {'fmap': np.zeros(fmap_shape), 'mask': np.ones(fmap_shape), 'name': 'No Field Map'}
            logging.debug("No field map file in dependency folder. Use zeros array instead. Field map should be .npz file.")
        else:
            fmap = np.load(fmap_path, allow_pickle=True)
            if 'name' not in fmap:
                fmap['name'] = 'No name.'
        if fmap_shape != list(fmap['fmap'].shape):
            logging.debug(f"Field Map dimensions do not fit. Fmap shape: {list(fmap['fmap'].shape)}, Img Shape: {fmap_shape}. Dont use field map in recon.")
            fmap['fmap'] = np.zeros(fmap_shape)
        if 'params' in fmap:
            logging.debug("Field Map regularisation parameters: %s",  fmap['params'].item())

    fmap_data = fmap['fmap']
    fmap_mask = fmap['mask']
    fmap_name = fmap['name']
    if slc_sel is not None:
        fmap_data = fmap_data[slc_sel]
        fmap_mask = fmap_mask[slc_sel]
    fmap_data = np.asarray(fmap_data)
    fmap_mask = np.asarray(fmap_mask)
    np.save(debugFolder+"/fmap_data.npy", fmap_data)
    np.save(debugFolder+"/fmap_mask.npy", fmap_mask)
    # remove slice dimension, if field map was 3D
    if len(fmap_data) == 1: fmap_data = fmap_data[0]
    if len(fmap_mask) == 1: fmap_mask = fmap_mask[0]
    if sms_factor > 1:
            fmap_data = reshape_fmap_sms(fmap_data, sms_factor) # reshape for SMS imaging

    dset_tmp.append_array('FieldMap', fmap_data) # [slices,nz,ny,nx] normally collapses to [slices/nz,ny,nx], 4 dims are only used in SMS case
    logging.debug("Field Map name: %s", fmap_name)

    # Calculate phase maps from shot images and append if necessary
    pcSENSE = False
    if shotimgs is not None:
        pcSENSE = True
        if slc_sel is not None:
            shotimgs = np.expand_dims(np.stack(shotimgs[slc_sel]),0)
        else:
            shotimgs = np.stack(shotimgs)
        shotimgs = np.swapaxes(shotimgs, 0, 1) # swap slice & contrast as slice phase maps should be ordered [contrast, slice, shots, ny, nx]
        mask = fmap_mask.copy()
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
                if slc_sel is not None:
                    if slc_ix != slc_sel:
                        continue
                    else:
                        acq.idx.slice = 0
                # Coil compression
                if process_raw.cc_mat is not None:
                    cc_data = bart(1, f'ccapply -S -p {cc_cha}', acq.data[:].T[np.newaxis,:,np.newaxis], process_raw.cc_mat[slc_ix]) # SVD based Coil compression
                    acq.resize(trajectory_dimensions=acq.trajectory_dimensions, number_of_samples=acq.number_of_samples, active_channels=cc_cha)
                    acq.data[:] = cc_data[0,:,0].T

                # get rid of k0 in 5th dim, we dont need it in PowerGrid
                save_trj = acq.traj[:,:4].copy()
                acq.resize(trajectory_dimensions=4, number_of_samples=acq.number_of_samples, active_channels=acq.active_channels)
                acq.traj[:] = save_trj.copy()
                dset_tmp.append_acquisition(acq)

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
    logging.debug(f'Readout is {1e3*readout_dur} ms. Use {ts} time segments.')

    # MPI and hyperthreading
    mpi = True
    hyperthreading = True
    import psutil
    if hyperthreading:
        cores = psutil.cpu_count(logical = True) # number of logical cores
        mpi_cmd = 'mpirun --use-hwthread-cpus'
    else:
        cores = psutil.cpu_count(logical = False) # number of phyiscal cores
        mpi_cmd = 'mpirun'

    # Source modules to use module load - module load sets correct LD_LIBRARY_PATH for MPI
    # the LD_LIBRARY_PATH is causing problems with BART though, so it has to be done here
    pre_cmd = 'source /etc/profile.d/modules.sh && module load /opt/nvidia/hpc_sdk/modulefiles/nvhpc/20.11 && '

    mps_server = False
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all' and mpi:
        # Start an MPS Server for faster MPI on GPU
        # See: https://stackoverflow.com/questions/34709749/how-do-i-use-nvidia-multi-process-service-mps-to-run-multiple-non-mpi-cuda-app
        # and https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf
        try:
            subprocess.run('nvidia-cuda-mps-control -d', shell=True, check=True, text=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            mps_server = True
        except subprocess.CalledProcessError as e:
            logging.debug("MPS Server not started. See error messages below.")
            logging.debug(e.stdout)

    # Define PowerGrid options
    pg_opts = f'-i {tmp_file} -o {pg_dir} -s {n_shots} -I {temp_intp} -t {ts} -B 1000 -n 15 -D 2' # -w option writes intermediate results as niftis in pg_dir folder
    if pcSENSE:
        if mpi:
            subproc = pre_cmd + f'{mpi_cmd} -n {cores} PowerGridPcSenseMPI_TS ' + pg_opts
        else:
            subproc = 'PowerGridPcSenseTimeSeg ' + pg_opts
    else:
        pg_opts += ' -F NUFFT'
        if mpi:
            subproc = pre_cmd + f'{mpi_cmd} -n {cores} PowerGridSenseMPI ' + pg_opts
        else:
            subproc = 'PowerGridIsmrmrd ' + pg_opts
    # Run in bash
    logging.debug("PowerGrid Reconstruction cmdline: %s",  subproc)
    try:
        process = subprocess.run(subproc, shell=True, check=True, text=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
    if sms_factor > 1: # reorder sms slices
        newshape = [_ for _ in data.shape]
        newshape[3:5] = [newshape[3]*newshape[4], 1]
        data = data.reshape(newshape, order='f')

    logging.debug("Image data is size %s" % (data.shape,))
   
    images = []
    dsets = []

    # If we have a diffusion dataset, b-value and direction contrasts are stored in contrast index
    # as otherwise we run into problems with the PowerGrid acquisition tracking.
    # We now (in case of diffusion imaging) split the b=0 image from other images and reshape to b-values (contrast) and directions (phase)
    n_bval = metadata.encoding[0].encodingLimits.contrast.center # number of b-values (incl b=0)
    n_dirs = metadata.encoding[0].encodingLimits.phase.center # number of directions
    if n_bval > 0:
        shp = data.shape
        b0 = np.expand_dims(np.abs(data[:,0]), 1)
        diffw_imgs = np.abs(data[:,1:]).reshape(shp[0], n_bval-1, n_dirs, shp[3], shp[4], shp[5], shp[6])
        scale = 1/np.max(diffw_imgs) # scale data to not run into problems with small numbers
        b0 *= scale
        diffw_imgs *= scale
        dsets.append(b0.copy())
        dsets.append(diffw_imgs.copy())
    else:
        dsets.append(data)

    # Diffusion evaluation
    if "b_values" in prot_arrays:
        mask = fmap_mask.copy()
        adc_maps = process_diffusion_images(b0, diffw_imgs, prot_arrays, mask)
        adc_maps = adc_maps[:,np.newaxis] # add empty nz dimension for correct flip
        dsets.append(adc_maps)

    # Correct orientation
    for k in range(len(dsets)):
        dsets[k] = np.swapaxes(dsets[k], -1, -2)
        dsets[k] = np.flip(dsets[k], (-3,-2,-1))
        # Normalize and convert to int16 for online recon
        if online_recon:
            dsets[k] *= 32767 * 0.8 / dsets[k].max()
            dsets[k] = np.around(dsets[k])
            dsets[k] = dsets[k].astype(np.int16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                        'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                        'WindowCenter':           '16384',
                        'WindowWidth':            '32768',
                        'Keep_image_geometry':    '1',
                        'PG_Options':              pg_opts,
                        'Field Map':               fmap_name})

    if slc_sel is None:
        meta['ImageRowDir'] = ["{:.18f}".format(acqGroup[0][0][0].read_dir[0]), "{:.18f}".format(acqGroup[0][0][0].read_dir[1]), "{:.18f}".format(acqGroup[0][0][0].read_dir[2])]
        meta['ImageColumnDir'] = ["{:.18f}".format(acqGroup[0][0][0].phase_dir[0]), "{:.18f}".format(acqGroup[0][0][0].phase_dir[1]), "{:.18f}".format(acqGroup[0][0][0].phase_dir[2])]
    else:
        meta['ImageRowDir'] = ["{:.18f}".format(acqGroup[slc_sel][0][0].read_dir[0]), "{:.18f}".format(acqGroup[slc_sel][0][0].read_dir[1]), "{:.18f}".format(acqGroup[slc_sel][0][0].read_dir[2])]
        meta['ImageColumnDir'] = ["{:.18f}".format(acqGroup[slc_sel][0][0].phase_dir[0]), "{:.18f}".format(acqGroup[slc_sel][0][0].phase_dir[1]), "{:.18f}".format(acqGroup[slc_sel][0][0].phase_dir[2])]

    xml = meta.serialize()

    series_ix = 0
    for data_ix,data in enumerate(dsets):
        # Format as ISMRMRD image data 
        # WIP: send data with repetition dimension (leave out the rep for loop)

        if data_ix < 2:
            for contr in range(data.shape[1]):
                series_ix += 1
                img_ix = 0
                for phs in range(data.shape[2]):
                    for slc in range(data.shape[3]):
                        for nz in range(data.shape[4]):
                            img_ix += 1
                            if slc_sel is None:
                                image = ismrmrd.Image.from_array(np.moveaxis(data[:,contr,phs,slc,nz],0,-1), acquisition=acqGroup[0][contr][0])
                                image.setHead(mrdhelper.update_img_header_from_raw(image.getHead(), acqGroup[0][contr][0].getHead()))
                            else:
                                image = ismrmrd.Image.from_array(np.moveaxis(data[:,contr,phs,slc,nz],0,-1), acquisition=acqGroup[slc_sel][contr][0])
                                image.setHead(mrdhelper.update_img_header_from_raw(image.getHead(), acqGroup[slc_sel][contr][0].getHead()))
                            image.image_index = img_ix
                            image.image_series_index = series_ix
                            image.slice = slc
                            if 'b_values' in prot_arrays:
                                image.user_int[0] = int(prot_arrays['b_values'][contr+data_ix])
                            if 'Directions' in prot_arrays and data_ix==1:
                                image.user_float[:3] = prot_arrays['Directions'][phs]
                            image.attribute_string = xml
                            image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
                            images.append(image)
        else:
            # atm only ADC maps
            series_ix += 1
            img_ix = 0
            for img in data:
                img_ix += 1
                image = ismrmrd.Image.from_array(img[0])
                image.image_index = img_ix
                image.image_series_index = series_ix
                image.slice = img_ix
                image.attribute_string = xml
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
                images.append(image)

    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images

def process_raw_online(acqGroup, metadata, sensmaps, shotimgs, cc_cha, slc_sel):

    logging.debug("Do fast single slice online reconstruction.")
    
    # Write ISMRMRD file for PowerGrid
    tmp_file = dependencyFolder+"/PowerGrid_tmpfile.h5"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    dset_tmp = ismrmrd.Dataset(tmp_file, create_if_needed=True)

    # Write header
    dset_tmp.write_xml_header(metadata.toXML())
    hdr_pg = ismrmrd.xsd.CreateFromDocument(dset_tmp.read_xml_header()) # copy to modify header
    hdr_pg.encoding[0].encodingLimits.slice.maximum = 0
    hdr_pg.encoding[0].encodingLimits.set.maximum = 0
    hdr_pg.encoding[0].encodingLimits.contrast.maximum = 0
    hdr_pg.encoding[0].encodingLimits.phase.maximum = 0
    hdr_pg.encoding[0].encodingLimits.average.maximum = 0
    dset_tmp.write_xml_header(hdr_pg.toXML())

    # Insert Sensitivity Maps
    sens = np.transpose(sensmaps[slc_sel], [3,2,1,0])
    fmap_shape = [len(sensmaps)*sens.shape[1], sens.shape[2], sens.shape[3]]
    dset_tmp.append_array("SENSEMap", sens.astype(np.complex128))

    # Insert Field Map
    if process_raw.fmap is not None:
        fmap = process_raw.fmap
        from scipy.ndimage import gaussian_filter
        fmap['fmap'][slc_sel] = gaussian_filter(fmap['fmap'][slc_sel], sigma=1)
    else:
        fmap_path = dependencyFolder+"/fmap.npz"
        if not os.path.exists(fmap_path):
            fmap = {'fmap': np.zeros(fmap_shape), 'mask': np.ones(fmap_shape), 'name': 'No Field Map'}
            logging.debug("No field map file in dependency folder. Use zeros array instead. Field map should be .npz file.")
        else:
            fmap = np.load(fmap_path, allow_pickle=True)
            if 'name' not in fmap:
                fmap['name'] = 'No name.'
        if fmap_shape != list(fmap['fmap'].shape):
            logging.debug(f"Field Map dimensions do not fit. Fmap shape: {list(fmap['fmap'].shape)}, Img Shape: {fmap_shape}. Dont use field map in recon.")
            fmap['fmap'] = np.zeros(fmap_shape)
        if 'params' in fmap:
            logging.debug("Field Map regularisation parameters: %s",  fmap['params'].item())

    fmap_data = np.asarray(fmap['fmap'][slc_sel])
    fmap_mask = np.asarray(fmap['mask'][slc_sel])
    fmap_name = fmap['name']
    dset_tmp.append_array('FieldMap', fmap_data)
    logging.debug("Field Map name: %s", fmap_name)

    # Calculate phase maps from shot images and append if necessary
    pcSENSE = False
    if shotimgs is not None:
        pcSENSE = True
        shotimgs = np.stack(shotimgs)[np.newaxis,np.newaxis] # stack shots & add axes for contrast & slice (which are =1 here)
        mask = fmap_mask.copy()
        phasemaps = calc_phasemaps(shotimgs, mask, metadata)
        dset_tmp.append_array("PhaseMaps", phasemaps)

    # Insert acquisitions
    for acq in acqGroup:
        slc_ix = acq.idx.slice
        acq.idx.slice = 0
        acq.idx.set = 0
        acq.idx.contrast = 0
        acq.idx.phase = 0
        acq.idx.average = 0
        if process_raw.cc_mat is not None:
            cc_data = bart(1, f'ccapply -S -p {cc_cha}', acq.data[:].T[np.newaxis,:,np.newaxis], process_raw.cc_mat[slc_ix])
            acq.resize(trajectory_dimensions=acq.trajectory_dimensions, number_of_samples=acq.number_of_samples, active_channels=cc_cha)
            acq.data[:] = cc_data[0,:,0].T

        save_trj = acq.traj[:,:4].copy()
        acq.resize(trajectory_dimensions=4, number_of_samples=acq.number_of_samples, active_channels=acq.active_channels)
        acq.traj[:] = save_trj.copy()
        dset_tmp.append_acquisition(acq)

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
    """
 
    temp_intp = 'hanning' # hanning / histo / minmax
    if temp_intp == 'histo' or temp_intp == 'minmax': ts = int(ts/1.5 + 0.5)
    logging.debug(f'Readout is {1e3*readout_dur} ms. Use {ts} time segments.')

    # Define PowerGrid options
    pg_opts = f'-i {tmp_file} -o {pg_dir} -s {n_shots} -I {temp_intp} -t {ts} -B 1000 -n 15 -D 2'
    logging.debug("PowerGrid Reconstruction options: %s",  pg_opts)
    if pcSENSE:
        subproc = 'PowerGridPcSenseTimeSeg ' + pg_opts
    else:
        pg_opts += ' -F NUFFT'
        subproc = 'PowerGridIsmrmrd ' + pg_opts
    # Run in bash
    try:
        process = subprocess.run(subproc, shell=True, check=True, text=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # logging.debug(process.stdout)
    except subprocess.CalledProcessError as e:
        logging.debug(e.stdout)
        raise RuntimeError("PowerGrid Reconstruction failed. See logfiles for errors.")

    # Image data is saved as .npy
    data = np.load(pg_dir + "/images_pg.npy")
    data = np.abs(data)

    """
    """

    # data to [nz,ny,nx]
    data = np.transpose(data, [3,4,2,1,0,5,6,7])
    data = data[0,0,0,0,0]

    # correct orientation at scanner (consistent with ICE)
    data = np.swapaxes(data, 1, 2)
    data = np.flip(data, (0,1,2))

    logging.debug("Image data is size %s" % (data.shape,))
   
    # Normalize and convert to int16
    data *= 32767 * 0.8 / data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                        'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                        'WindowCenter':           '16384',
                        'WindowWidth':            '32768',
                        'Keep_image_geometry':    '1',
                        'PG_Options':              pg_opts,
                        'Field Map':               fmap_name})

    xml = meta.serialize()
    images = []
    for nz in range(data.shape[0]):
        image = ismrmrd.Image.from_array(data[nz], acquisition=acqGroup[0])
        image.image_index = process_raw.img_ix
        process_raw.img_ix += 1
        image.image_series_index = 1
        image.slice = 0
        image.attribute_string = xml
        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        images.append(image)

    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images

def process_acs(group, metadata, cc_cha, dmtx=None, te_diff=None, sens_shots=False):
    """ Process reference scans for parallel imaging calibration
    """
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
        data = remove_os(data)
        data = np.swapaxes(data,0,1) # for correct orientation in PowerGrid

        slc_ix = group[0].idx.slice

        #--- FOV shift is done in the Pulseq sequence by tuning the ADC frequency   ---#
        #--- However leave this code to fall back to reco shifts, if problems occur ---#
        #--- and for reconstruction of old data                                     ---#
        # rotmat = calc_rotmat(group[0])
        # if not rotmat.any(): rotmat = -1*np.eye(3) # compatibility if refscan has no rotmat in protocol
        # res = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x
        # shift = pcs_to_gcs(np.asarray(group[0].position), rotmat) / res
        # data = fov_shift(data, shift)

        # SVD based Coil compression 
        n_cha = metadata.acquisitionSystemInformation.receiverChannels
        if cc_cha < n_cha:
            logging.debug(f'Calculate coil compression matrix.')
            process_raw.cc_mat[slc_ix] = bart(1, f'cc -A -M -S -p {cc_cha}', data[...,0])
            data_sens = np.zeros_like(data)[...,:cc_cha,:]
            for k in range(data.shape[-1]):
                data_sens[...,k] = bart(1, f'ccapply -S -p {cc_cha}', data[...,k], process_raw.cc_mat[slc_ix])
        else:
            data_sens = data.copy()

        # ESPIRiT calibration - use only first contrast
        gpu = False
        if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
            gpu = True
        if gpu and data_sens.shape[2] > 1: # only for 3D data, otherwise the overhead makes it slower than CPU
            logging.debug("Run Espirit on GPU.")
            sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I -c 0.9', data_sens[...,0]) # c: crop value ~0.9, t: threshold ~0.005, r: radius (default is 24)
        else:
            logging.debug("Run Espirit on CPU.")
            sensmaps = bart(1, 'ecalib -m 1 -k 6 -I -c 0.9', data_sens[...,0])

        # Field Map calculation - if acquired (dont use coil compressed data)
        refimg = cifftn(data, [0,1,2])
        if te_diff is not None and data.shape[-1] > 1:
            process_raw.fmap['fmap'][slc_ix], process_raw.fmap['mask'][slc_ix] = calc_fmap(refimg, te_diff, metadata)

        # calculate low resolution sensmaps for shot images (dont use coil compressed data)
        if sens_shots:
            os_region = metadata.userParameters.userParameterDouble[4].value
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

        np.save(debugFolder + "/" + "refimg.npy", refimg)
        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps, sensmaps_shots
    else:
        return None

def calc_fmap(imgs, te_diff, metadata):
    """ Calculate field maps from reference images with two different contrasts

        imgs: [nx,ny,nz,nc,n_contr] - atm: n_contr=2 mandatory
        te_diff: TE difference [s]
    """
    from skimage.restoration import unwrap_phase
    from skimage.transform import resize
    from scipy.ndimage import gaussian_filter, median_filter
    from scipy.ndimage.morphology import binary_fill_holes
    from dipy.segment.mask import median_otsu
    
    phasediff = imgs[...,0] * np.conj(imgs[...,1]) # phase difference
    phasediff = np.sum(phasediff, axis=-1) # coil combination
    if phasediff.shape[2] == 1:
        phasediff = phasediff[:,:,0]
    phasediff = unwrap_phase(np.angle(phasediff))
    phasediff = np.atleast_3d(phasediff)
    fmap = phasediff/te_diff

    # mask image with median otsu from dipy
    img = np.sqrt(np.sum(np.abs(imgs[...,0])**2, axis=-1))
    img_masked, mask_otsu = median_otsu(img, median_radius=1, numpass=20)

    # simple threshold mask
    thresh = 0.13
    mask_thresh = np.sqrt(np.sum(np.abs(imgs[...,0])**2, axis=-1))
    mask_thresh /= np.max(mask_thresh)
    mask_thresh[mask_thresh<thresh] = 0
    mask_thresh[mask_thresh>=thresh] = 1

    # combine masks
    mask = mask_thresh + mask_otsu
    mask[mask>0] = 1
    if mask.shape[-1] == 1:
        mask = binary_fill_holes(mask[...,0])[...,np.newaxis]
    else:
        mask = binary_fill_holes(mask)

    # apply masking and some regularization
    fmap *= mask
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

    # sort data - no coil compression
    data, trj = sort_spiral_data(group, metadata)

    # undo the swap in process_acs as BART needs different orientation
    sensmaps = np.swapaxes(sensmaps_shots, 0, 1) 

    # Reconstruct low resolution images
    # dont use GPU as it creates a lot of overhead, which causes longer recon times
    pics_config = 'pics -l1 -r 0.001 -S -e -i 15 -t'

    imgs = []
    for k in range(data.shape[2]):
        img = bart(1, pics_config, np.expand_dims(trj[:,:,k],2), np.expand_dims(data[:,:,k],2), sensmaps)
        imgs.append(img)
    
    np.save(debugFolder + "/" + "shotimgs.npy", imgs)

    return imgs

def calc_phasemaps(shotimgs, mask, metadata):
    """ Calculate phase maps for phase corrected reconstruction
    """

    from skimage.restoration import unwrap_phase
    from scipy.ndimage import  median_filter, gaussian_filter
    from skimage.transform import resize

    nx = metadata.encoding[0].encodedSpace.matrixSize.x

    phasemaps = np.conj(shotimgs[:,:,0,np.newaxis]) * shotimgs # 1st shot is taken as reference phase
    phasemaps = np.angle(phasemaps)

    # phase unwrapping & interpolation to higher resolution
    unwrapped_phasemaps = np.zeros([phasemaps.shape[0],phasemaps.shape[1],phasemaps.shape[2],nx,nx])
    for k in range(phasemaps.shape[0]):
        for j in range(phasemaps.shape[1]):
            for i in range(phasemaps.shape[2]):
                unwrapped = unwrap_phase(phasemaps[k,j,i], wrap_around=(False, False))
                unwrapped_phasemaps[k,j,i] = resize(unwrapped, [nx,nx])

    # mask all slices - need to swap shot and slice axis
    phasemaps = np.swapaxes(np.swapaxes(unwrapped_phasemaps, 1, 2) * mask, 1, 2)

    # filter phasemaps - seems to make it worse as resolution of phase maps is low
    phasemaps_filt = np.zeros_like(phasemaps)
    for k in range(phasemaps_filt.shape[0]):
        for j in range(phasemaps_filt.shape[1]):
            for i in range(phasemaps_filt.shape[2]):
                phasemaps_filt[k,j,i] = median_filter(phasemaps[k,j,i], size=9)
                # phasemaps_filt[k,j,i] = gaussian_filter(filtered, sigma=0.5)
    phasemaps = phasemaps_filt.copy()

    np.save(debugFolder + "/" + "phsmaps.npy", phasemaps)
    
    return phasemaps

def process_diffusion_images(b0, diffw_imgs, prot_arrays, mask):
    """ Calculate ADC maps from diffusion images
    """

    def geom_mean(arr, axis):
        return (np.prod(arr, axis=axis))**(1.0/arr.shape[axis])

    b_val = prot_arrays['b_values']
    n_bval = b_val.shape[0] - 1
    directions = prot_arrays['Directions']
    n_directions = directions.shape[0]

    # reshape images - atm: just average repetitions and Nz is not used (no 3D imaging for diffusion)
    b0 = b0[:,0,0,:,0,:,:].mean(0) # [slices, Ny, Nx]
    imgshape = [s for s in b0.shape]
    diff = np.transpose(diffw_imgs[:,:,:,:,0].mean(0), [2,3,4,1,0]) # from [Rep, b_val, Direction, Slice, Nz, Ny, Nx] to [Slice, Ny, Nx, Direction, b_val]

    #  WIP & not used: Fit ADC for each direction by linear least squares
    diff_norm = np.divide(diff.T, b0.T, out=np.zeros_like(diff.T), where=b0.T!=0).T # Nan is converted to 0
    diff_log  = -np.log(diff_norm, out=np.zeros_like(diff_norm), where=diff_norm!=0)
    if n_bval<4:
        d_dir = (diff_log / b_val[1:]).mean(-1)
    else:
        d_dir = np.polynomial.polynomial.polyfit(b_val[1:], diff_log.reshape([-1,n_bval]).T, 1)[1,].T.reshape(imgshape+[n_directions])

    # calculate trace images (geometric mean)
    trace = geom_mean(diff, axis=-2)

    # calculate trace ADC map with LLS
    trace_norm = np.divide(trace.T, b0.T, out=np.zeros_like(trace.T), where=b0.T!=0).T
    trace_log  = -np.log(trace_norm, out=np.zeros_like(trace_norm), where=trace_norm!=0)

    # calculate trace diffusion coefficient - WIP: Is the fitting function working right?
    if n_bval<3:
        adc_map = (trace_log / b_val[1:]).mean(-1)
    else:
        adc_map = np.polynomial.polynomial.polyfit(b_val[1:], trace_log.reshape([-1,n_bval]).T, 1)[1,].T.reshape(imgshape)

    adc_map *= mask

    return adc_map
    
# %%
#########################
# Sort Data
#########################

def sort_spiral_data(group, metadata):

    sig = list()
    trj = list()
    for acq in group:

        # signal - already fov shifted in insert_prot_ismrmrd
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
    nc = metadata.acquisitionSystemInformation.receiverChannels

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

    nx = 2 * metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.x
    # ny = metadata.encoding[0].encodedSpace.matrixSize.y
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
        
        if dmtx is None:
            kspace[enc1, enc2, :, col, contr] += acq.data
        else:
            kspace[enc1, enc2, :, col, contr] += apply_prewhitening(acq.data, dmtx)
        if contr==0:
            counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc, n_contr)
    kspace = np.transpose(kspace, [3, 0, 1, 2, 4])

    return kspace

def reshape_sens_sms(sens, sms_factor):
    # WIP: is this correct???
    # reshape sensmaps array for sms imaging, sensmaps for one acquisition are stored at nz
    sens_cpy = sens.copy() # [slices, coils, nz, ny, nx]
    slices_eff = sens_cpy.shape[0]//sms_factor
    sens = np.zeros([slices_eff, sens_cpy.shape[1], sms_factor, sens_cpy.shape[3], sens_cpy.shape[4]], dtype=sens_cpy.dtype)
    for slc in range(sens_cpy.shape[0]):
        sens[slc%slices_eff,:,slc//slices_eff] = sens_cpy[slc,:,0] 
    return sens

def reshape_fmap_sms(fmap, sms_factor):
    # WIP: is this correct???
    # reshape field map array for sms imaging

    fmap_cpy = fmap.copy() # [slices, ny, nx] slices could also be nz for 3D refscan, but doesnt matter here
    slices_eff = fmap_cpy.shape[0]//sms_factor
    fmap = np.zeros([slices_eff, sms_factor, fmap_cpy.shape[1], fmap_cpy.shape[2]], dtype=fmap_cpy.dtype)
    for slc in range(fmap_cpy.shape[0]):
        fmap[slc%slices_eff, slc//slices_eff] = fmap_cpy[slc] 
    return fmap
