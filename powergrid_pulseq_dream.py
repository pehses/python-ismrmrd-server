
import ismrmrd
import os
import logging
import numpy as np
import ctypes
import xml.dom.minidom
import tempfile
import psutil
from time import perf_counter

from bart import bart
import subprocess
from cfft import cfftn, cifftn

from pulseq_helper import insert_hdr, insert_acq, get_ismrmrd_arrays, read_acqs
import reco_helper as rh
from DreamMap import calc_fa, DREAM_filter_fid


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
    
    # -- manual parameters --- #

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
        os.makedirs(debugFolder, mode=0o774)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Check if ecalib maps calculated
    global read_ecalib
    if read_ecalib and not os.path.isfile(debugFolder + "/sensmaps.npy"):
        read_ecalib = False

    # Insert protocol header
    insert_hdr(prot_file, metadata)

    # Read user parameters
    up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}

    # Get additional arrays from protocol file for diffusion imaging
    prot_arrays = get_ismrmrd_arrays(prot_file)

    # parameters for reapplying FOV shift
    matr_sz = np.array([metadata.encoding[0].encodedSpace.matrixSize.x, metadata.encoding[0].encodedSpace.matrixSize.y])
    res = np.array([metadata.encoding[0].encodedSpace.fieldOfView_mm.x / matr_sz[0], metadata.encoding[0].encodedSpace.fieldOfView_mm.y / matr_sz[1], 1])

    # parameters for B0 correction
    dwelltime = 1e-6*up_double["dwellTime_us"] # [s]
    if "t_min_fid" in up_double:
        t_min_fid = up_double["t_min_fid"] # [s]
        t_min_ste = up_double["t_min_ste"] # [s]
    else:
        t_min_fid = 0
        t_min_ste = 0 
    spiral_delay =  up_double["traj_delay"] # [s]
    t_min_fid += int(spiral_delay/dwelltime) * dwelltime # account for removing possibly corrupted ADCs at the start (insert_acq)
    t_min_ste += int(spiral_delay/dwelltime) * dwelltime
    ste_ix = int(prot_arrays['dream'][0])

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
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1

    acqGroup = [[] for _ in range(n_contr)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = []
    sensmaps = None
    dmtx = None
    base_trj = None
    process_acs.cc_mat = None # compression matrix

    # field map, if it was acquired - needs at least 2 reference contrasts
    if 'echo_times' in prot_arrays:
        echo_times = prot_arrays['echo_times']
        process_acs.fmap = {'fmap': None, 'TE': echo_times, 'name': 'Field Map from reference scan'}
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

                # Process reference scans
                if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup.append(item)
                    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                        # Coil Compression calibration
                        if process_acs.cc_cha < n_cha and process_acs.cc_mat is None:
                            process_acs.cc_mat = rh.calibrate_cc(acsGroup, process_acs.cc_cha, apply_cc=False)
                        # run parallel imaging calibration
                        sensmaps = process_acs(acsGroup, metadata, dmtx) # [nx,ny,nz,nc]
                        acsGroup.clear()
                    continue                       

                # Process imaging scans
                nsamples = item.number_of_samples
                if item.idx.contrast == ste_ix:
                    t_min = t_min_ste
                else:
                    t_min = t_min_fid
                t_vec = t_min + dwelltime * np.arange(nsamples) # time vector for B0 correction
                item.traj[:,3] = t_vec.copy()

                # Noise whitening
                if dmtx is not None:
                    item.data[:] = rh.apply_prewhitening(item.data[:], dmtx)

                # Apply coil compression to spiral data
                if process_acs.cc_mat is not None:
                    rh.apply_cc(item, process_acs.cc_mat)

                # Reapply FOV Shift with predicted trajectory
                rotmat = rh.calc_rotmat(item)
                shift = rh.pcs_to_gcs(np.asarray(item.position), rotmat) # shift [mm] in GCS, as traj is in GCS
                shift_px = shift / res # shift in pixel
                item.data[:] = rh.fov_shift_spiral_reapply(item.data[:], item.traj[:], base_trj, shift_px, matr_sz)

                # filter signal to avoid Gibbs Ringing
                traj = np.swapaxes(item.traj[:,:3],0,1) # traj to [dim, samples]
                item.data[:] = rh.filt_ksp(item.data[:], traj, filt_fac=0.95)

                # remove ADC oversampling
                os_factor = up_double["os_factor"] if "os_factor" in up_double else 1
                if os_factor == 2:
                    rh.remove_os_spiral(item)
                
                acqGroup[item.idx.contrast].append(item)

                # Process acquisitions with PowerGrid - full recon
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
                    process_and_send(connection, acqGroup, metadata, sensmaps, prot_arrays)

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

def process_and_send(connection, acqGroup, metadata, sensmaps, prot_arrays):
    # Start data processing
    logging.info("Processing a group of k-space data")
    images = process_raw(acqGroup, metadata, sensmaps, prot_arrays)
    logging.debug("Sending images to client.")
    connection.send_image(images)
    acqGroup.clear()

def process_raw(acqGroup, metadata, sensmaps, prot_arrays):

    # Make temporary directory for PowerGrid file
    tmpdir = tempfile.TemporaryDirectory(dir=debugFolder)
    tempdir = tmpdir.name
    logging.debug(f"Temporary directory for PowerGrid results: {tempdir}")
    tmp_file = tempdir+"/PowerGrid_tmpfile.h5"

    # Write ISMRMRD file for PowerGrid
    dset_tmp = ismrmrd.Dataset(tmp_file, create_if_needed=True)

    # Write header
    dset_tmp.write_xml_header(metadata.toXML())

    # Insert Sensitivity Maps
    if read_ecalib:
        sens = np.load(debugFolder + "/sensmaps.npy")
    else:
        sens = np.transpose(np.stack(sensmaps), [3,2,1,0]) # [nc,nz,ny,nx]
    np.save(debugFolder + "/sensmaps.npy", sens)
    dset_tmp.append_array("SENSEMap", sens.astype(np.complex128))

    # Insert Field Map
    fmap = process_acs.fmap
    fmap_data = fmap['fmap']
    fmap_name = fmap['name']
    fmap_data = np.asarray(fmap_data)
    np.save(debugFolder+"/fmap_data.npy", fmap_data)
    dset_tmp.append_array('FieldMap', fmap_data) # [nz,ny,nx]
    logging.debug("Field Map name: %s", fmap_name)

    # DREAM parameters
    filt_fid = True # filter FID images?
    dream = prot_arrays['dream']
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    ste_ix = int(dream[0])
    fid_ix = n_contr-1-ste_ix
    alpha = dream[1]     # preparation FA
    tr = dream[2]        # [s]
    beta = dream[3]      # readout FA
    dummies = dream[4]   # number of dummy scans before readout echo train starts
    t1 = dream[5]        # [s]
    TM = dream[6]        # [s]
    tau = dream[7]       # [s]
    ste_data = np.asarray([acq.data[:] for acq in acqGroup[ste_ix]])
    fid_data = np.asarray([acq.data[:] for acq in acqGroup[fid_ix]])
    mean_alpha = calc_fa(ste_data.mean(), fid_data.mean())
    mean_beta = mean_alpha / alpha * beta

    # Insert acquisitions
    for contr in acqGroup:
        shot = 0
        ctr = 0
        for acq in contr:
            if filt_fid and acq.idx.contrast == fid_ix:
                if acq.idx.set != shot: # reset counter at start of new shot (= new STEAM prep)
                    ctr = 0
                shot = acq.idx.set
                ti = tr * (dummies + ctr) + (TM+tau)# TI estimate (time from STEAM prep to readout) [s]
                filt = DREAM_filter_fid(mean_alpha, mean_beta, tr, t1, ti)
                acq.data[:] *= filt
                ctr += 1
            dset_tmp.append_acquisition(acq)

    readout_dur = acq.traj[-1,3] - acq.traj[0,3]
    ts_time = int((acq.traj[-1,3] - acq.traj[0,3]) / 1e-3 + 0.5) # 1 time segment per ms readout
    ts_fmap = int(np.max(abs(fmap_data)) * (acq.traj[-1,3] - acq.traj[0,3]) / (np.pi/2)) # 1 time segment per pi/2 maximum phase evolution
    ts = max(ts_time, ts_fmap)
    dset_tmp.close()

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
    n_shots = metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1
    pg_opts = f'-i {tmp_file} -o {tempdir} -s {n_shots} -B 1000 -n 20 -D 2 -I {temp_intp} -t {ts} -F NUFFT' # -w option writes intermediate results as niftis in pg_dir folder
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
    data = np.load(os.path.join(tempdir, "images_pg.npy"))
    data = np.abs(data)

    """
    """

    # data should have output [Slice, Phase, Contrast/Echo, Avg, Rep, Nz, Ny, Nx]
    # change to [Contrast/Echo, Nz, Ny, Nx] and correct orientation
    data = data[0,0,:,0,0]
    data = np.transpose(data, [0,-1,-2,-3])
    data = np.flip(data, (-3,-2,-1))
    logging.debug("Image data is size %s" % (data.shape,))
   
    images = []
    dsets = []
    dsets.append(data)

    # Calculate FA maps and Ref Voltage maps
    ste = data[ste_ix]
    fid = data[fid_ix]
    fa_map = calc_fa(abs(ste), abs(fid))
    up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}
    current_refvolt = up_double["RefVoltage"]
    ref_volt_map = current_refvolt * (alpha/fa_map)
    fa_map *= 10 # increase dynamic range
    logging.debug(f"FA map is size {fa_map.shape}")
    dsets.append(fa_map)
    dsets.append(ref_volt_map)

    # Normalize and convert to int16 for online recon
    int_max = np.iinfo(np.uint16).max
    dsets[0] *= int_max / dsets[0].max() # scale FID/STE images
    for k in range(len(dsets)):
        dsets[k] = np.around(dsets[k])
        dsets[k] = dsets[k].astype(np.uint16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                        'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                        'WindowCenter':           str((int_max+1)//2),
                        'WindowWidth':            str(int_max+1),
                        'Keep_image_geometry':    1,
                        'PG_Options':              subproc,
                        'Field Map':               fmap_name})

    meta2 = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '512',
                         'WindowWidth':            '1024',
                         'Keep_image_geometry':    1})

    series_ix = 0
    for data_ix,data in enumerate(dsets):
        # Format as 3D ISMRMRD image data
        if data.ndim > 3:
            for contr in range(data.shape[0]):
                series_ix += 1
                image = ismrmrd.Image.from_array(data[contr], acquisition=acqGroup[contr][0])
                meta['ImageRowDir'] = ["{:.18f}".format(acqGroup[contr][0].read_dir[0]), "{:.18f}".format(acqGroup[contr][0].read_dir[1]), "{:.18f}".format(acqGroup[contr][0].read_dir[2])]
                meta['ImageColumnDir'] = ["{:.18f}".format(acqGroup[contr][0].phase_dir[0]), "{:.18f}".format(acqGroup[contr][0].phase_dir[1]), "{:.18f}".format(acqGroup[contr][0].phase_dir[2])]
                image.image_index = 0
                image.image_series_index = series_ix
                image.contrast = contr
                image.attribute_string = meta.serialize()
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                        ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                        ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
                images.append(image)
        else:
            # FA maps, Ref Voltage maps
            series_ix += 1
            image = ismrmrd.Image.from_array(data, acquisition=acqGroup[0][0])
            meta['ImageRowDir'] = ["{:.18f}".format(acqGroup[0][0].read_dir[0]), "{:.18f}".format(acqGroup[0][0].read_dir[1]), "{:.18f}".format(acqGroup[0][0].read_dir[2])]
            meta['ImageColumnDir'] = ["{:.18f}".format(acqGroup[0][0].phase_dir[0]), "{:.18f}".format(acqGroup[0][0].phase_dir[1]), "{:.18f}".format(acqGroup[0][0].phase_dir[2])]
            image.image_index = 0
            image.image_series_index = series_ix
            image.attribute_string = meta2.serialize()
            image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
            images.append(image)

    logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(meta.serialize()).toprettyxml())
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images

def process_acs(group, metadata, dmtx=None):
    """ Process reference scans for parallel imaging calibration
    """

    if len(group)==0:
        raise ValueError("Process ACS was triggered for empty acquisition group.")

    data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
    data = np.swapaxes(data,0,1) # for correct orientation in PowerGrid

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
            # sensmaps = bart(1, 'caldir 32', data[...,0])
        else:
            logging.debug("Run Espirit on CPU.")
            sensmaps = bart(1, 'ecalib -m 1 -k 6 -I', data[...,0])

    # Field Map calculation - if acquired
    refimg = cifftn(data, [0,1,2])
    np.save(debugFolder + "/refimg.npy", refimg)
    if process_acs.fmap is not None:
        echo_times = process_acs.fmap['TE']
        refimg = refimg[np.newaxis] # add slice axis
        process_acs.fmap['fmap'] = rh.calc_fmap(refimg, echo_times, metadata)

    np.save(debugFolder + "/" + "acs.npy", data)

    return sensmaps
    
# %%
#########################
# Sort Data
#########################

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
        if process_acs.cc_mat is not None:
            rh.apply_cc(acq, process_acs.cc_mat)

        kspace[enc1, enc2, :, col, contr] += acq.data   
        
        if contr==0:
            counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc, n_contr)
    kspace = np.transpose(kspace, [3, 0, 1, 2, 4])

    return kspace
