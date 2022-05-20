
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
import bartpy.tools
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

########################
# Main Function
########################

def process(connection, config, metadata):
    
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # ISMRMRD protocol file
    prot_folder = os.path.join(dependencyFolder, "metadata")
    prot_filename = os.path.splitext(metadata.userParameters.userParameterString[0].value)[0] # protocol filename from Siemens protocol parameter tFree, remove .seq ending in Pulseq version 1.4
    prot_file = prot_folder + "/" + prot_filename + ".h5"

    # Check if local protocol folder is available, if protocol is not in dependency protocol folder
    if not os.path.isfile(prot_file):
        prot_folder_local = "/tmp/local/metadata" # optional local protocol mountpoint (via -v option)
        date = prot_filename.split('_')[0] # folder in Protocols (=date of seqfile)
        prot_folder_loc = os.path.join(prot_folder_local, date)
        prot_file_loc = prot_folder_loc + "/" + prot_filename + ".h5"
        if os.path.isfile(prot_file_loc):
            prot_file = prot_file_loc
        else:
            raise ValueError(f"Metadata file {prot_file} not available.")

    # check signature
    prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
    prot_hdr = ismrmrd.xsd.CreateFromDocument(prot.read_xml_header())
    check_signature(metadata, prot_hdr) # check MD5 signature
    prot.close()

    # Insert protocol header
    insert_hdr(prot_file, metadata)

    # Read user parameters
    up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}

    # Get additional arrays for B0 correction
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
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1 # effective slices acquired (slices/sms_factor)
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1

    acqGroup = [[[] for _ in range(n_contr)] for _ in range(n_slc)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * n_slc
    dmtx = None
    base_trj = None

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
                               
                # Dummyscan data
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):
                    continue

                # Process reference scans
                if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    continue

                # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                if sensmaps[item.idx.slice] is None:
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], metadata, dmtx, te_diff) # [nx,ny,nz,nc]
                    acsGroup[item.idx.slice].clear()

                # Process imaging scans - deal with ADC segments 
                # (not needed in newer spiral sequence versions, but kept for compatibility also with other scanners)
                if item.idx.segment == 0:
                    nsamples = item.number_of_samples
                    t_vec = t_min + dwelltime * np.arange(nsamples) # time vector for B0 correction
                    traj_tmp = item.traj[:]
                    item.resize(number_of_samples=item.number_of_samples, active_channels=item.active_channels, trajectory_dimensions=4)
                    item.traj[:,:3] = traj_tmp[:,:3]
                    item.traj[:,3] = t_vec.copy()
                    acqGroup[item.idx.slice][item.idx.contrast].append(item)
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

                    # Reapply FOV Shift with predicted trajectory
                    rotmat = rh.calc_rotmat(item)
                    shift = rh.pcs_to_gcs(np.asarray(last_item.position), rotmat) # shift [mm] in GCS, as traj is in GCS
                    shift_px = shift / res # shift in pixel
                    last_item.data[:] = rh.fov_shift_spiral_reapply(last_item.data[:], last_item.traj[:], base_trj, shift_px, matr_sz)

                    # filter signal to avoid Gibbs Ringing
                    traj = np.swapaxes(last_item.traj[:,:3],0,1) # traj to [dim, samples]
                    last_item.data[:] = rh.filt_ksp(last_item.data[:], traj, filt_fac=0.95)

                # Process acquisitions with PowerGrid
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

    # Write ISMRMRD file for PowerGrid
    tmp_file = dependencyFolder+"/PowerGrid_tmpfile.h5"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    dset_tmp = ismrmrd.Dataset(tmp_file, create_if_needed=True)

    # Write header
    dset_tmp.write_xml_header(metadata.toXML())

    # Insert Sensitivity Maps
    sens = np.transpose(np.stack(sensmaps), [0,4,3,2,1]) # [slices,nc,nz,ny,nx]
    np.save(debugFolder + "/sensmaps.npy", sens)
    dset_tmp.append_array("SENSEMap", sens.astype(np.complex128))

    # Insert Field Map
    if process_acs.fmap is not None: # field map from reference scan
        fmap = process_acs.fmap
        fmap['fmap'] = gaussian_filter(fmap['fmap'], sigma=1.5) # 3D filtering, list will be converted to ndarray
    else: # external field map
        fmap_path = dependencyFolder+"/fmap.npz"
        fmap_shape = [sens.shape[0]*sens.shape[2], sens.shape[3], sens.shape[4]] # shape to check field map dims
        fmap = load_external_fmap(fmap_path, fmap_shape)

    fmap_data = fmap['fmap']
    fmap_name = fmap['name']
    if fmap_data.ndim == 4: # remove slice dimension, if 3D dataset 
        fmap_data = fmap_data[0]
    fmap_data = np.asarray(fmap_data)
    np.save(debugFolder+"/fmap_data.npy", fmap_data)

    dset_tmp.append_array('FieldMap', fmap_data) # [slices,nz,ny,nx] normally collapses to [slices/nz,ny,nx], 4 dims are only used in SMS case
    logging.debug("Field Map name: %s", fmap_name)

    # Insert acquisitions
    for slc in acqGroup:
        for contr in slc:
            for acq in contr:
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
    """
 
    temp_intp = 'hanning' # hanning / histo / minmax
    if temp_intp == 'histo' or temp_intp == 'minmax': ts = int(ts/1.5 + 0.5)
    else:
        logging.debug(f'Readout is {1e3*readout_dur} ms. Use {ts} time segments.')

    # MPI and hyperthreading
    mpi = False # MPI support - may need an mps server for higher performance: Run "nvidia-cuda-mps-control -d" to start an mps server
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

    # Define PowerGrid options / -w option writes intermediate results as niftis in pg_dir folder
    pg_opts = f'-i {tmp_file} -o {pg_dir} -s {n_shots} -B 1000 -n 20 -D 2  -I {temp_intp} -t {ts} -F NUFFT' 
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
    except subprocess.CalledProcessError as e:
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
   
    # Correct orientation, normalize and convert to int16 for online recon
    int_max = np.iinfo(np.uint16).max
    imascale = data.max()
    data = np.swapaxes(data, -1, -2)
    data = np.flip(data, (-4,-3,-2,-1))
    data *= int_max / imascale # images from PowerGrid (T2 and diff images)
    data = np.around(data)
    data = data.astype(np.uint16)

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
    rotmat = rh.calc_rotmat(acqGroup[0][0][0]) # rotmat is always the same
    images = []
    # Format as 2D ISMRMRD image data [nx,ny]
    for contr in range(data.shape[1]):
        series_ix += 1
        for phs in range(data.shape[2]):
            for rep in range(data.shape[0]): # save one repetition after another
                for slc in range(data.shape[3]):
                    img_ix += 1
                    for nz in range(data.shape[4]):
                        image = ismrmrd.Image.from_array(data[rep,contr,phs,slc,nz], acquisition=acqGroup[0][contr][0])
                        meta['ImageRowDir'] = ["{:.18f}".format(acqGroup[0][0][0].read_dir[0]), "{:.18f}".format(acqGroup[0][0][0].read_dir[1]), "{:.18f}".format(acqGroup[0][0][0].read_dir[2])]
                        meta['ImageColumnDir'] = ["{:.18f}".format(acqGroup[0][0][0].phase_dir[0]), "{:.18f}".format(acqGroup[0][0][0].phase_dir[1]), "{:.18f}".format(acqGroup[0][0][0].phase_dir[2])]
                        image.image_index = img_ix
                        image.image_series_index = series_ix
                        image.slice = slc
                        image.repetition = rep
                        image.phase = phs
                        image.contrast = contr
                        image.attribute_string = meta.serialize()
                        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                                ctypes.c_float(1)) # 2D
                        offset = [0, 0, -1*slc_res*(slc-(n_slc-1)/2)] # slice offset in GCS
                        image.position[:] += rh.gcs_to_pcs(offset, rotmat) # correct image position in PCS
                        images.append(image)

    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images

def process_acs(group, metadata, dmtx=None, te_diff=None):
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
    if gpu and data.shape[2] > 1: # only for 3D data, otherwise the overhead makes it slower than CPU
        logging.debug("Run Espirit on GPU.")
        sensmaps = bart(1, 'ecalib -g -m 1 -k 6 -I', data[...,0]) # c: crop value ~0.9, t: threshold ~0.005, r: radius (default is 24)
    else:
        logging.debug("Run Espirit on CPU.")
        sensmaps = bart(1, 'ecalib -m 1 -k 6 -I', data[...,0])

    # Field Map calculation - if acquired
    refimg = cifftn(data, [0,1,2])
    if te_diff is not None and data.shape[-1] > 1:
        process_acs.fmap['fmap'][slc_ix], process_acs.fmap['mask'][slc_ix] = calc_fmap(refimg, te_diff, metadata)

    np.save(debugFolder + "/" + "acs.npy", data)

    return sensmaps

def calc_fmap(imgs, te_diff, metadata):
    """ Calculate field maps from reference images with two different contrasts

        imgs: [nx,ny,nz,nc,n_contr] - atm: n_contr=2 mandatory
        te_diff: TE difference [s]
    """
    
    phasediff = imgs[...,0] * np.conj(imgs[...,1]) # phase difference
    phasediff = np.sum(phasediff, axis=-1) # coil combination
    if phasediff.shape[2] == 1:
        phasediff = phasediff[:,:,0]
    phasediff = unwrap_phase(np.angle(phasediff))
    phasediff = np.atleast_3d(phasediff)
    fmap = phasediff/te_diff

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

    nc = metadata.acquisitionSystemInformation.receiverChannels
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
        
        kspace[enc1, enc2, :, col, contr] += acq.data   
        
        if contr==0:
            counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc, n_contr)
    kspace = np.transpose(kspace, [3, 0, 1, 2, 4])

    return kspace

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
