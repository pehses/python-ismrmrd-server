
import ismrmrd
import os
import itertools
import logging
import numpy as np
import base64

from bart import bart
# from PowerGridPy import PowerGridIsmrmrd
import subprocess
from cfft import cfftn, cifftn

from pulseq_prot import insert_hdr, insert_acq, get_ismrmrd_arrays
from reco_helper import calculate_prewhitening, apply_prewhitening, calc_rotmat, pcs_to_gcs, fov_shift_spiral, fov_shift, remove_os, filt_ksp

""" Reconstruction of imaging data acquired with the Pulseq Sequence via the FIRE framework
    Reconstruction is done with the BART toolbox and the PowerGrid toolbox

Short Comments on FOV Shifting and rotations:
    - Translational shifts should not be acticvated in the GUI, when using the Pulseq Sequence.
      However in-plane FOV shifts can be selected also without activating FOV positioning and are applied in this reco.

    - Rotations are not yet possible as only the standard rotation matrix for Pulseq is considered and taken from the protocol dile.
      The standard rotation matrix was obtained by simulating predefined gradients.
      To implement rotations selected in the GUI, the correct rotation matrix has to be obtained from the dataset ISMRMRD file coming from the scanner.
      However, that rotation matrix seems to be incorrect, as it switches phase and read gradients. It seems like an additional transformation would have to be applied.

"""

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

########################
# Main Function
########################

def process(connection, config, metadata):
    
    # Select a slice (only for debugging purposes) - if "None" reconstruct all slices
    slc_sel = None

    # Set this True, if a Skope trajectory is used (protocol file with skope trajectory has to be available)
    skope = False

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    protFolder = os.path.join(dependencyFolder, "pulseq_protocols")
    protFolder_local = "/tmp/local/pulseq_protocols" # Protocols mountpoint (not at the scanner)
    prot_filename = metadata.userParameters.userParameterString[0].value_ # protocol filename from Siemens protocol parameter tFree
    if skope:
        prot_filename += "_skopetraj"

    # Check if local protocol folder is available - if not use protFolder (scanner)
    date = prot_filename.split('_')[0] # folder in Protocols (=date of seqfile)
    protFolder_loc = os.path.join(protFolder_local, date)
    if os.path.exists(protFolder_loc):
        protFolder = protFolder_loc

    # Insert protocol header
    prot_file = protFolder + "/" + prot_filename
    insert_hdr(prot_file, metadata)

    # Get additional arrays from protocol file - e.g. for diffusion imaging
    prot_arrays = get_ismrmrd_arrays(prot_file)

    # define variables for FOV shift
    nsegments = metadata.userParameters.userParameterDouble[2].value_
    matr_sz = metadata.encoding[0].encodedSpace.matrixSize.x
    res = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / matr_sz

    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier

    try:
        # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))
        # logging.info("Metadata: \n%s", metadata.serialize())

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3", 
            metadata.encoding[0].trajectory, 
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
    shim_currents = [k.value_ for k in metadata.userParameters.userParameterDouble[6:15]]
    ref_volt = metadata.userParameters.userParameterDouble[5].value_
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
    dmtx = None
    base_trj_ = []

    if "b_values" in prot_arrays and n_intl > 1:
        # we use the contrast index here to get the PhaseMaps into the correct order
        # PowerGrid reconstructs with ascending contrast index, so the phase maps should be ordered like that
        shotimgs = [[[] for _ in range(n_contr)] for _ in range(n_slc)]
    else:
        shotimgs = None

    try:
        for acq_ctr, item in enumerate(connection):

            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                # insert acquisition protocol
                base_trj = insert_acq(prot_file, item, acq_ctr)
                if base_trj is not None: # base_trj is calculated e.g. for future trajectory comparisons
                    base_trj_.append(base_trj)

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
                
                # Check for additional flags
                if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # skope sync scans
                    continue

                if slc_sel is None or item.idx.slice == slc_sel:
                    # Process reference scans
                    if item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                        acsGroup[item.idx.slice].append(item)
                        continue
                    elif sensmaps[item.idx.slice] is None:
                        # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                        sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx) # [nx,ny,nz,nc]
                        acsGroup[item.idx.slice].clear()

                    # Process imaging scans - deal with ADC segments
                    if item.idx.segment == 0:
                        acqGroup[item.idx.slice][item.idx.contrast].append(item)
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

                        # In-Plane FOV-shift
                        rotmat = calc_rotmat(item)
                        shift = pcs_to_gcs(np.asarray(item.position), rotmat) / res
                        traj = np.swapaxes(acqGroup[item.idx.slice][item.idx.contrast][-1].traj[:,:3],0,1)
                        traj = traj[[1,0,2],:]  # switch x and y dir for correct orientation
                        data = fov_shift_spiral(data, traj, shift, matr_sz)

                        # filter signal to avoid Gibbs Ringing
                        acqGroup[item.idx.slice][item.idx.contrast][-1].data[:] = filt_ksp(data, traj, filt_fac=0.95)
                        
                        # Set an offresonance frequency for debugging
                        offres_freq = 0 # [Hz]
                        t_vec = acqGroup[item.idx.slice][item.idx.contrast][-1].traj[:,3]
                        phase = 2*np.pi*offres_freq * t_vec
                        acqGroup[item.idx.slice][item.idx.contrast][-1].data[:] *= np.exp(-1j*phase)

                    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                        # if no refscan, calculate sensitivity maps from raw data
                        if sensmaps[item.idx.slice] is None: 
                            sensmaps[item.idx.slice] = sens_from_raw(acqGroup[item.idx.slice][item.idx.contrast], metadata)
                        # Reconstruct shot images for phase maps in multishot diffusion imaging
                        if shotimgs is not None:
                            shotimgs[item.idx.slice][item.idx.contrast] = process_shots(acqGroup[item.idx.slice][item.idx.contrast], metadata, sensmaps[item.idx.slice])

                # When all acquisitions are processed, write them to file for PowerGrid Reco,
                # which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
                    logging.info("Processing a group of k-space data")
                    images = process_raw(acqGroup, metadata, sensmaps, shotimgs, prot_arrays, slc_sel)
                    logging.debug("Sending images to client:\n%s", images)
                    connection.send_image(images)

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

def process_raw(acqGroup, metadata, sensmaps, shotimgs, prot_arrays, slc_sel=None):

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
    if slc_sel is not None:
        metadata.encoding[0].encodingLimits.slice.maximum = 0
    if avg_before:
        n_avg = metadata.encoding[0].encodingLimits.average.maximum + 1
        metadata.encoding[0].encodingLimits.average.maximum = 0
    dset_tmp.write_xml_header(metadata.toxml())

    # Insert Field Map
    fmap_path = dependencyFolder+"/fmap.npz"
    if not os.path.exists(fmap_path):
        raise ValueError("No field map file in dependency folder. Field map should be .npz file containing the field map and field map regularisation parameters")
    fmap = np.load(fmap_path, allow_pickle=True)
    fmap_data = fmap['fmap']
    if slc_sel is not None:
        fmap_data = fmap_data[slc_sel]

    logging.debug("Field Map name: %s", fmap['name'].item())
    if 'params' in fmap:
        logging.debug("Field Map regularisation parameters: %s",  fmap['params'].item())
    dset_tmp.append_array('FieldMap', fmap_data) # dimensions in PowerGrid seem to be [slices/nz,ny,nx]

    # Insert Sensitivity Maps
    if slc_sel is not None:
        sens = np.transpose(sensmaps[slc_sel], [3,2,1,0])
    else:
        sens = np.transpose(np.stack(sensmaps), [0,4,3,2,1]) # [slices,nc,nz,ny,nx] - only tested for 2D, nx/ny might be changed depending on orientation
    dset_tmp.append_array("SENSEMap", sens.astype(np.complex128))

    # Calculate phase maps from shot images and append if necessary
    # WIPs: Compare bet_mask and mask
    pcSENSE = False
    if shotimgs is not None:
        pcSENSE = True
        if slc_sel is not None:
            shotimgs = np.expand_dims(np.stack(shotimgs[slc_sel]),0)
        else:
            shotimgs = np.stack(shotimgs)
        shotimgs = np.swapaxes(shotimgs, 0, 1) # swap slice & contrast as slice phase maps should be ordered [contrast, slice, shots, ny, nx]
        shotimgs = np.swapaxes(shotimgs, -1, -2) # swap nx & ny
        # mask = fmap['bet_mask']
        mask = fmap['mask'] # seems to make no difference which mask is used
        if slc_sel:
            mask = mask[slc_sel]
        phasemaps = calc_phasemaps(shotimgs, mask)
        np.save(debugFolder + "/" + "phsmaps.npy", phasemaps)
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
                if avg_before:
                    if acq.idx.average == 0:
                        acq.data[:] = avgData[avg_ix]
                        avg_ix += 1
                    else:
                        continue
                if slc_sel is not None:
                    if acq.idx.slice != slc_sel:
                        continue
                    else:
                        acq.idx.slice = 0
                dset_tmp.append_acquisition(acq)
    dset_tmp.close()
    acqGroup.clear() # free memory

    # Define in- and output for PowerGrid
    tmp_file = dependencyFolder+"/PowerGrid_tmpfile.h5"
    pg_dir = dependencyFolder+"/powergrid_results"
    if not os.path.exists(pg_dir):
        os.makedirs(pg_dir)
    if os.path.exists(pg_dir+"/images_pg.npy"):
        os.remove(pg_dir+"/images_pg.npy")
    n_shots = metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1

    """ PowerGrid reconstruction
    # Comment from Alex Cerjanic, who built PowerGrid: 'histo' option can generate a bad set of interpolators in edge cases
    # He recommends using the Hanning interpolator with ~1 time segment per ms of readout (which is based on experience @3T)
    # However, histo lead to quite nice results so far & does not need as many time segments
    """
 
    # Source modules to use module load - module load sets correct LD_LIBRARY_PATH for MPI
    # the LD_LIBRARY_PATH is causing problems with BART though, so it has to be done here
    mpi = True
    pre_cmd = 'source /etc/profile.d/modules.sh && module load /opt/nvidia/hpc_sdk/modulefiles/nvhpc/20.11 && '
    import psutil
    cores = psutil.cpu_count(logical = False) # number of physical cores

    # Define PowerGrid options
    pg_opts = f'-i {tmp_file} -o {pg_dir} -s {n_shots} -I hanning -t 20 -B 1000 -n 15 -D 2' # -w option writes intermediate results as niftis in pg_dir folder
    logging.debug("PowerGrid Reconstruction options: %s",  pg_opts)
    if pcSENSE:
        if mpi:
            subproc = pre_cmd + f'mpirun -n {cores} PowerGridPcSenseMPI_TS ' + pg_opts
        else:
            subproc = 'PowerGridPcSenseTimeSeg ' + pg_opts
    else:
        pg_opts += ' -F NUFFT'
        if mpi:
            subproc = pre_cmd + f'mpirun -n {cores} PowerGridSenseMPI ' + pg_opts
        else:
            subproc = 'PowerGridIsmrmrd ' + pg_opts
    # Run in bash
    try:
        subprocess.run(subproc, shell=True, check=True, text=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.debug(e.stdout)
        raise RuntimeError("PowerGrid Reconstruction failed. See logfiles for errors.")

    # Image data is saved as .npy
    data = np.load(pg_dir + "/images_pg.npy")
    data = np.abs(data)

    ### old implementation with pybind
    # data = PowerGridIsmrmrd(inFile=tmp_file, outFile=debug_pg+"/img", timesegs=20, niter=10, nShots=n_shots, beta=0, 
                                # ts_adapt=False, TSInterp='hanning', FourierTrans='NUFFT', pcSENSE=pcSENSE)
    # shapes = data["shapes"] 
    # data = np.asarray(data["img_data"]).reshape(shapes)
    # data = np.abs(data)

    """
    """

    # data should have output [Slice, Phase, Contrast/Echo, Avg, Rep, Nz, Ny, Nx]
    # change to [Avg, Rep, Contrast/Echo, Phase, Slice, Nz, Ny, Nx] and average
    data = np.transpose(data, [3,4,2,1,0,5,6,7]).mean(axis=0)

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
        b0 = np.expand_dims(data[:,0], 1)
        diffw_imgs = data[:,1:].reshape(shp[0], n_bval-1, n_dirs, shp[3], shp[4], shp[5], shp[6])
        dsets.append(b0)
        dsets.append(diffw_imgs)
    else:
        dsets.append(data)

    # Diffusion evaluation
    if "b_values" in prot_arrays:
        mask = fmap['mask']
        if slc_sel is not None:
            mask = mask[slc_sel]
        adc_maps = process_diffusion_images(b0, diffw_imgs, prot_arrays, mask)
        dsets.append(adc_maps)

    # Normalize and convert to int16
    for k in range(len(dsets)):
        dsets[k] *= 32767 * 0.8 / dsets[k].max()
        dsets[k] = np.around(dsets[k])
        dsets[k] = dsets[k].astype(np.int16)

    # Set ISMRMRD Meta Attributes
        meta = ismrmrd.Meta({'DataRole':               'Image',
                            'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                            'WindowCenter':           '16384',
                            'WindowWidth':            '32768'})
        xml = meta.serialize()

    series_ix = 0
    for data_ix,data in enumerate(dsets):
        # Format as ISMRMRD image data
        if data_ix < 2:
            for rep in range(data.shape[0]):
                for contr in range(data.shape[1]):
                    series_ix += 1
                    img_ix = 0
                    for phs in range(data.shape[2]):
                        for slc in range(data.shape[3]):
                            for nz in range(data.shape[4]):
                                img_ix += 1
                                image = ismrmrd.Image.from_array(data[rep,contr,phs,slc,nz])
                                image.image_index = img_ix # contains slices/partitions and phases
                                image.image_series_index = series_ix # contains repetitions, contrasts
                                image.slice = 0 # WIP: test counting slices, contrasts, ... at scanner
                                if 'b_values' in prot_arrays:
                                    image.user_int[0] = int(prot_arrays['b_values'][contr+data_ix])
                                if 'Directions' in prot_arrays:
                                    image.user_float[:3] = prot_arrays['Directions'][phs]
                                image.attribute_string = xml
                                images.append(image)
        else:
            # atm only ADC maps
            series_ix += 1
            img_ix = 0
            for img in data:
                img_ix += 1
                image = ismrmrd.Image.from_array(img)
                image.image_index = img_ix
                image.image_series_index = series_ix
                image.slice = 0
                image.attribute_string = xml
                images.append(image)

    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images

def process_acs(group, config, metadata, dmtx=None):
    """ Process reference scans for parallel imaging calibration
    """
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
        data = remove_os(data)

        # fov shift
        rotmat = calc_rotmat(group[0])
        if not rotmat.any(): rotmat = -1*np.eye(3) # compatibility if refscan has no rotmat in protocol
        res = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x
        shift = pcs_to_gcs(np.asarray(group[0].position), rotmat) / res
        data = fov_shift(data, shift)

        data = np.swapaxes(data,0,1) # in Pulseq gre_refscan sequence read and phase are changed, might change this in the sequence
        if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
            print("Run Espirit on GPU.")
            sensmaps = bart(1, 'ecalib -g -m 1 -k 8 -I', data)  # ESPIRiT calibration
        else:
            print("Run Espirit on CPU.")
            sensmaps = bart(1, 'ecalib -m 1 -k 8 -I', data)  # ESPIRiT calibration

        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None

def sens_from_raw(group, metadata):
    """ Calculate sensitivity maps from imaging data (if no reference scan was done)
    """
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    
    data, trj = sort_spiral_data(group, metadata)

    sensmaps = bart(1, 'nufft -i -l 0.005 -t -d %d:%d:%d'%(nx, nx, nz), trj, data) # nufft
    sensmaps = cfftn(sensmaps, [0, 1, 2]) # back to k-space
    sensmaps = bart(1, 'ecalib -m 1 -I', sensmaps)  # ESPIRiT calibration
    return sensmaps

def process_shots(group, metadata, sensmaps):
    """ Reconstruct images from single shots for calculation of phase maps

    WIP: maybe use PowerGrid for B0-correction? If recon without B0 correction is sufficient, BART is more time efficient
    """

    from skimage.transform import resize

    # sort data
    data, trj = sort_spiral_data(group, metadata)

    # Interpolate sensitivity maps to lower resolution
    os_region = metadata.userParameters.userParameterDouble[4].value_
    if np.allclose(os_region,0):
        os_region = 0.25 # use default if no region provided
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    newshape = [int(nx*os_region), int(nx*os_region)] + [k for k in sensmaps.shape[2:]]
    sensmaps = resize(sensmaps.real, newshape, anti_aliasing=True) + 1j*resize(sensmaps.imag, newshape, anti_aliasing=True)

    # Reconstruct low resolution images
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
       pics_config = 'pics -g -l1 -r 0.001 -S -e -i 30 -t'
    else:
       pics_config = 'pics -l1 -r 0.001 -S -e -i 30 -t'

    imgs = []
    for k in range(data.shape[2]):
        img = bart(1, pics_config, np.expand_dims(trj[:,:,k],2), np.expand_dims(data[:,:,k],2), sensmaps)
        img = resize(img.real, [nx,nx], anti_aliasing=True) + 1j*resize(img.imag, [nx,nx], anti_aliasing=True) # interpolate back to high resolution
        imgs.append(img)
    
    np.save(debugFolder + "/" + "shotimgs.npy", imgs)

    return imgs

def calc_phasemaps(shotimgs, mask):
    """ Calculate phase maps for phase corrected reconstruction
    """

    from skimage.restoration import unwrap_phase
    from scipy.ndimage import  median_filter, gaussian_filter

    phasemaps = np.conj(shotimgs[:,:,0,np.newaxis]) * shotimgs # 1st shot is taken as reference phase
    phasemaps = np.angle(phasemaps)
    phasemaps = np.swapaxes(np.swapaxes(phasemaps, 1, 2) * mask, 1, 2) # mask all slices - need to swap shot and slice axis

    # phase unwrapping & smooting with median and gaussian filter
    unwrapped_phasemaps = np.zeros_like(phasemaps)
    for k in range(phasemaps.shape[0]):
        for j in range(phasemaps.shape[1]):
            for i in range(phasemaps.shape[2]):
                unwrapped = unwrap_phase(phasemaps[k,j,i], wrap_around=(False, False))
                unwrapped = median_filter(unwrapped, size=3)
                unwrapped_phasemaps[k,j,i] = gaussian_filter(unwrapped, sigma=1.5)

    return unwrapped_phasemaps

def process_diffusion_images(b0, diffw_imgs, prot_arrays, mask):
    """ Calculate ADC maps from diffusion images
    """

    def geom_mean(arr, axis):
        return (np.prod(arr, axis=axis))**(1.0/arr.shape[axis])

    b_val = prot_arrays['b_values']
    n_bval = b_val.shape[0] - 1
    directions = prot_arrays['Directions']
    n_directions = directions.shape[0]

    # reshape images - we dont use repetions and Nz (no 3D imaging for diffusion)
    b0 = b0[0,0,0,:,0,:,:] # [slices, Ny, Nx]
    imgshape = [s for s in b0.shape]
    diff = np.transpose(diffw_imgs[0,:,:,:,0], [2,3,4,1,0]) # from [Rep, b_val, Direction, Slice, Nz, Ny, Nx] to [Slice, Ny, Nx, Direction, b_val]

    # Fit ADC for each direction by linear least squares
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
        traj = traj[[1,0,2],:]  # switch x and y dir for correct orientation
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

    nx = 2 * metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.x
    # ny = metadata.encoding[0].encodedSpace.matrixSize.y
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
