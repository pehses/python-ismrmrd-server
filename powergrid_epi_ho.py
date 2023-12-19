
import ismrmrd
import os
import logging
import numpy as np
import ctypes
import xml.dom.minidom
import psutil
from time import perf_counter
import importlib

import subprocess
import reco_helper as rh
from bart import bart
from skimage.transform import resize
from cfft import cfftn, cifftn

""" Higher order reconstruction of imaging data acquired with the dzne_ep2d_diff sequence
    Reconstruction is done with the BART toolbox and the PowerGrid toolbox

    The higher order reconstruction is done in the physical coordinate system (DCS) and currently only possible, when Skope data is already inserted.
    The k-space coefficients from the Skope system have to stay in the physical coordinate system without rescaling, when they are inserted into the MRD file. 
    Units are: 1st order: [rad/m], 2nd order: [rad/m^2], ...

    The MRD trajectory field should have the following trajectory dimensions
    0: kx, 1: ky, 2: kz, 3: t-vec, 4-8: 2nd order, 9-15: 3rd order, 16-19: concomitant fields

    If 2nd or 3rd order was not measured, the concomitant field terms move to lower dimensions (no zero-filling), to reduce the required space
    t-vec is the time vector of the readout regarding the echo time
        
"""

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

read_ecalib = False # read sensitivity maps from file (requires previous recon)
save_cmplx = True # save images as complex data

snr_map = False # calculate only SNR map from the first volume by pseudo-replicas
n_replica = 50 # number of replicas used for SNR map calculation

reco_n_contr = 0 # if >0 only the volumes up to the specified number will be reconstructed
first_vol = 0 # index of first volume, that is reconstructed

########################
# Main Function
########################

def process(connection, config, metadata):
    
    # -- Some manual parameters --- #

    global snr_map
    global reco_n_contr
    if snr_map:
        logging.debug("Calculate SNR maps for first contrast.")
        reco_n_contr = 1
    
   # ----------------------------- #

    # reload reco_helper
    importlib.reload(rh)

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Check if ACS and field map available
    if not os.path.isfile(os.path.join(dependencyFolder, "acs.npy")):
        raise ValueError("No reference data for coil sensitivity mapping available.")
    if not os.path.isfile(os.path.join(dependencyFolder, "fmap.npz")):
        raise ValueError("No field map available.")

    # Read user parameters
    up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}
    up_long = {item.name: item.value for item in metadata.userParameters.userParameterLong}

    # Check SMS, in the 3D case we can have an acceleration factor, but its not SMS
    sms_factor = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2)

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
    n_slc_red = n_slc//sms_factor
    n_contr = reco_n_contr if reco_n_contr else metadata.encoding[0].encodingLimits.contrast.maximum + 1

    acqGroup = [[[] for _ in range(n_contr)] for _ in range(n_slc_red)]
    noiseGroup = []
    img_coord = [None] * (n_slc_red)
    dmtx = None

    # chronological slice order
    if "chronSliceIndex1" in up_long:
        slc_ix1 = up_long["chronSliceIndex1"]
        slc_ix2 = up_long["chronSliceIndex2"]
        if slc_ix2 - slc_ix1 == 2:
            # interleaved
            slc_chron_tmp = np.arange(0,n_slc_red,1)
            slc_chron = np.concatenate([slc_chron_tmp[slc_ix1::2], slc_chron_tmp[::2]])
        elif slc_ix1 == 0:
            # ascending
            slc_chron = np.arange(0,n_slc_red,1)
        else:
            # descending
            slc_chron = np.arange(0,n_slc_red,1)[::-1]
    else:
        # assume interleaved
        slc_chron_tmp = np.arange(0,n_slc_red,1)
        slc_chron = np.concatenate([slc_chron_tmp[1::2], slc_chron_tmp[::2]])

    try:
        for item in connection:

            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                item.idx.slice = slc_chron[item.idx.slice]

                # run noise decorrelation
                if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                    noiseGroup.append(item)
                    continue
                elif len(noiseGroup) > 0 and dmtx is None:
                    noise_data = []
                    for acq in noiseGroup:
                        noise_data.append(rh.remove_os(acq.data, axis=1))
                    noise_data = np.concatenate(noise_data, axis=1)
                    # calculate pre-whitening matrix
                    dmtx = rh.calculate_prewhitening(noise_data)
                    del(noise_data)
                    noiseGroup.clear()
                               
                # Skip these flags (shouldnt be included in the raw data anyway)
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA) or item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    continue

                # trigger recon early
                if reco_n_contr and item.idx.contrast > reco_n_contr + first_vol - 1:
                    if len(acqGroup) > 0:
                        process_and_send(connection, acqGroup, metadata, img_coord)
                    continue
                if item.idx.contrast < first_vol:
                    continue # skip until first volume index
                else:
                    item.idx.contrast -= first_vol

                # Process Imaging Scans
                # Calculate image coordinates in DCS
                if img_coord[item.idx.slice] is None:
                    img_coord[item.idx.slice] = rh.calc_img_coord(metadata, item, pulseq=False) 

                # Noise whitening
                if dmtx is not None:
                    item.data[:] = rh.apply_prewhitening(item.data[:], dmtx)

                # Undo FOV shift - this is only necessary, if the FOV shift was not disabled in the sequence
                # shift = rh.pcs_to_dcs(np.asarray(item.position)) * 1e-3 # shift [m] in DCS
                # item.data[:] *= np.exp(1j*(shift*item.traj[:,:3]).sum(axis=-1)) # base_trj in [rad/m]

                # invert trajectory sign (is necessary as field map and k0 also need sign change)
                # WIP: for some unknown reason, not inverting the sign for concomitant field terms (item.traj[:,:-4] *= -1) yields better results in vivo
                # In phantoms the results are better when using the same sign as for the spherical harmonics (as expected)
                item.traj[:] *= -1
                item.traj[:,3] *= -1 # dont invert time vector

                # below is undoing the inverted sign for concomitant fields (works better in vivo)
                if item.traj.shape[1] > 4:
                    item.traj[:,-4:] *= -1

                # T2* filter
                if freq < 2e8:
                    t2_star = 70e-3 # 3T
                else:
                    t2_star = 40e-3 # 7T
                t_vec = item.traj[:,3]
                item.data[:] *= 1/np.exp(-t_vec/t2_star)

                # append item
                acqGroup[item.idx.slice][item.idx.contrast].append(item)
                    
                # Process acquisitions with PowerGrid - full recon
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
                    process_and_send(connection, acqGroup, metadata, img_coord)

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

def process_and_send(connection, acqGroup, metadata, img_coord):
    # Start data processing
    logging.info("Processing a group of k-space data")
    images = process_raw(acqGroup, metadata, img_coord)
    logging.debug("Sending images to client.")
    connection.send_image(images)
    acqGroup.clear()

def process_raw(acqGroup, metadata, img_coord):

    # Get some header info
    sms_factor = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2)
    nx = metadata.encoding[0].reconSpace.matrixSize.x
    ny = metadata.encoding[0].reconSpace.matrixSize.y

    # Write ISMRMRD file for PowerGrid
    tmp_file = dependencyFolder+"/PowerGrid_tmpfile.h5"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    dset_tmp = ismrmrd.Dataset(tmp_file, create_if_needed=True)

    # Insert Coordinates
    img_coord = np.asarray(img_coord) # [n_slc, 3, nx, ny, nz]
    img_coord = np.transpose(img_coord, [1,0,4,3,2]) # [3, n_slc, nz, ny, nx]
    dset_tmp.append_array("ImgCoord", img_coord.astype(np.float64))

    # Load ACS and calculate reference image
    acs = np.load(os.path.join(dependencyFolder, "acs.npy")) # [slices,nx,ny,nz,nc]
    if acs.shape[-2] != 1:
        raise ValueError("3D reference scan not supported.")
    acs = bart(1,f'resize -c 1 {nx} 2 {ny} ', acs)
    refimg = rh.rss(cifftn(acs, [1,2]), axis=-1) # save at spiral matrix size
    refimg = np.transpose(refimg, [0,3,2,1])[::-1,:,::-1]

    # Calculate and insert Sensitivity Maps from ACS data
    sens_path = os.path.join(dependencyFolder, "sensmaps.npy")
    if read_ecalib and os.path.exists(sens_path):
        sens = np.load(sens_path)
    else:
        logging.debug("Start sensitivity map calculation.")
        sensmaps = []
        for slc, acs_slc in enumerate(acs):
            sensmaps.append(rh.ecalib(acs_slc, chunk_sz=0, n_maps=1, crop=0.92, kernel_size=6, threshold=0.003, use_gpu=False))
            # sensmaps.append(bart(1, 'caldir 32', acs_slc))
            logging.debug(f"Finished sensitivity map calculation for slice {slc}.")
        sens = np.transpose(np.array(sensmaps), [0,4,3,2,1]) # [slices,nc,nz,ny,nx]
        sens = sens[::-1,...,::-1,:] # refscan is with Pulseq and has different orientation

    np.save(sens_path, sens)
    if sms_factor > 1:
        sens = reshape_sens_sms(sens, sms_factor)
    dset_tmp.append_array("SENSEMap", sens.astype(np.complex128))

    # Insert Field Map
    fmap = np.load(os.path.join(dependencyFolder, "fmap.npz"), allow_pickle=True)
    fmap_data = fmap['fmap']
    fmap_name = fmap['name']
    shape = [fmap_data.shape[0], ny, nx] # [slices,ny,nx]
    fmap_data = resize(fmap_data, shape, anti_aliasing=True) # interpolate to correct raster
    fmap_data = np.swapaxes(fmap_data, -1, -2)[::-1,::-1] # refscan is with Pulseq and has different orientation
    fmap_data_sv = fmap_data.copy()
    if sms_factor > 1:
        fmap_data = reshape_fmap_sms(fmap_data, sms_factor)  # [slices,nz,ny,nx], nz=sms factor
    dset_tmp.append_array('FieldMap', fmap_data.astype(np.float64))

    # Write header
    if sms_factor > 1:
        metadata.encoding[0].encodedSpace.matrixSize.z = sms_factor
        metadata.encoding[0].encodingLimits.slice.maximum = int((metadata.encoding[0].encodingLimits.slice.maximum + 1) / sms_factor + 0.5) - 1
    if reco_n_contr:
        metadata.encoding[0].encodingLimits.contrast.maximum = reco_n_contr - 1
        metadata.encoding[0].encodingLimits.repetition.maximum = 0
    dset_tmp.write_xml_header(metadata.toXML())

    # Insert acquisitions
    for slc in acqGroup:
        for contr in slc:
            for acq in contr:
                dset_tmp.append_acquisition(acq)
    dset_tmp.close()

    # Define in- and output for PowerGrid
    pg_dir = dependencyFolder+"/powergrid_results"
    if not os.path.exists(pg_dir):
        os.makedirs(pg_dir)
    if os.path.exists(pg_dir+"/images_pg.npy"):
        os.remove(pg_dir+"/images_pg.npy")

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
    regu = 'QUAD' # regularization (QUAD - quadratic or TV - total variation)
    pg_opts = f'-i {tmp_file} -o {pg_dir} -n 20 -B 500 -D 2 -e 0.0001 -R {regu}'
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
    dsets = {}
    dsets['data'] = data.copy()

    # Append refscan
    dsets['refimg'] = refimg.copy()

    # Append field map
    dsets['fmap'] = fmap_data_sv[:,np.newaxis] /2/np.pi # add axis for [slc,z,y,x], save in [Hz]

    # Calculate SNR maps based on pseudo-replicas
    # based on https://github.com/hansenms/ismrm_sunrise_matlab/blob/master/ismrm_pseudo_replica.m
    if snr_map:
        # save acquisitions
        acqs_save = []
        dset_tmp = ismrmrd.Dataset(tmp_file, create_if_needed=False)
        for j in range(dset_tmp.number_of_acquisitions()):
            acqs_save.append(dset_tmp.read_acquisition(j).data[:].copy())
        dset_tmp.close()
        data_snr_list = []

        # reconstruct replicas by adding gaussian noise with sigma=1
        for k in range(n_replica):
            logging.debug(f"Reconstruct replica volume {k+1} of {n_replica}")
            dset_tmp = ismrmrd.Dataset(tmp_file, create_if_needed=False)
            acq = dset_tmp.read_acquisition(0)
            for j in range(dset_tmp.number_of_acquisitions()):
                noise = np.random.randn(np.prod(acq.data.shape)).reshape(acq.data.shape) + 1j* np.random.randn(np.prod(acq.data.shape)).reshape(acq.data.shape)
                acq = dset_tmp.read_acquisition(j)
                acq.data[:] = acqs_save[j] + noise
                dset_tmp.write_acquisition(acq, j)
            dset_tmp.close()
            process = subprocess.run(subproc, shell=True, check=True, text=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            data_snr = abs(np.load(pg_dir + "/images_pg.npy"))
            data_snr = np.transpose(data_snr, [3,4,2,1,0,5,6,7]).mean(axis=0)
            data_snr_list.append(data_snr.reshape(newshape, order='f'))
        data_snr = np.array(data_snr_list)

        # calculate SNR maps
        std_dev = np.std(data_snr + np.max(data_snr), axis=0)
        snr = np.divide(abs(data), std_dev, where=std_dev!=0, out=np.zeros_like(std_dev))
        snr = snr[0,0,0]
        dsets['snr_map'] = snr

    # Correct orientation, normalize and convert to int16 for online recon - WIP: correct for EPI?
    for key in dsets:
        dsets[key] = np.flip(dsets[key], -1)
        if key == 'data' and save_cmplx:
            dsets[key] /= abs(dsets[key]).max()
        elif key == 'fmap':
            dsets[key] = np.around(dsets[key])
            dsets[key] = dsets[key].astype(np.int16) # field map

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                        'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                        'Keep_image_geometry':    1,
                        'PG_Options':              subproc,
                        'Field Map':               fmap_name})
    # Set ISMRMRD Meta Attributes
    meta2 = ismrmrd.Meta({'DataRole':              'Quantitative',
                        'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                        'Keep_image_geometry':    1,
                        'PG_Options':              subproc,
                        'Field Map':               fmap_name})

    # Calculate affine matrix - WIP: currently only pure transversal orientation supported
    res_x = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x
    res_y = metadata.encoding[0].encodedSpace.fieldOfView_mm.y / metadata.encoding[0].encodedSpace.matrixSize.y
    slc_res = metadata.encoding[0].encodedSpace.fieldOfView_mm.z
    res = [-1*res_x, res_y, slc_res, 0]
    edge_coord = 1e3 * np.array([img_coord[0,0,0,0,0], img_coord[1,0,0,0,0], img_coord[2,0,0,0,0]]) # [slc=0,nz=0,nx=0,ny=0] for edge at right/posterior/feet in transversal orientation
    affine = np.diag(res)
    affine[:3,-1] = rh.dcs_to_ras(edge_coord)
    np.save(debugFolder+"/affine.npy", affine)

    # Send images
    img_ix = 0
    for series_ix, key in enumerate(dsets):
        # Format as 2D ISMRMRD image data [nx,ny]
        if key == 'data':
            meta['ImgType'] = 'Imgdata'
            for contr in range(dsets[key].shape[1]):
                for phs in range(dsets[key].shape[2]):
                    for rep in range(dsets[key].shape[0]): # save one repetition after another
                        for slc in range(dsets[key].shape[3]):
                            img_ix += 1
                            for nz in range(dsets[key].shape[4]):
                                image = ismrmrd.Image.from_array(dsets[key][rep,contr,phs,slc,nz], acquisition=acqGroup[0][contr][0])
                                meta['ImageRowDir'] = ["{:.18f}".format(acqGroup[0][0][0].read_dir[0]), "{:.18f}".format(acqGroup[0][0][0].read_dir[1]), "{:.18f}".format(acqGroup[0][0][0].read_dir[2])]
                                meta['ImageColumnDir'] = ["{:.18f}".format(acqGroup[0][0][0].phase_dir[0]), "{:.18f}".format(acqGroup[0][0][0].phase_dir[1]), "{:.18f}".format(acqGroup[0][0][0].phase_dir[2])]
                                image.image_index = img_ix
                                image.image_series_index = series_ix
                                image.slice = slc
                                image.repetition = rep
                                image.phase = phs
                                image.contrast = contr
                                image.user_int[1] = sms_factor
                                if img_ix <= len(affine):
                                    image.user_int[2] = 1 # indicate affine is set
                                    image.user_float[3:7] = affine[img_ix-1]
                                image.attribute_string = meta.serialize()
                                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                                       ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                                       ctypes.c_float(slc_res))
                                images.append(image)
        else:
            # Fmap & SNR maps
            for slc, img in enumerate(dsets[key]):
                image = ismrmrd.Image.from_array(img[0], acquisition=acqGroup[0][0][0])
                image.image_index = slc + 1
                image.image_series_index = series_ix
                image.slice = slc
                if series_ix != len(dsets) - 1:
                    meta['ImgType'] = key
                    image.attribute_string = meta.serialize()
                else:
                    meta2['ImgType'] = key
                    image.attribute_string = meta2.serialize()
                    image.image_type = ismrmrd.IMTYPE_REAL
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                       ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                       ctypes.c_float(slc_res))
                images.append(image)

    logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(meta.serialize()).toprettyxml())
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images
    
# %%
#########################
# Helper
#########################

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
