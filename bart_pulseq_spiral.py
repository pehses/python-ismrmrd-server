
import ismrmrd
import os
import logging
import numpy as np
import ctypes

from bart import bart
from cfft import cfftn, cifftn
from pulseq_helper import insert_hdr, insert_acq, get_ismrmrd_arrays, read_acqs
import reco_helper as rh

""" Reconstruction of imaging data acquired with the Pulseq Sequence via the FIRE framework
    Reconstruction is done with the BART toolbox

"""


# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

parallel_reco = True # parallel reconstruction of slices/contrasts
save_complex = False

snr_map = False # calculate SNR map (only if parallel_reco is True)
n_replica = 50 # number of replicas

compressed_coils = None

########################
# Main Function
########################

def process_spiral(connection, config, metadata, prot_file):
  
    # Check protocol arrays to process with possible subscript
    prot_arrays = get_ismrmrd_arrays(prot_file)
    if "dream" in prot_arrays:
        import bart_pulseq_spiral_dream
        import importlib
        importlib.reload(bart_pulseq_spiral_dream)
        bart_pulseq_spiral_dream.process_spiral_dream(connection, config, metadata, prot_file)
        return

    # -- Some manual parameters --- #
    
    # Select a slice (only for debugging purposes) - if "None" reconstruct all slices
    slc_sel = None
    up_long = {item.name: item.value for item in metadata.userParameters.userParameterLong}
    if 'recon_slice' in up_long:
        online_slc = up_long['recon_slice'] # Only online reco can send single slice number (different xml)
        if online_slc >= 0:
            slc_sel = int(online_slc)

    # Coil Compression
    n_cha = metadata.acquisitionSystemInformation.receiverChannels
    global compressed_coils
    if compressed_coils is not None:
        if compressed_coils > 0 and compressed_coils<=n_cha:
            cc_cha = compressed_coils
            logging.debug(f'Coil Compression from {n_cha} to {cc_cha} channels.')
        else:
            cc_cha = n_cha
            logging.debug('Invalid number of compressed coils. Set back to original number of coils.')
    else:
        cc_cha = n_cha

    # ----------------------------- #

    # Insert protocol header
    insert_hdr(prot_file, metadata)
    
    logging.info("Config: \n%s", config)

    # Check for GPU availability
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        gpu = True
    else:
        gpu = False

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier

    try:
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("Trajectory type '%s', matrix size (%s x %s x %s), field of view (%s x %s x %s)mm^3", 
            metadata.encoding[0].trajectory.value, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # user parameters
    up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}
    up_base = {item.name: item.value for item in metadata.userParameters.userParameterBase64}

    # # Initialize lists for datasets
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1

    sms_factor = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2) if metadata.encoding[0].encodingLimits.slice.maximum > 0 else 1
    if sms_factor == 0:
        sms_factor = 1
    if sms_factor > 1:
        global parallel_reco
        parallel_reco = True

    acqGroup = [[[] for _ in range(n_slc//sms_factor)] for _ in range(n_contr)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    acs = [None] * n_slc
    sensmaps = None
    dmtx = None

    process_raw.imascale = None
    process_raw.refimg = [None] * n_slc
    
    # compression matrix
    process_raw.cc_mat = [None] * n_slc

    # parameters for reapplying FOV shift
    nsegments = metadata.encoding[0].encodingLimits.segment.maximum + 1
    matr_sz = np.array([metadata.encoding[0].encodedSpace.matrixSize.x, metadata.encoding[0].encodedSpace.matrixSize.y])
    res = np.array([metadata.encoding[0].encodedSpace.fieldOfView_mm.x / matr_sz[0], metadata.encoding[0].encodedSpace.fieldOfView_mm.y / matr_sz[1], 1])

    base_trj = None

    # Oversampling factor
    os_factor = up_double["os_factor"] if "os_factor" in up_double else 1

    # read protocol acquisitions - faster than doing it one by one
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
                    dmtx = rh.calculate_prewhitening(noise_data, scale_factor=os_factor, os_removed=False)
                    del(noise_data)
                    noiseGroup.clear()
                    
                # skip slices in single slice reconstruction
                if slc_sel is not None and item.idx.slice != slc_sel:
                    continue
                
                # Accumulate all imaging readouts in a group
                if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # skope sync scans
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    if item.idx.contrast == 0: # if B0 mapping refscan was done, use only 1st contrast
                        acsGroup[item.idx.slice].append(item)
                    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                        # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                        acs[item.idx.slice] = process_acs(acsGroup[item.idx.slice], metadata, cc_cha, dmtx, gpu)
                        acsGroup[item.idx.slice].clear()
                    continue
                if acs[0] is not None and sensmaps is None:
                    # ESPIRiT calibration
                    use_gpu_sens = False
                    acs = np.moveaxis(np.asarray(acs),0,-1) # move slices to last dim
                    if gpu and acs.shape[2] > 1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
                        use_gpu_sens = True
                    sensmaps = rh.ecalib(acs, n_maps=1, kernel_size=6, use_gpu=use_gpu_sens)
                    sensmaps = np.moveaxis(sensmaps,-1,0) # move slices back to first dim
                    if sms_factor > 1:
                        sensmaps = reshape_sens_sms(sensmaps, sms_factor)

                if item.idx.segment == 0:
                    acqGroup[item.idx.contrast][item.idx.slice].append(item)

                    # for reapplying FOV shift (see below)
                    pred_trj = item.traj[:]
                    rotmat = rh.calc_rotmat(item)
                    shift = rh.pcs_to_gcs(np.asarray(item.position), rotmat) / res
                else:
                    # append data to first segment of ADC group
                    idx_lower = item.idx.segment * item.number_of_samples
                    idx_upper = (item.idx.segment+1) * item.number_of_samples
                    acqGroup[item.idx.contrast][item.idx.slice][-1].data[:,idx_lower:idx_upper] = item.data[:]
                if item.idx.segment == nsegments - 1:
                    # Reapply FOV Shift with predicted trajectory
                    last_item = acqGroup[item.idx.contrast][item.idx.slice][-1]
                    if "spiral_nopos" in up_base and up_base["spiral_nopos"] == "1":
                        base_trj = None
                    last_item.data[:] = rh.fov_shift_spiral_reapply(last_item.data[:], pred_trj, base_trj, shift, matr_sz)

                    # remove oversampling
                    if os_factor == 2:
                        rh.remove_os_spiral(last_item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if (item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION)) and not parallel_reco:
                    logging.info("Processing a group of k-space data")
                    if sensmaps is None:
                        sensmaps_slc = None
                    else:
                        sensmaps_slc = sensmaps[item.idx.slice]
                    images = process_raw(acqGroup[item.idx.contrast][item.idx.slice], metadata, cc_cha, dmtx, sensmaps_slc, gpu)
                    logging.debug("Sending images to client:\n%s", images)
                    connection.send_image(images)
                    acqGroup[item.idx.contrast][item.idx.slice].clear() # free memory

                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT) and parallel_reco:
                    logging.info("Process slices/constrasts in parallel.")
                    images = process_raw(acqGroup, metadata, cc_cha, dmtx, sensmaps, gpu, parallel=True)
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
            if len(acqGroup[item.idx.contrast][item.idx.slice]) > 0:
                logging.info("Processing a group of k-space data (untriggered)")
                if sensmaps[item.idx.slice] is None:
                    # run parallel imaging calibration
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], metadata, dmtx) 
                image = process_raw(acqGroup[item.idx.contrast][item.idx.slice], metadata, cc_cha, dmtx, sensmaps[item.idx.slice])
                logging.debug("Sending image to client:\n%s", image)
                connection.send_image(image)
                acqGroup = []

    finally:
        connection.send_close()

# %%
#########################
# Process Data
#########################

def process_raw(group, metadata, cc_cha, dmtx=None, sensmaps=None, gpu=False, parallel=False):

    force_pics = False # force PICS reconstruction (only if parallel=False)
    adjoint_nufft = False # do adjoint nufft instead of inverse nufft (only if parallel=False)

    up_base = {item.name: item.value for item in metadata.userParameters.userParameterBase64}
    up_double = {item.name: item.value for item in metadata.userParameters.userParameterDouble}

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    
    rNx = metadata.encoding[0].reconSpace.matrixSize.x
    rNy = metadata.encoding[0].reconSpace.matrixSize.y
    rNz = metadata.encoding[0].reconSpace.matrixSize.z

    n_cha = metadata.acquisitionSystemInformation.receiverChannels
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    sms_factor = int(metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2) if metadata.encoding[0].encodingLimits.slice.maximum > 0 else 1
    slc_res = metadata.encoding[0].encodedSpace.fieldOfView_mm.z

    scale_fac_pics = 1500
    if gpu and nz>1: # only use GPU for 3D data, as otherwise the overhead makes it slower than CPU
        nufft_config = 'nufft -g -i -m 15 -l 0.005 -t -d %d:%d:%d'%(nx, nx, nz)
        ecalib_config = 'ecalib -g -m 1 -I'
        pics_config = f'pics -g -S -e -l1 -r 0.001 -i 50'
    else:
        nufft_config = 'nufft -i -m 15 -l 0.005 -t -d %d:%d:%d'%(nx, nx, nz)
        ecalib_config = 'ecalib -m 1 -I'
        pics_config = f'pics -S -e -l1 -r 0.001 -i 50'
    if scale_fac_pics is not None:
        pics_config += f' -w {scale_fac_pics}'

    if parallel:
        ksp = []
        traj = []
        for contr_acq in group:
            for slc_acq in contr_acq:
                data, trj = sort_spiral_data(slc_acq, dmtx)
                traj.append(trj)

                slc = slc_acq[0].idx.slice
                if cc_cha < n_cha and process_raw.cc_mat[slc] is None:
                    logging.debug(f'Calculate coil compression matrix.')
                    process_raw.cc_mat[slc] = bart(1, f'cc -A -M -S -p {cc_cha}', data) # SVD based Coil compression        

                if process_raw.cc_mat[slc] is not None:
                    logging.debug(f'Perform Coil Compression to {cc_cha} channels.')
                    data = bart(1, f'ccapply -S -p {cc_cha}', data, process_raw.cc_mat[slc])

                ksp.append(data)

        # move slices to last dim
        ksp = np.moveaxis(np.asarray(ksp), 0, -1)
        traj = np.moveaxis(np.asarray(traj)[...,np.newaxis], 0, -1)

        snr = None
        if sensmaps is None:
            logging.debug("Do inverse nufft")
            nufft_config = f'--parallel-loop {2**(ksp.ndim-1)} -e {ksp.shape[-1]} ' + nufft_config
            data = bart(1, nufft_config, traj, ksp) # iterative inverse nufft

            if data.ndim == 4:
                data = data[..., np.newaxis]

            # Sum of squares coil combination
            data = np.sqrt(np.sum(np.abs(data)**2, axis=-2))
        else:
            sensmaps = np.moveaxis(np.asarray(sensmaps), 0, -1)
            if sensmaps.ndim == 6:
                #  add empty maps dimension if more than one ecalib map is used
                ksp = ksp[..., np.newaxis, :]
                traj = traj[..., np.newaxis, :]
            pics_config = f'--parallel-loop {2**(ksp.ndim-1)} -e {ksp.shape[-1]} ' + pics_config
            if "slice_profile_meas" in up_base:
                sensmaps = np.repeat(sensmaps, nz, axis=-3)
            if sms_factor > 1:
                logging.info("BART SMS reconstruction.")
                sms_dim = 13
                ksp = rh.add_naxes(ksp, sms_dim+1-ksp.ndim)
                for s in range(sms_factor-1):
                    ksp = np.concatenate((ksp, np.zeros_like(ksp[...,0,np.newaxis])),axis=sms_dim)
                sensmaps = rh.add_naxes(sensmaps, sms_dim+1-sensmaps.ndim) # first dim is sms_dim
                sensmaps = np.moveaxis(sensmaps,0,sms_dim)
                pat = bart(1, 'pattern', ksp)
                pics_config += " -M"
                data = bart(1, pics_config, ksp, sensmaps, t=traj, p=pat)
                if "slice_profile_meas" in up_base:
                    # upper mb slices need FOV shift, which is the slice distance in voxel
                    res_z = up_double["res_z"] if "res_z" in up_double else 1 # voxel size in z
                    shift = n_slc // sms_factor * slc_res / res_z
                    data[...,1] = rh.fov_shift_img_axis(data[...,1], shift, axis=2)
                data = data.reshape(data.shape[:4] + (data.shape[4]*data.shape[sms_dim],), order='f') # merge slice and sms dim
            else:
                data = bart(1, pics_config, ksp, sensmaps, t=traj)
            if data.ndim == 6:
                data = data[...,0,:]
            if not save_complex:
                data = np.abs(data)
        
        if snr_map and not "slice_profile_meas" in up_base:
            data_snr = []
            for k in range(n_replica):
                logging.debug(f"Reconstruct replica {k+1} of {n_replica}.")
                noise = np.random.randn(np.prod(ksp.shape)).reshape(ksp.shape) + 1j* np.random.randn(np.prod(ksp.shape)).reshape(ksp.shape)
                ksp_noise = ksp + noise
                if sms_factor > 1:
                    img_snr = bart(1, pics_config, ksp_noise, sensmaps, t=traj, p=pat)
                    img_snr = img_snr.reshape(img_snr.shape[:4] + (img_snr.shape[4]*img_snr.shape[sms_dim],), order='f')
                    data_snr.append(img_snr)
                else:
                    data_snr.append(bart(1, pics_config, ksp_noise, sensmaps, t=traj))
            data_snr = np.asarray(data_snr)
            std_dev = np.std(np.abs(data_snr + np.max(np.abs(data_snr))), axis=0)
            snr = np.divide(np.abs(data), std_dev, where=std_dev!=0, out=np.zeros_like(std_dev))
            snr = np.swapaxes(snr, 0, 1)
            snr = np.flip(snr, (0,1,2))
            if snr.ndim == 3:
                snr = snr[..., np.newaxis, np.newaxis]
            snr = np.moveaxis(snr, -1, 0)[...,0]
            newshape = [n_contr, n_slc] + list(snr.shape[1:])
            snr = snr.reshape(newshape)

        if nz > rNz:
            # remove oversampling in slice direction
            data = data[:,:,(nz - rNz)//2:-(nz - rNz)//2]

        logging.debug("Image data is size %s" % (data.shape,))
        np.save(debugFolder + "/" + "img.npy", data)

        # correct orientation at scanner (consistent with ICE)
        data = np.swapaxes(data, 0, 1)
        data = np.flip(data, (0,1,2))
        if data.ndim == 3: # if only 1 slice, BART removes the last 2 dims
            data = data[..., np.newaxis, np.newaxis]
        data = np.moveaxis(data, -1, 0)[...,0] # [slc,x,y,z]
        newshape = [n_contr, n_slc] + list(data.shape[1:])
        data = data.reshape(newshape)

        # Normalize and convert to int16
        # save one scaling in 'static' variable
        if not save_complex:
            if process_raw.imascale is None:
                process_raw.imascale = 0.8 / data.max()
            data *= 32767 * process_raw.imascale
            data = np.around(data)
            data = data.astype(np.int16)

        # Set ISMRMRD Meta Attributes
        meta = ismrmrd.Meta({'DataRole':               'Image',
                            'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                            'WindowCenter':           '16384',
                            'WindowWidth':            '32768',
                            'Keep_image_geometry':    1,
                            'ImgType':                'Imgdata'})
        xml = meta.serialize()
        
        images = []
        n_par = data.shape[-1]
        rotmat = rh.calc_rotmat(group[0][0][0])

        for k,contr in enumerate(data):
            for slc,img in enumerate(contr):
                if n_par > 1:
                    image = ismrmrd.Image.from_array(img, acquisition=group[0][0][0])
                else:
                    image = ismrmrd.Image.from_array(img[...,0], acquisition=group[0][0][0])
                    offset = [0, 0, slc_res*(slc-(n_slc-1)/2)] # slice offset in GCS
                    image.position[:] += rh.gcs_to_pcs(offset, rotmat) # correct image position in PCS
                image.image_index = slc
                image.image_series_index = k
                image.slice = slc
                image.attribute_string = xml
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
                images.append(image)

        for j in range(n_slc):
            if process_raw.refimg[j] is not None:
                meta['ImgType'] = 'refimg'
                xml = meta.serialize()
                refimg = np.swapaxes(process_raw.refimg[j], 0, 1)
                refimg = np.flip(refimg, (0,1,2))
                refimg *= 32767 / np.max(refimg)
                refimg = np.around(refimg)
                refimg = refimg.astype(np.int16)
                
                image = ismrmrd.Image.from_array(refimg, acquisition=group[0][0][0])
                image.image_index = j
                image.image_series_index = n_contr
                image.attribute_string = xml
                image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                    ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
                images.append(image)

        if snr is not None:
            meta['ImgType'] = 'snr_map'
            xml = meta.serialize()
            for k,contr in enumerate(snr):
                for slc,img in enumerate(contr):
                    if n_par > 1:
                        image = ismrmrd.Image.from_array(img, acquisition=group[0][0][0])
                    else:
                        image = ismrmrd.Image.from_array(img[...,0], acquisition=group[0][0][0])
                        offset = [0, 0, slc_res*(slc-(n_slc-1)/2)] # slice offset in GCS
                        image.position[:] += rh.gcs_to_pcs(offset, rotmat) # correct image position in PCS
                    image.image_index = slc
                    image.image_series_index = n_contr + 1
                    image.slice = slc
                    image.attribute_string = xml
                    image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                        ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                        ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
                    images.append(image)

        return images
    
    else:
        data, trj = sort_spiral_data(group, dmtx)
        
        slc = group[0].idx.slice
        if cc_cha < n_cha and process_raw.cc_mat[slc] is None:
            logging.debug(f'Calculate coil compression matrix.')
            process_raw.cc_mat[slc] = bart(1, f'cc -A -M -S -p {cc_cha}', data) # SVD based Coil compression        

        if process_raw.cc_mat[slc] is not None:
            logging.debug(f'Perform Coil Compression to {cc_cha} channels.')
            data = bart(1, f'ccapply -S -p {cc_cha}', data, process_raw.cc_mat[slc])

        if sensmaps is None and force_pics:
            sensmaps = bart(1, nufft_config, trj, data) # nufft
            sensmaps = cfftn(sensmaps, [0, 1, 2]) # back to k-space
            sensmaps = bart(1, ecalib_config, sensmaps)  # ESPIRiT calibration

        if sensmaps is None:       
            if adjoint_nufft:
                logging.debug("Do adjoint nufft")
                # calculate and apply dcf
                dcf = rh.calc_dcf(np.swapaxes(trj[...,0],0,1))
                dcf /= np.max(abs(dcf))
                data *= dcf[np.newaxis,:,np.newaxis,np.newaxis]

                data = bart(1, 'nufft -a', trj, data) # adjoint nufft
            else:
                logging.debug("Do inverse nufft")
                data = bart(1, nufft_config, trj, data) # iterative inverse nufft

            # Sum of squares coil combination
            data = np.sqrt(np.sum(np.abs(data)**2, axis=-1))
        else:
            if "slice_profile_meas" in up_base:
                sensmaps = np.repeat(sensmaps, nz, axis=-2)
            data = bart(1, pics_config, data, sensmaps, t=trj)
        if not save_complex:    
            data = np.abs(data)
        
        # make sure that data is 3d
        while np.ndim(data) < 3:
            data = data[..., np.newaxis]
        data = data[(slice(None),) * 3 + (data.ndim-3) * (0,)]  # select first 3 dims

        if nz > rNz:
            # remove oversampling in slice direction
            data = data[:,:,(nz - rNz)//2:-(nz - rNz)//2]

        logging.debug("Image data is size %s" % (data.shape,))

        # correct orientation at scanner (consistent with ICE)
        data = np.swapaxes(data, 0, 1)
        data = np.flip(data, (0,1,2))

        # Normalize and convert to int16
        # save one scaling in 'static' variable
        if not save_complex:
            if process_raw.imascale is None:
                process_raw.imascale = 0.8 / data.max()
            data *= 32767 * process_raw.imascale
            data = np.around(data)
            data = data.astype(np.int16)

        # Set ISMRMRD Meta Attributes
        meta = ismrmrd.Meta({'DataRole':               'Image',
                            'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                            'WindowCenter':           '16384',
                            'WindowWidth':            '32768',
                            'Keep_image_geometry':    1,
                            'ImgType':                'Imgdata'})
        xml = meta.serialize()
        
        images = []
        n_par = data.shape[-1]
        n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1

        # Format as ISMRMRD image data
        if n_par > 1:
            image = ismrmrd.Image.from_array(data, acquisition=group[0])
        else:
            image = ismrmrd.Image.from_array(data[...,0], acquisition=group[0])
        image.image_index = group[0].idx.slice
        image.image_series_index = group[0].idx.contrast
        image.slice = group[0].idx.slice
        image.attribute_string = xml
        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        images.append(image)

    # send reference image if available
    if process_raw.refimg[group[0].idx.slice] is not None and group[0].idx.contrast == 0:
        meta['ImgType'] = 'refimg'
        xml = meta.serialize()
        refimg = np.swapaxes(process_raw.refimg[group[0].idx.slice], 0, 1)
        refimg = np.flip(refimg, (0,1,2))
        refimg *= 32767 / np.max(refimg)
        refimg = np.around(refimg)
        refimg = refimg.astype(np.int16)
        
        image = ismrmrd.Image.from_array(refimg, acquisition=group[0])
        image.image_index = group[0].idx.slice
        image.image_series_index = n_contr
        image.attribute_string = xml
        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        images.append(image)

    return images

def process_acs(group, metadata, cc_cha, dmtx=None, gpu=False):
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)

        n_cha = metadata.acquisitionSystemInformation.receiverChannels
        slc = group[0].idx.slice
        if cc_cha < n_cha:
            logging.debug(f'Calculate coil compression matrix.')
            process_raw.cc_mat[slc] = bart(1, f'cc -A -M -S -p {cc_cha}', data) # SVD based Coil compression        
            data = bart(1, f'ccapply -S -p {cc_cha}', data, process_raw.cc_mat[slc]) # SVD based Coil compression

        refimg = cifftn(data,axes=[0,1,2])
        refimg = np.sqrt(np.sum(np.abs(refimg)**2, axis=-1))
        process_raw.refimg[slc] = refimg

        np.save(debugFolder + "/" + "refimg.npy", refimg)

        np.save(debugFolder + "/" + "acs.npy", data)
        # np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return data
    else:
        return None

# %%
#########################
# Sort Data
#########################

def sort_spiral_data(group, dmtx=None):
    
    sig = list()
    trj = list()
    enc = list()
    for acq in group:

        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        enc.append([enc1, enc2])
        
        # append data after optional prewhitening
        if dmtx is None:
            sig.append(acq.data)
        else:
            sig.append(rh.apply_prewhitening(acq.data, dmtx))

        # update trajectory
        traj = np.swapaxes(acq.traj[:,:3],0,1) # [samples, dims] to [dims, samples]
        trj.append(traj)

    np.save(debugFolder + "/" + "enc.npy", enc)
    
    # convert lists to numpy arrays
    trj = np.asarray(trj) # current size: (nacq, 3, ncol)
    sig = np.asarray(sig) # current size: (nacq, ncha, ncol)

    # rearrange trj & sig for bart
    trj = np.transpose(trj, [1, 2, 0]) # [3, ncol, nacq]
    sig = np.transpose(sig, [2, 0, 1])[np.newaxis] # [1, ncol, nacq, ncha]
    logging.debug("Trajectory shape = %s , Signal Shape = %s "%(trj.shape, sig.shape))
    
    np.save(debugFolder + "/" + "trj.npy", trj)
    np.save(debugFolder + "/" + "raw.npy", sig)

    return sig, trj

def sort_into_kspace(group, metadata, dmtx=None, zf_around_center=False):
    # initialize k-space
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

        # Oversampling removal - WIP: assumes 2x oversampling at the moment
        data = rh.remove_os(acq.data[:], axis=-1)
        acq.resize(number_of_samples=data.shape[-1], active_channels=data.shape[0])
        acq.data[:] = data

    nc = metadata.acquisitionSystemInformation.receiverChannels
    nx = group[0].data.shape[-1]
    ny = enc1_max + 1
    nz = enc2_max + 1

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
            kspace[enc1, enc2, :, col] += rh.apply_prewhitening(acq.data, dmtx)
        counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc)
    kspace = np.transpose(kspace, [3, 0, 1, 2])

    # resize to spiral data size
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z

    up_base = {item.name: item.value for item in metadata.userParameters.userParameterBase64}
    if "slice_profile_meas" in up_base:
        kspace = bart(1,f'resize -c 0 {nx} 1 {ny} 2 {1}', kspace)
    else:
        kspace = bart(1,f'resize -c 0 {nx} 1 {ny} 2 {nz}', kspace)

    return kspace

def reshape_sens_sms(sens, sms_factor):
    # reshape sensmaps array for sms imaging, sensmaps for one acquisition are stored at nz
    sens_cpy = sens.copy() # [slices, nx, ny, nz, nc]
    slices_eff = sens_cpy.shape[0]//sms_factor
    sens = np.zeros([slices_eff, sms_factor, sens_cpy.shape[1] , sens_cpy.shape[2], sens_cpy.shape[3], sens_cpy.shape[4]], dtype=sens_cpy.dtype)
    for slc in range(sens_cpy.shape[0]):
        sens[slc%slices_eff,slc//slices_eff] = sens_cpy[slc] 
    return sens
