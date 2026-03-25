import ismrmrd
import os
import logging
import traceback
import numpy as np
import xml.dom.minidom
import base64
import constants

import ants
import antspynet

# Folder for debug output files
debugFolder = "/tmp/share/debug"


def _get_sequence_description(image: ismrmrd.Image) -> str:
    try:
        meta = ismrmrd.Meta.deserialize(image.attribute_string)
        return str(meta.get('SequenceDescription', ''))
    except Exception:
        return ''

def _config_with_ants_none(config):
    if not isinstance(config, dict):
        logging.info("Received non-dict config; creating default config dict")
        config_override = {}
    else:
        config_override = dict(config)

    parameters_raw = config_override.get('parameters', {})
    if not isinstance(parameters_raw, dict):
        logging.info("Config 'parameters' is not a dict; creating default parameters dict")
        parameters = {}
    else:
        parameters = dict(parameters_raw)

    parameters['ANTsConfig'] = 'None'
    parameters['BrainMaskConfig'] = 'None'
    config_override['parameters'] = parameters
    return config_override

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
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

    # Continuously parse incoming data parsed from MRD messages
    currentSeries = 0
    process_image.image_series_index_offset = 0
    imgGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                raise Exception("Raw k-space data is not supported by this module")

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # When this criteria is met, run process_group() on the accumulated
                # data, which returns images that are sent back to the client.
                # e.g. when the series number changes:
                if item.image_series_index != currentSeries:
                    logging.info("Processing a group of images because series index changed to %d", item.image_series_index)
                    currentSeries = item.image_series_index
                    config_for_group = config
                    if len(imgGroup) > 0:
                        previous_desc = _get_sequence_description(imgGroup[0])
                        current_desc = _get_sequence_description(item)
                        previous_base = previous_desc[:-3] if previous_desc.endswith('_ND') else ''

                        if (len(previous_base) > 0) and (current_desc == previous_base):
                            logging.info(
                                "Do not process non-distortion corrected images '%s' when distortion correction is active.",
                                previous_desc,
                            )
                            config_for_group = _config_with_ants_none(config)

                    image = process_image(imgGroup, config_for_group)
                    connection.send_image(image)

                    if len(imgGroup) > 0:
                        process_image.image_series_index_offset += 1
                        logging.info(
                            "Incremented image series offset between groups: %d",
                            process_image.image_series_index_offset,
                        )

                    imgGroup = []

                # Only process magnitude images -- send phase images back without modification (fallback for images with unknown type)
                if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()

                    connection.send_image(item)
                    continue

            elif item is None:
                break

            else:
                raise Exception("Unsupported data type %s", type(item).__name__)

        # Process any remaining groups of image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, config)
            connection.send_image(image)
            imgGroup = []

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())
        
        # Close connection without sending MRD_MESSAGE_CLOSE message to signal failure
        connection.shutdown_close()

    finally:
        try:
            connection.send_close()
        except:
            logging.error("Failed to send close message!")

class ImageFactory:

    def __init__(self) -> None:
        self.image_series_index_offset    : int       =  0
        self.ImageProcessingHistory       : list[str] = []
        self.SequenceDescriptionAdditional: list[str] = []
        self.mrdHeader                    : list[ismrmrd.ImageHeader]
        self.mrdMeta                      : list[ismrmrd.Meta]

    @staticmethod
    def MRD5Dto3D(data_mrd5D: np.array) -> np.array:

        # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
        data_mrd5D = data_mrd5D.transpose((3, 4, 2, 1, 0))

        logging.debug("Original image data is size %s" % (data_mrd5D.shape,))

        data_5d = data_mrd5D.astype(np.float64)

        # Reformat data from [y x z cha img] to [y x img]
        data_3d = data_5d[:,:,0,0,:]
        
        return data_3d
    
    def ANTsImageToMRD(self, ants_image: ants.ants_image.ANTsImage, history: str|list[str] = '', seq_descrip_add: str = '') -> list[ismrmrd.Image]:

        if   type(history) is list:
            self.ImageProcessingHistory += history
        elif type(history) is str and len(history)>0:
            self.ImageProcessingHistory.append(history)
        else:
            TypeError('bad `history` type')

        if len(seq_descrip_add)>0:
            self.image_series_index_offset += 1
            self.SequenceDescriptionAdditional = [seq_descrip_add]

        # Reformat data from [y x img] to [y x z cha img]
        data = ants_image.numpy()[:,:,np.newaxis,np.newaxis,:].astype(np.int16)

        # Re-slice back into 2D images
        imagesOut = [None] * data.shape[-1]
        for iImg in range(data.shape[-1]):

            # Create new MRD instance for the inverted image
            # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
            # from_array() should be called with 'transpose=False' to avoid warnings, and when called
            # with this option, can take input as: [cha z y x], [z y x], or [y x]
            imagesOut[iImg] = ismrmrd.Image.from_array(data[...,iImg].transpose((3, 2, 0, 1)), transpose=False)

            # Create a copy of the original fixed header and update the data_type
            # (we changed it to int16 from all other types)
            oldHeader = self.mrdHeader[iImg]
            oldHeader.data_type = imagesOut[iImg].data_type

            # Set the image_type to match the data_type for complex data
            if (imagesOut[iImg].data_type == ismrmrd.DATATYPE_CXFLOAT) or (imagesOut[iImg].data_type == ismrmrd.DATATYPE_CXDOUBLE):
                oldHeader.image_type = ismrmrd.IMTYPE_COMPLEX

            oldHeader.image_series_index += self.image_series_index_offset

            imagesOut[iImg].setHead(oldHeader)

            # Create a copy of the original ISMRMRD Meta attributes and update
            tmpMeta = self.mrdMeta[iImg]
            tmpMeta['DataRole']                       = 'Image'
            if len(self.ImageProcessingHistory       ) > 0: tmpMeta['ImageProcessingHistory'       ] = self.ImageProcessingHistory
            if len(self.SequenceDescriptionAdditional) > 0: tmpMeta['SequenceDescriptionAdditional'] = '_'.join(self.SequenceDescriptionAdditional)
            tmpMeta['Keep_image_geometry']            = 1

            metaXml = tmpMeta.serialize()
            # logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
            # logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

            imagesOut[iImg].attribute_string = metaXml

        logging.info(f'ImageFactory: {self.image_series_index_offset=}')
        logging.info(f'ImageFactory: {self.ImageProcessingHistory=}')
        logging.info(f'ImageFactory: {self.SequenceDescriptionAdditional=}')

        return imagesOut

def check_OR_arguments(config, arg_name: str, arg_type: type, arg_default: any) -> any:
    arg_value = arg_default

    if ('parameters' in config) and (arg_name in config['parameters']):
        logging.info(f"found config['parameters']['{arg_name}'] : type={type(config['parameters'][arg_name])} content={config['parameters'][arg_name]}")
        arg_value =  config['parameters'][arg_name]
    else:
        logging.warning(f"config['parameters']['{arg_name}'] NOT FOUND !!")

    # in OR, the config only provides strings, so need to cast to the correct type
    if arg_type is str:
        pass
    elif arg_type is bool:
        if type(arg_value) is not bool:
            if   arg_value == 'True' : arg_value = True
            elif arg_value == 'False': arg_value = False
            else: raise ValueError(f"{arg_name} is detected as `str` but is not 'True' or 'False' ! Cannot cast it to `bool`")
    elif arg_type is int:
        if type(arg_value) is not int:
            arg_value = int(arg_value)
    elif arg_type is float:
        if type(arg_value) is not float:
            arg_value = float(arg_value)
    else:
        raise TypeError('wrong type in the config)')

    logging.info(f'{arg_name} = {arg_value}')
    return arg_value

def get_direction(image_header):

    # Extract necessary fields
    read_dir      = image_header.read_dir
    phase_dir     = image_header.phase_dir
    slice_dir     = image_header.slice_dir 

    # Tested only for sagittal orientation
    read_dir_ants  = [read_dir[0],  read_dir[1],  -read_dir[2]]
    phase_dir_ants = [phase_dir[0], phase_dir[1], -phase_dir[2]]
    slice_dir_ants = [slice_dir[0], slice_dir[1], -slice_dir[2]]

    direction = np.column_stack([
        np.array(read_dir_ants),
        np.array(phase_dir_ants),
        np.array(slice_dir_ants)
    ])

    return direction

def process_image(images, config):
    if not hasattr(process_image, 'image_series_index_offset'):
        process_image.image_series_index_offset = 0
    
    if len(images) == 0:
        return []

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # OR parameters
    BrainMaskConfig    = check_OR_arguments(config, 'BrainMaskConfig'   , str , 'ApplyInBrainMask')
    ANTsConfig         = check_OR_arguments(config, 'ANTsConfig'        , str , 'N4Dn'            )
    SaveOriginalImages = check_OR_arguments(config, 'SaveOriginalImages', bool, False             )

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    logging.info(f'MRD supposed organization : [img cha z y x]')
    logging.info(f'MRD data shape : {data.shape}')
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    logging.warning(f'MRD SequenceDescription : {meta[0]['SequenceDescription']}')

    # Display MetaAttributes for first image
    # logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(meta[0]))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta[0]:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(meta[0]['IceMiniHead']).decode('utf-8'))

    # Diagnostic info
    matrix    = np.array(head[0].matrix_size  [:]) 
    fov       = np.array(head[0].field_of_view[:])
    voxelsize = fov/matrix
    read_dir  = np.array(images[0].read_dir )
    phase_dir = np.array(images[0].phase_dir)
    slice_dir = np.array(images[0].slice_dir)
    logging.info(f'MRD computed maxtrix [x y z] : {matrix   }')
    logging.info(f'MRD computed fov     [x y z] : {fov      }')
    logging.info(f'MRD computed voxel   [x y z] : {voxelsize}')
    logging.info(f'MRD read_dir         [x y z] : {read_dir }')
    logging.info(f'MRD phase_dir        [x y z] : {phase_dir}')
    logging.info(f'MRD slice_dir        [x y z] : {slice_dir}')

    imgfactory = ImageFactory()
    imgfactory.image_series_index_offset = process_image.image_series_index_offset
    imgfactory.mrdHeader = head
    imgfactory.mrdMeta   = meta

    direction = get_direction(head[0])

    data_3d = imgfactory.MRD5Dto3D(data_mrd5D=data)
    logging.info(f'ANTs input data shape : {data_3d.shape}')
    ants_image_in = ants.from_numpy(data_3d, spacing=list(voxelsize), direction=direction)

    masking_args  = {}
    masking_label = ''
    if BrainMaskConfig != 'None':

        mask = antspynet.brain_extraction(ants_image_in, modality="t1")
        thresh = 0.2
        mask = ants.threshold_image(mask, thresh, 10, 1, 0)

        masking_args['mask'] = mask
        masking_label = '@AntspynetMask'

    # default configuration, just copy original images
    if SaveOriginalImages:
        images_out = imgfactory.ANTsImageToMRD(ants_image_in) # !!! still need to "Keep_image_geometry"
    else:
        images_out = []

    if   BrainMaskConfig == 'ApplyInBrainMask':
        images_out      += imgfactory.ANTsImageToMRD(mask, history='AntspynetMask', seq_descrip_add='Brainmask')
    elif BrainMaskConfig == 'SkullStripping':
        images_out      += imgfactory.ANTsImageToMRD(mask, history='AntspynetMask', seq_descrip_add='Brainmask')
        ants_image_in    = ants.mask_image(ants_image_in, mask)
        imgfactory.SequenceDescriptionAdditional.pop()
        images_out      += imgfactory.ANTsImageToMRD(ants_image_in, history='AntspynetMasked', seq_descrip_add='SS')

    if ANTsConfig == 'None':
        pass
    
    elif ANTsConfig == 'N4':
        ants_image_n4    = ants.n4_bias_field_correction(ants_image_in, verbose=True, **masking_args)
        images_out      += imgfactory.ANTsImageToMRD(ants_image_n4, history='ANTsN4BiasFieldCorrection'+masking_label, seq_descrip_add='N4')

    elif ANTsConfig == 'Dn':
        ants_image_dn    = ants.denoise_image(ants_image_in, v=1, **masking_args)
        images_out      += imgfactory.ANTsImageToMRD(ants_image_dn, history='ANTsDenoiseImage'+masking_label, seq_descrip_add='Dn')

    elif ANTsConfig == 'N4Dn':
        ants_image_n4    = ants.n4_bias_field_correction(ants_image_in, verbose=True, **masking_args)
        images_out      += imgfactory.ANTsImageToMRD(ants_image_n4, history='ANTsN4BiasFieldCorrection'+masking_label, seq_descrip_add='N4')
        ants_image_n4_dn = ants.denoise_image(ants_image_n4, v=1, **masking_args)
        images_out      += imgfactory.ANTsImageToMRD(ants_image_n4_dn, history='ANTsDenoiseImage'+masking_label, seq_descrip_add='N4_Dn')

    elif ANTsConfig == 'DnN4':
        ants_image_dn    = ants.denoise_image(ants_image_in, v=1, **masking_args)
        images_out      += imgfactory.ANTsImageToMRD(ants_image_dn, history='ANTsDenoiseImage'+masking_label, seq_descrip_add='Dn')
        ants_image_dn_n4 = ants.n4_bias_field_correction(ants_image_dn, verbose=True, **masking_args)
        images_out      += imgfactory.ANTsImageToMRD(ants_image_dn_n4, history='ANTsN4BiasFieldCorrection'+masking_label, seq_descrip_add='Dn_N4')

    process_image.image_series_index_offset = imgfactory.image_series_index_offset
    return images_out
