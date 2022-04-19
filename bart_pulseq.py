
import ismrmrd
import os
import logging
import importlib

import bart_pulseq_spiral 
import bart_pulseq_cartesian
import bart_jemris
from pulseq_helper import check_signature

""" Checks trajectory type and launches reconstruction
"""

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

########################
# Main Function
########################

def process(connection, config, hdr):
  
    # Create debug folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    meta_folder = os.path.join(dependencyFolder, "metadata")
    meta_filename = os.path.splitext(hdr.userParameters.userParameterString[0].value)[0] # metadata filename (Siemens: raw data parameter tFree)
    meta_file = meta_folder + "/" + meta_filename + ".h5"
    if not os.path.isfile(meta_file):
        raise ValueError("No metadata file available.")

    meta = ismrmrd.Dataset(meta_file, create_if_needed=False)
    meta_hdr = ismrmrd.xsd.CreateFromDocument(meta.read_xml_header())
    meta.close()
    trajtype = meta_hdr.encoding[0].trajectory.value
    check_signature(hdr, meta_hdr) # check MD5 signature

    if trajtype == 'spiral':
        importlib.reload(bart_pulseq_spiral)
        logging.info("Starting spiral reconstruction.")
        bart_pulseq_spiral.process_spiral(connection, config, hdr, meta_file)
    elif trajtype == 'cartesian':
        importlib.reload(bart_pulseq_cartesian)
        logging.info("Starting cartesian reconstruction.")
        bart_pulseq_cartesian.process_cartesian(connection, config, hdr, meta_file)
    elif trajtype == 'other':
        importlib.reload(bart_jemris)
        logging.info("Starting JEMRIS reconstruction.")
        bart_jemris.process(connection, config, hdr, meta_file)
    else:
        raise ValueError('Trajectory type not recognized')
