
import ismrmrd
import os
import logging

import bart_pulseq_spiral 
import bart_pulseq_cartesian

""" Checks trajectory type and launches reconstruction
"""

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

########################
# Main Function
########################

def process(connection, config, metadata):
  
    # Create debug folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    prot_folder = os.path.join(dependencyFolder, "metadata")
    prot_filename = os.path.splitext(metadata.userParameters.userParameterString[0].value)[0] # protocol filename from Siemens protocol parameter tFree
    prot_file = prot_folder + "/" + prot_filename + ".h5"
    if not os.path.isfile(prot_file):
        raise ValueError("No protocol file available.")

    prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
    hdr = ismrmrd.xsd.CreateFromDocument(prot.read_xml_header())
    trajtype = hdr.encoding[0].trajectory.value

    if trajtype == 'spiral':
        import importlib
        importlib.reload(bart_pulseq_spiral)
        logging.info("Starting spiral reconstruction.")
        bart_pulseq_spiral.process_spiral(connection, config, metadata, prot_file)
    elif trajtype == 'cartesian':
        import importlib
        importlib.reload(bart_pulseq_cartesian)
        logging.info("Starting cartesian reconstruction.")
        bart_pulseq_cartesian.process_cartesian(connection, config, metadata, prot_file)
    else:
        raise ValueError('Trajectory type not recognized')
