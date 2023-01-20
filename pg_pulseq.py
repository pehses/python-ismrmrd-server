
import ismrmrd
import os
import logging
import importlib

import powergrid_pulseq 
import powergrid_pulseq_ho
import powergrid_pulseq_dream
from pulseq_helper import get_ismrmrd_arrays, check_signature

""" Checks trajectory type and launches reconstruction
"""

# Folder for sharing data
shareFolder = "/tmp/share"
dependencyFolder = os.path.join(shareFolder, "dependency")

########################
# Main Function
########################

def process(connection, config, metadata):
  
    prot_folder = os.path.join(dependencyFolder, "metadata")
    prot_filename = os.path.splitext(metadata.userParameters.userParameterString[0].value)[0] # protocol filename from Siemens protocol parameter tFree
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
    prot_arrays = get_ismrmrd_arrays(prot_file)
    prot_hdr = ismrmrd.xsd.CreateFromDocument(prot.read_xml_header())
    check_signature(metadata, prot_hdr) # check MD5 signature
    prot.close()

    if "dream" in prot_arrays:
        importlib.reload(powergrid_pulseq_dream)
        logging.info("Starting PowerGrid 3DREAM reconstruction.")
        powergrid_pulseq_dream.process(connection, config, metadata, prot_file)
    elif len(metadata.userParameters.userParameterBase64) or len(prot_hdr.userParameters.userParameterBase64):
        importlib.reload(powergrid_pulseq_ho)
        logging.info("Starting PowerGrid spiral higher order reconstruction.")
        powergrid_pulseq_ho.process(connection, config, metadata, prot_file)
    else:
        importlib.reload(powergrid_pulseq)
        logging.info("Starting PowerGrid spiral reconstruction.")
        powergrid_pulseq.process(connection, config, metadata, prot_file)
