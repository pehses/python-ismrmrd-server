
import ismrmrd
import os
import itertools
import logging
import base64

import bart_pulseq_spiral 
import bart_pulseq_cartesian
import bart_jemris

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
  
    protFolder = os.path.join(dependencyFolder, "pulseq_protocols")
    protFolder_local = "/tmp/local/pulseq_protocols" # optional local protocol mountpoint (via -v option)
    prot_filename = metadata.userParameters.userParameterString[0].value_ # protocol filename from Siemens protocol parameter tFree

    # Check if local protocol folder is available - if not use protFolder (scanner)
    date = prot_filename.split('_')[0] # folder in Protocols (=date of seqfile)
    protFolder_loc = os.path.join(protFolder_local, date)
    if os.path.exists(protFolder_loc):
        protFolder = protFolder_loc

    prot_file = protFolder + "/" + prot_filename + ".h5"
    prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
    hdr = ismrmrd.xsd.CreateFromDocument(prot.read_xml_header())
    trajtype = hdr.encoding[0].trajectory

    if trajtype == 'spiral':
        import importlib
        importlib.reload(bart_pulseq_spiral)
        logging.info("Starting spiral reconstruction.")
        bart_pulseq_spiral.process_spiral(connection, config, metadata)
    elif trajtype == 'cartesian':
        import importlib
        importlib.reload(bart_pulseq_cartesian)
        logging.info("Starting cartesian reconstruction.")
        bart_pulseq_cartesian.process_cartesian(connection, config, metadata)
    elif trajtype == 'other': # WIP: support Jemris sequences (not available yet)
        pass
        import importlib
        importlib.reload(bart_jemris)
        logging.info("Starting cartesian reconstruction.")
        bart_jemris.process_cartesian(connection, config, metadata)
    else:
        raise ValueError('Trajectory type not recognized')


