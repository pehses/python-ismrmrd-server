
import constants
from connection import Connection

import socket
import logging
import multiprocessing
import ismrmrd.xsd
import importlib
import cProfile
import os


class Server:
    """
    Something something docstring.
    """

    def __init__(self, address, port, savedata, savedataFolder, multiprocessing):
        logging.info("Starting server and listening for data at %s:%d", address, port)
        if (savedata is True):
            logging.debug("Saving incoming data is enabled.")

        if (multiprocessing is True):
            logging.debug("Multiprocessing is enabled.")

        self.multiprocessing = multiprocessing
        self.savedata = savedata
        self.savedataFolder = savedataFolder
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((address, port))

    def serve(self):
        logging.debug("Serving... ")
        self.socket.listen(0)

        while True:
            sock, (remote_addr, remote_port) = self.socket.accept()

            logging.info("Accepting connection from: %s:%d", remote_addr, remote_port)

            if (self.multiprocessing is True):
                process = multiprocessing.Process(target=self.handle, args=[sock])
                process.daemon = True
                process.start()
                logging.debug("Spawned process %d to handle connection.", process.pid)
            else:
                self.handle(sock)

    def handle(self, sock):

        try:
            connection = Connection(sock, self.savedata, "", self.savedataFolder, "dataset")

            # First message is the config (file or text)
            config = next(connection)

            # Break out if a connection was established but no data was received
            if ((config is None) & (connection.is_exhausted is True)):
                logging.info("Connection closed without any data received")
                return

            # Second messages is the metadata (text)
            metadata_xml = next(connection)
            # logging.debug("XML Metadata: %s", metadata_xml)
            try:
                metadata = ismrmrd.xsd.CreateFromDocument(metadata_xml)
                if (metadata.acquisitionSystemInformation.systemFieldStrength_T != None):
                    logging.info("Data is from a %s %s at %1.1fT", metadata.acquisitionSystemInformation.systemVendor, metadata.acquisitionSystemInformation.systemModel, metadata.acquisitionSystemInformation.systemFieldStrength_T)
            except:
                logging.warning("Metadata is not a valid MRD XML structure.  Passing on metadata as text")
                metadata = metadata_xml

            # Decide what program to use based on config
            # If not one of these explicit cases, try to load file matching name of config
            if (config == "simplefft"):
                import simplefft3D
                importlib.reload(simplefft3D)
                logging.info("Starting simplefft processing based on config")
                simplefft3D.process(connection, config, metadata)
            elif (config == "invertcontrast"):
                import invertcontrast
                importlib.reload(invertcontrast)
                logging.info("Starting invertcontrast processing based on config")
                invertcontrast.process(connection, config, metadata)
            elif (config == "bart_pics"):
                import bart_pics
                importlib.reload(bart_pics)
                logging.info("Starting bart_pics processing based on config")
                bart_pics.process(connection, config, metadata)
            elif (config == "bart_cs"):
                import bart_cs
                importlib.reload(bart_cs)
                logging.info("Starting compressed sensing processing based on config")
                bart_cs.process(connection, config, metadata)
                #with cProfile.Profile() as pr:
                #    bart_cs.process(connection, config, metadata)
                #pr.dump_stats("/tmp/share/debug/bart_cs.prof")
            elif (config == "bart_radial"):
                import bart_radial
                importlib.reload(bart_radial)
                logging.info("Starting bart_radial processing based on config")
                bart_radial.process(connection, config, metadata)
            elif (config == "bart_spiral"):
                import bart_spiral
                importlib.reload(bart_spiral)
                logging.info("Starting bart_spiral processing based on config")
                bart_spiral.process(connection, config, metadata)
            elif (config == "bart_jemris"):
                import bart_jemris
                importlib.reload(bart_jemris)
                logging.info("Starting bart_jemris processing based on config")
                bart_jemris.process(connection, config, metadata)
            elif (config == "bart_pulseq"):
                import bart_pulseq
                importlib.reload(bart_pulseq)
                logging.info("Starting bart_pulseq processing based on config")
                bart_pulseq.process(connection, config, metadata)
            elif (config == "powergrid_pulseq"):
                import powergrid_pulseq
                importlib.reload(powergrid_pulseq)
                logging.info("Starting powergrid_pulseq processing based on config")
                with cProfile.Profile() as pr:
                    powergrid_pulseq.process(connection, config, metadata)
                pr.dump_stats("/tmp/share/debug/powergrid_pulseq.prof")
            elif (config == "analyzeflow"):
                import analyzeflow
                importlib.reload(analyzeflow)
                logging.info("Starting analyzeflow processing based on config")
                analyzeflow.process(connection, config, metadata)
            elif (config == "ir_fit"):
                import ir_fit
                importlib.reload(ir_fit)
                logging.info("Starting ir_fit processing based on config")
                ir_fit.process(connection, config, metadata)
            elif (config == "null"):
                logging.info("No processing based on config")
                try:
                    for msg in connection:
                        if msg is None:
                            break
                finally:
                    connection.send_close()
            elif (config == "savedataonly"):
                # Dummy loop with no processing
                try:
                    for msg in connection:
                        if msg is None:
                            break
                finally:
                    connection.send_close()
            else:
                try:
                    # Load module from file having exact name as config
                    module = importlib.import_module(config)
                    logging.info("Starting config %s", config)
                    module.process(connection, config, metadata)
                except ImportError:
                    logging.info("Unknown config '%s'.  Falling back to 'invertcontrast'", config)
                    invertcontrast.process(connection, config, metadata)

        except Exception as e:
            logging.exception(e)

        finally:
            # Encapsulate shutdown in a try block because the socket may have
            # already been closed on the other side
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            sock.close()
            logging.info("Socket closed")

            # Dataset may not be closed properly if a close message is not received
            if connection.savedata is True:
                try:
                    connection.dset.close()
                except:
                    pass

                if (connection.savedataFile == ""):
                    try:
                        # Rename the saved file to use the protocol name
                        dset = ismrmrd.Dataset(connection.mrdFilePath, connection.savedataGroup, False)
                        groups = dset.list()

                        if ('xml' in groups):
                            xml_header = dset.read_xml_header()
                            xml_header = xml_header.decode("utf-8")
                            mrdHead = ismrmrd.xsd.CreateFromDocument(xml_header)

                            if (mrdHead.measurementInformation.protocolName != ""):
                                newFilePath = connection.mrdFilePath.replace("MRD_input_", mrdHead.measurementInformation.protocolName + "_")
                                os.rename(connection.mrdFilePath, newFilePath)
                                connection.mrdFilePath = newFilePath
                    except:
                        pass

                if connection.mrdFilePath is not None:
                    logging.info("Incoming data was saved at %s", connection.mrdFilePath)
