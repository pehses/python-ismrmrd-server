# ----- 1. First stage to create a devcontainer -----
# Start from standard python-ismrmrd-server devcontainer
FROM kspacekelvin/fire-python-devcon AS fire-python-custom-devcon

# Clone the latest version of python-ismrmrd-server
RUN cd /opt/code && \
    git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git

# Install any additional package dependencies here
RUN pip install scipy

# ----- 2. Second stage to create a runtime container for deployment -----
FROM fire-python-custom-devcon AS fire-python-custom-runtime

# Copy in modules and other files as needed
COPY filter.py    /opt/code/python-ismrmrd-server
COPY filter.json  /opt/code/python-ismrmrd-server

# Set the starting directory so that code can use relative paths
WORKDIR /opt/code/python-ismrmrd-server

# Use the -d argument at the end to indicate the default (intended) module to be run by this Docker image
CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "-d=filter"]

# Replace the above CMD with this ENTRYPOINT to allow allow "docker stop"
# commands to be passed to the server.  This is useful for deployments, but
# more annoying for development
# ENTRYPOINT [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "-d=filter"]