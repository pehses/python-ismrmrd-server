# Getting started with dev containers
Traditional development involves the local installation of libraries, packages, and other dependencies that are required by code being developed.  This can be complex if the native environment is a different operating system than the production environment, such as when developing in Windows because the Docker images will run in Linux.  Managing the development environment can also be complicated by different projects requiring different version of the same package.  If developing in Python, [virtual environments (venvs)](https://docs.python.org/3/library/venv.html) can be created and managed using tools such as [conda](https://docs.conda.io/en/latest/) or [mamba](https://github.com/mamba-org/mamba).

[Development containers (devcontainers)](https://code.visualstudio.com/docs/devcontainers/containers) are an alternative approach, where the development environment itself is created within a Docker image.  This can simplify the process of setting up a working environment and reduce issues when moving between development and deployment environments.  During development, the devcontainer is started automatically in Docker and Visual Studio Code executes the code inside the devcontainer, complete with a debugging environment.

1. [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and clone this repository into a local folder:
    ```
    git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git
    ```

1. [Install Visual Studio Code](https://code.visualstudio.com/) and install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) plugin from its webpage or from the Extensions panel in Visual Studio Code.

1. Start Docker.

1. Open the `python-ismrmrd-server` folder in VS Code.  The dev container should be detected automatically and a prompt will appear at the bottom right to "Reopen in Container".  This action can also be found in the [Command Palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette), which can be invoked using the `Ctrl+Shift+P` key combination in Windows/Linux or `Command+Shift+P` in MacOS and typing in "Reopen in Container".  This step may take a few minutes when run for the first time as the dev container Docker image is built (the devcontainer is cached for future runs).

1. When the repository is opened in a dev container, the green "Remote Window" section at the bottom left of VS Code will indicate "Dev Container: python-ismrmrd-server-devcon"

1. Select "Run and Debug" from the Activity Bar along the left side and then "Start server" from the top left.  This will start the server (i.e. main.py) and breakpoints can be marked by opening a .py file and clicking to the left of the the line number (a red circle will appear).

1. The [RunClientServerRecon.ipynb](RunClientServerRecon.ipynb) Python notebook contains code snippets for generating an input raw data set, running the client, and displaying the results.  Alternatively, the [client.py](client.py) can be run the Terminal window.  Note that files (e.g. datasets) placed in this repo's folder will automatically be mapped inside the dev container and output files generated inside this repo's folder will also be visible on the host file system.

Note: The Docker devcontainer can also be built manually by opening a terminal in the python-ismrmrd-server folder and running:
```
docker build --no-cache -t python-mrd-devcontainer --target python-mrd-devcontainer -f docker/Dockerfile ./
```

## GitHub Codespaces
[Codespaces](https://github.com/features/codespaces) is a feature of GitHub that allows for dev containers to be configured and run in the cloud.  Codespaces can be accessible via a browser, running a web-based version of Visual Studio Code or Jupyter.  A codespace can also be opened up in a native VS Code instance running locally for faster performnce.  Codespaces are free for registerd GitHub users for up to 60 hours/month as of December 2023 with billable additional usage.

To set up a GitHub codespace:
1. Create a GitHub account and log in.

1. Browse to the main directory of this repository: https://github.com/kspaceKelvin/python-ismrmrd-server

1. Click on the green "Code" button at the top right select "Codespaces", and then the "+" symbol (Create a codespace on master).  It may take a few minutes for the codespace to be created for the first time.

1. The web version of VS Code can be used in the same way as the native program.  For example the server can be started using "Run Server" configuration in Run and Debug, and the [RunClientServerRecon.ipynb](RunClientServerRecon.ipynb) notebook can also be used.

The default idle timeout of 30 minutes (configurable in the GitHub user settings) will stop the codespace when inactive.  Changed and new files are kept within the codespace even if the codespace is stopped (up to 15 GB of free storage per month).  Codespaces can be manually stopped or deleted if needed.