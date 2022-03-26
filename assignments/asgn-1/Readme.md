The home assignments in the course use [Docker](https://www.docker.com/) as a way to encapsulate the necessary environment state and provide flawless experience of using necessary frameworks, such as [OpenCV](https://opencv.org/), [PyTorch](https://pytorch.org/) and [rcognita](https://github.com/AIDynamicAction/rcognita). The preparatory stage (Docker installation, image building) will take some time, but after that the workflow will lack any version mismatches and dependency problems. Essentially, you will have a standalone Linux with all the necessary modules already installed.

In order to set up the environment for the course, it is necessary to perform certain preparatory steps.

#

### 1: Cloning the repository
First of all,  make sure you have any Git system installed on your computer. You can use [GitHub Client](https://desktop.github.com/) for Windows and MacOS or standalone [Git](https://git-scm.com/) which gives you
```sh
git clone --recurse-submodules https://github.com/AIDynamicAction/edu-2022-safeAI.git
```
or clone it with the desktop app. In such a case **please ensure** that the submodule is being downloaded as well.

#

### 2: Installing Docker

Please download the installer https://docs.docker.com/desktop/ for your OS and run it. The installation could take some time.

#

### 3: Setting up the container

Depending on your OS, do the following:

**FOR WINDOWS:**
***Please note that it is very important to install the HyperV-based version of Docker when using Windows. Otherwise, you may experience difficulties with file sharing.***
1. Open your desktop docker app, go to settings -> resources -> file sharing 
2. Press "+" button and add current directory as shared 
3. launch sequentially build_docker.bat (builds image) , run_docker.bat (runs container), into_docker.bat (attaches to container)

**FOR UNIX (Linux, MacOS, ...):**

- go to the folder *edu-2022-safeAI* in Terminal (with ```cd``` command)
- go to the subfolder *assignments/asgn-1*
- grant *executable* property to the *docker_unix_setup.sh* with ``` chmod +x docker_unix_setup.sh ```command
- run the setup script with ```./docker_unix_setup.sh``` command. (In case of permission issues run ```sudo ./docker_unix_setup.sh``` instead)

This command will install the container with the environment: all the modules, specific Python version, etc. Also it will move a script named *into_docker.sh* into the *asgn-1* folder and run it. Its usage and functions are described below.

Note that the installation **may take time** (typically under half an hour) and it **may require considerable amount of disc space** (under 10 Gb).
Please make sure that you have enough free disc space prior to the running of the setup script. The error with "read-only file system" indicates that there is not enough space.

#

### 4: Accessing and using the container

Run *./into_dicker.sh* command in order to enter the container. After the installation it runs automatically, after that it should be done manually.

Running that command gives a command linne access into the virtual Ubuntu machine, that shares the *asgn-1* folder with the main operating system. The files that were modified in the container will remain modified after the container shutdown.

After entering the virtual machine you will see *root@docker-desktop:/usr/src/assignment_1#* command line greeting, which indicates that everything worked out well. Make yourself comfortable in the terminal, explore it with the commands *whoami*, *ls*, *pwd*, as long as the folder structure.

Now you can use regular Python interpreter, try running ... from ... folder with ... command.

#

### 5: Assignment 1
