# ## For Agisoft Metashape Professional 2.2.0
# - python 3.9
#
# #### Based on:
# - https://github.com/agisoft-llc/metashape-scripts/blob/master/src/detect_objects.py
# - https://docs.ultralytics.com/
#
# ## How to install (Windows):
# How to install external Python module to Metashape Professional package https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-metashape-professional-package


# ## Install packages
# 1. Update pip in the command line run
# ```
#  - cd /d %programfiles%\Agisoft\python
#  - python.exe -m pip install --upgrade pip

# ```
# 2. Add this script to auto-launch, copy script to folder yolo11_detected to %programfiles%\Agisoft\modules and copy script run_scripts.py to C:/Users/<username>/AppData/Local/Agisoft/Metashape Pro/scripts/
# 3. Restart Metashape.
# 4. Wait for the end of the installation packages.
# 5. Uninstall torch, torchvision ( for cpu) and install torch+cuda torchvision+cuda (for gpu) from https://download.pytorch.org/whl/cu118
# OR see your cuda version for torch and torchvision at https://pytorch.org/get-started/previous-versions/ for python 3.9
#
# ```
#  - cd /d %programfiles%\Agisoft\python
#  - python.exe -m pip uninstall -y torch torchvision
#  - python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# ```
# 6. Restart Metashape.

# The script has modules:
# - Detected objects yolo model.
# - Creating a dataset in yolo format.




import os, sys, subprocess, pkg_resources


requirements_txt = """
numpy==2.0.2
pandas==2.2.3
opencv-python==4.11.0.86
shapely==2.0.7
pathlib==1.0.1
Rtree==1.3.0
tqdm==4.67.1
ultralytics==8.3.84
torch
torchvision
scikit-learn==1.6.1
albumentations==2.0.5
"""

def is_package_installed(package_name, version=None):
    """
    Checks whether the package is installed with the specified version.

    :param package_name: The name of the package to check.
    ::type package_name: string
    :param version: The version of the package to check (optional).
    :type version: str
    :return: Package version if the specified version is installed, or False in case of an error.
    ::type: str | bool
    """
    try:
        if version:
            package_str = f"{package_name}=={version}"
            pkg_resources.require(package_str)

        installed_version = pkg_resources.get_distribution(package_name).version
        return installed_version
    except pkg_resources.DistributionNotFound:
        return False
    except pkg_resources.VersionConflict:
        return False

def check_package_installed(requirements_str):
    packages_to_install = []
    for line in requirements_str.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            package, version = line.split("==", 1)
        else:
            package, version = line, None


        installed = is_package_installed(package, version)
        if installed:
            print(f"✅ {package} {installed} installed")

        else:
            if version:
                packages_to_install.append("{}=={}".format(package, version))
            else:
                packages_to_install.append("{}".format(package))

            print(f"❌ {package} not installed or wrong version")

    return packages_to_install

def install_packages(requirements):
    """
    Checks and installs a predefined list of packages with specific versions if they are not already installed.
    It ensures the required versions of the packages are installed.
    If the required versions of the packages are not present, it installs them using pip and provides a link to a page where the correct versions can be found.

    :return: None
    """

    packages_to_install = check_package_installed(requirements)
    if packages_to_install:

        command = [sys.executable, "-m", "pip", "install", *packages_to_install]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ Error during installation:")
            print(result.stderr)
            sys.exit(1)
        else:
            print("✅ All packages installed !")

install_packages(requirements_txt)

def create_yolo_dataset():

    from PySide2 import QtWidgets
    from .create_yolo_dataset import WindowCreateYoloDataset
    chunk = Metashape.app.document.chunk

    if chunk is None or chunk.orthomosaic is None:
        raise Exception("No active orthomosaic.")

    if chunk.orthomosaic.resolution > 0.105:
        raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = WindowCreateYoloDataset(parent)

def detect_objects():

    from PySide2 import QtWidgets
    from .detect_yolo import MainWindowDetect
    chunk = Metashape.app.document.chunk

    if chunk is None or chunk.orthomosaic is None:
        raise Exception("No active orthomosaic.")

    if chunk.orthomosaic.resolution > 0.105:
        raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")

    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = MainWindowDetect(parent)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import Metashape

Metashape.app.addMenuItem("Scripts/YOLO Tools/Prediction", detect_objects)
Metashape.app.addMenuItem("Scripts/YOLO Tools/Create yolo dataset", create_yolo_dataset)