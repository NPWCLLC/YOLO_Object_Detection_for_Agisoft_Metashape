
import os, sys, subprocess, pkg_resources, re


requirements_txt = """
numpy==2.0.2
pandas==2.2.3
opencv-python==4.11.0.86
shapely==2.0.7
pathlib==1.0.1
Rtree==1.3.0
tqdm==4.67.1
ultralytics
scikit-learn==1.6.1
albumentations==2.0.5
"""
def update_pip():
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Error upgrade pip:")
        print(result.stderr)
        sys.exit(1)
    else:
        print("✅ Upgrade pip successful !")

def get_cuda_version_from_nvidia_smi():
    """
    Trying to get the CUDA version supported by the driver,
    using the nvidia-smi command.
    """
    try:
        # Run nvidia-smi and get the output
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout
        # print(output) # For debugging

        # We are looking for a string with a CUDA version, for example, "CUDA Version: 12.1" or "CUDA Version: 11.8"
        # Regular expression for searching for "CUDA Version: X.Y"
        match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', output)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))

            return f"https://download.pytorch.org/whl/cu{major}{minor}"

        else:
            print("Couldn't find the CUDA version in the nvidia-smi output.")
            return "https://download.pytorch.org/whl/cpu"

    except FileNotFoundError:
        raise "nvidia-smi was not found. Make sure that the NVIDIA drivers are installed and nvidia-smi is available in the PATH."

    except subprocess.CalledProcessError:
        raise "Error when executing nvidia-smi."

    except Exception as e:
        raise f"An error occurred while trying to get the version CUDA: {e}"

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

def install_packages(requirements, index_url=None):
    """
    Checks and installs a predefined list of packages with specific versions if they are not already installed.
    It ensures the required versions of the packages are installed.
    If the required versions of the packages are not present, it installs them using pip and provides a link to a page where the correct versions can be found.

    :param requirements: String containing the list of requirements (e.g., "torch==2.8.0\ntorchvision==0.23.0")
    :type requirements: str
    :param index_url: Optional URL for the package index (e.g., for CUDA-specific PyTorch)
    :type index_url: str or None
    :return: None
    """
    packages_to_install = check_package_installed(requirements)
    if packages_to_install:

        if index_url:
            command = [sys.executable, "-m", "pip", "install", *packages_to_install, "--index-url", index_url]
        else:
            command = [sys.executable, "-m", "pip", "install", *packages_to_install]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ Error during installation:")
            print(result.stderr)
            sys.exit(1)
        else:
            print(f"✅ All {packages_to_install} packages installed !")

def uninstall_packages(pakages):
    command = [sys.executable, "-m", "pip", "uninstall","-y", *pakages]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Error uninstalling packages:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"✅ All {pakages} packages uninstalled!")

if __name__ == "__main__":
    update_pip()
    install_packages(requirements_txt)

    requirements_torch = ['torch', 'torchvision']
    uninstall_packages(requirements_torch)
    install_packages("\n".join(requirements_torch), index_url=get_cuda_version_from_nvidia_smi())