# YOLO Object Detection for Agisoft Metashape

A Python module for Agisoft Metashape Professional that enables YOLO-based object detection on orthomosaic images.
The module can be used for various tasks related to orthomosaic processing, including animal population monitoring, mapping, agriculture, forestry and other areas where automatic detection and classification of objects in aerial photographs is required.


## Overview

This module integrates the YOLO (You Only Look Once) object detection framework Ultralitycs with Agisoft Metashape Professional, allowing users to:

1. Detect objects on orthomosaic images using pre-trained or custom YOLO models
2. Create YOLO-format datasets from Metashape data for training custom models

The module is designed to work with Agisoft Metashape Professional 2.2.0 and above, using Python 3.9 and CUDA >= 11.8 for GPU acceleration.
OR see your cuda version for torch and torchvision at https://pytorch.org/get-started/previous-versions/ for python 3.9

## Requirements

- Agisoft Metashape Professional 2.2.0 or higher
- Python 3.9
- CUDA >= 11.8 (for GPU acceleration)
- The following Python packages:
  - numpy==2.0.2
  - pandas==2.2.3
  - opencv-python==4.11.0.86
  - shapely==2.0.7
  - pathlib==1.0.1
  - Rtree==1.3.0
  - tqdm==4.67.1
  - ultralytics
  - torch
  - torchvision
  - scikit-learn==1.6.1
  - albumentations==2.0.5


## 🚀 Installation

### Prerequisites

*   Agisoft Metashape is installed.
*   For GPU acceleration, an NVIDIA GPU is required with the appropriate CUDA drivers installed.

### Windows Installation

**1. Download and Place the Module**

📥 **[Download the module files here.](https://github.com/NPWCLLC/YOLO_Object_Detection_for_Agisoft_Metashape/archive/refs/heads/main.zip)**

After downloading, you'll need to copy the module to the Agisoft Metashape installation directory.

*   In your file explorer, navigate to the Agisoft Metashape installation folder. This is typically located at `C:\Program Files\Agisoft\Metashape Pro`.
*   Copy the downloaded and unzipped `yolo11_detected` folder into the `modules` subfolder.

The final file structure should look like this:

```
%programfiles%\Agisoft\Metashape Pro\modules\yolo11_detected/
 ├── __init__.py
 ├── auto_install_packages.py
 ├── create_yolo_dataset.py
 └── detect_yolo.py
```

**2. Install Required Packages**

You can install the necessary Python packages automatically or manually. The automatic method is recommended.

#### **Automatic Installation (Recommended)**

*   Start Agisoft Metashape.
*   Go to `Tools > Run Script`.
*   Navigate to the module directory (`%programfiles%\Agisoft\Metashape Pro\modules\yolo11_detected`) and select the `auto_install_packages.py` file.
*   Click `Open` and wait for the installation to complete.
*   You will see a confirmation message, `Packages installed successfully`, when it is finished.

**What the Automatic Installer Does:**
   The script automatically installs the following pinned versions of key libraries:  
   - `numpy==2.0.2`  
   - `pandas==2.2.3`  
   - `opencv-python==4.11.0.86`  
   - `shapely==2.0.7`  
   - `pathlib==1.0.1`  
   - `Rtree==1.3.0`  
   - `tqdm==4.67.1`  
   - `ultralytics` (latest version)  
   - `scikit-learn==1.6.1`  
   - `albumentations==2.0.5`  

   **Removes Default PyTorch Dependencies**  
   After installing `ultralytics`, the script uninstalls the default `torch` and `torchvision` packages that were automatically pulled in as dependencies (which may not be CUDA-optimized or compatible with your system).

   **Detects CUDA Version & Installs Optimized PyTorch**  
   The installer runs `nvidia-smi` to detect your system’s CUDA driver version, then installs the *officially recommended* versions of `torch` and `torchvision` that match your CUDA environment—ensuring optimal performance and GPU support.

> ✅ **Result**: A clean, GPU-ready environment with version-controlled dependencies and CUDA-optimized PyTorch—no manual configuration needed.

#### **Manual Installation**

If you prefer to install the packages manually, you can find detailed instructions on how to add external Python modules to Metashape [here](https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-metashape-professional-package).

For GPU acceleration with PyTorch, ensure you install a version compatible with your CUDA setup. You can check your CUDA version by running `nvidia-smi` in a terminal and find the corresponding PyTorch versions [here](https://pytorch.org/get-started/previous-versions/).

**Important Note on PyTorch Installation for GPU Acceleration:**
The ultralytics package will automatically install torch and torchvision, but these are typically the CPU-only versions. To enable GPU acceleration, you must first install ultralytics, then uninstall the default torch and torchvision, and finally install the versions that correspond to your CUDA toolkit.

Install the following packages using the pip command in the Metashape Python console:
`python.exe -m pip install <package_name>`

*   `numpy==2.0.2`
*   `pandas==2.2.3`
*   `opencv-python==4.11.0.86`
*   `shapely==2.0.7`
*   `pathlib==1.0.1`
*   `Rtree==1.3.0`
*   `tqdm==4.67.1`
*   `ultralytics`
*   `scikit-learn==1.6.1`
*   `albumentations==2.0.5`
*   `torch` (CUDA version specific, see [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/))
*   `torchvision` (CUDA version specific, see [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/))

> **Tip:** You can uninstall packages if needed with the command: `python.exe -m pip uninstall -y <package_name>`

**3. Enable Automatic Script Loading**

To ensure the module is loaded every time you start Metashape, you need to create a simple script in your user directory.

*   Navigate to `C:\Users\<YourUsername>\AppData\Local\Agisoft\Metashape Pro\scripts\`.
*   Create a new file named `run_scripts.py`.
*   Open this file in a text editor and add the following line:

```python
from modules import yolo11_detected
```

*   Save and close the file.

For more information on running scripts automatically at startup, refer to the [official Agisoft documentation](https://agisoft.freshdesk.com/support/solutions/articles/31000133123-how-to-run-python-script-automatically-on-metashape-professional-start).

The installation is now complete. The YOLO Object Detection module will be available the next time you launch Agisoft Metashape.




## Usage

After installation, two new menu items appear under **Scripts > YOLO Tools** in Metashape:


### 1. YOLO Object Detection  
**Menu:** `Scripts > YOLO Tools > Prediction`

Run object detection on orthomosaics using YOLO models (default: `yolo11x-seg.pt`).

#### Features:
- Supports pre-trained or custom YOLO models (e.g., YOLOv8–YOLO11)
- Detect on the full orthomosaic or within user-defined polygon zones
- Adjustable parameters: confidence threshold, IoU threshold, resolution, etc.
- Results saved as **vector shapes** in the Metashape project and exported to **CSV** in the working directory.

**CSV Output Columns:**
- `Label`
- `Score (avg)`
- `Area 2D (m²)`
- `Centroid (x, y)`
- `Width (m)`
- `Length (m)`

#### Requirements:
- Orthomosaic resolution ≤ 10 cm/pixel

#### Configuration Options:
| Option | Description                                                                                                                    |
|--------|--------------------------------------------------------------------------------------------------------------------------------|
| **Working Directory** | Directory for temp files and outputs                                                                                           |
| **Resolution** | Detection resolution (default: 0.5 cm/pix)                                                                                     |
| **Debug Mode** | Saves cropped detections + coordinates for inspection                                                                          |
| **Max Tile Size** | Max tile dimension (px); orthomosaics are tiled for processing                                                                 |
| **Zone Layer** | Optional polygon layer defining detection regions                                                                              |
| **Model Path** | Path to `.pt` model (e.g., `yolo11x-seg.pt`). See supported models: [Ultralytics Models](https://docs.ultralytics.com/models/) |
| **Confidence Threshold** | Min detection score (default: 0.9)                                                                                             |
| **IoU Threshold** | Merge intersection-over-union cutoff (default: 0.6)                                                                            |

> ✅ GPU acceleration strongly recommended.



### 2. Create YOLO Dataset  
**Menu:** `Scripts > YOLO Tools > Create YOLO Dataset`

Generates YOLO-formatted datasets from orthomosaic + vector annotations for model training.

#### Features:
- Exports tiles with labels (bounding boxes or polygons)
- Built-in data augmentation (rotation, mirroring, color/noise transforms)
- Splits data into `train`/`val` sets
- Auto-generates `data.yaml` config

#### Configuration Options:
| Option | Description |
|--------|-------------|
| **Working Directory** | Output base path |
| **Resolution** | Export resolution (default: 0.5 cm/pix) |
| **Debug Mode** | Saves annotated tile previews |
| **Max Tile Size** | Tile dimension (px) |
| **Zone Layer** | Optional AOI polygons |
| **Annotation Layer** | Required: vector layer with labeled objects |
| **Train/Val Split** | Ratio (default: 80% train / 20% val) |
| **Background Ratio** | % of tiles *without* objects (default: 0%) |
| **Augmentation** | Enables geometric transforms (7 rotations/mirrors) |
| **Color Augmentation** | Random HSV shifts, ISO noise, brightness/contrast |
| **Mode** | Export boxes (`detect`) or polygons (`segment`) |

#### Dataset Structure:
```
dataset_yolo/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

#### Label Formats:
- **Detection (YOLOv8+ format):**  
  `<class_id> <x_center> <y_center> <width> <height>`  
- **Segmentation:**  
  `<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>`  
  *(All coordinates normalized to [0,1])*  

#### `data.yaml` Example:
```yaml
train: train/images
val: val/images
nc: 3
names: ['car', 'tree', 'building']
```



## Technical Notes

- **Orthomosaic size handling**: Large rasters are processed via:
  * Tiling with overlap
  * Per-tile inference
  * NMS-based merge of overlapping detections  
- **Coordinate pipeline**:  
  World → Pixel (tile extraction) → Tile pixel → Orthomosaic pixel → World (shape placement)  
- For best accuracy: use orthomosaics ≥ 0.1 m/pix resolution  
- Custom model training: use exported dataset + [Ultralytics YOLO](https://docs.ultralytics.com/)



## Credits  
Based on:
- [Metashape Detect Objects Script](https://github.com/agisoft-llc/metashape-scripts/blob/master/src/detect_objects.py)  
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)  
