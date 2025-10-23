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

## Installation

### Windows Installation

Open a terminal window.
Press Win + R, type cmd, and press Enter.

1. Update pip in the Agisoft Python environment:
   ```
   cd /d %programfiles%\Agisoft\python
   python.exe -m pip install --upgrade pip
   ```

2. Copy the module to the Agisoft modules directory:
   - Copy the files project to folder `%programfiles%\Agisoft\modules\yolo11_detected`
   - Create the `run_scripts.py` script to `C:/Users/<username>/AppData/Local/Agisoft/Metashape Pro/scripts/`
   - 
   How to install external Python module to Metashape Professional package https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-metashape-professional-package
   
   file run_scripts.py:
   ```
   from modules import yolo11_detected
   ```

3. Restart Metashape and wait for the automatic installation of required packages.

4. Install CUDA-enabled PyTorch (for GPU acceleration):
   Check in a terminal window your cuda version `nvidia-smi`.
   See your cuda version for torch and torchvision at https://pytorch.org/get-started/previous-versions/ for python 3.9
   ```
   cd /d %programfiles%\Agisoft\python
   python.exe -m pip uninstall -y torch torchvision
   python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/<your cuda version>
   ```
   `exemple for cuda 11.8 (python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118)`

5. Restart Metashape.

   If everything is fine, you will see in the logs terminal Agisoft Metashape (exemple for cu118):
   - ✅ numpy 2.0.2 installed
   - ✅ pandas 2.2.3 installed
   - ✅ opencv-python 4.11.0.86 installed
   - ✅ shapely 2.0.7 installed
   - ✅ pathlib 1.0.1 installed
   - ✅ Rtree 1.3.0 installed
   - ✅ tqdm 4.67.1 installed
   - ✅ ultralytics 8.3.84 installed
   - ✅ torch 2.7.1+cu118 installed
   - ✅ torchvision 0.22.1+cu118 installed
   - ✅ scikit-learn 1.6.1 installed
   - ✅ albumentations 2.0.5 installed

## Usage

After installation, two new menu items will be available in Metashape under the "Scripts > YOLO Tools" menu:

### 1. YOLO Object Detection

Access via: `Scripts > YOLO Tools > Prediction`

This tool allows you to detect objects on orthomosaic images using YOLO models (default model 'yolo11x-seg.pt'). 

### Features include:
- Detection using pre-trained or custom models
- Option to detect in specific zones or the entire orthomosaic
- Adjustable detection parameters (confidence threshold, iou threshold, resolution, etc.)
- Results are saved as shapes in the Metashape project and exported in format CSV in your workdir.

  **Output file result csv:**
    - Label
    - Score (avg)
    - Area 2D (m²)
    - Centroid (x,y)
    - Width (m)
    - Length (m)

**Requirements:**
- An active orthomosaic with resolution ≤ 10 cm/pixel

### Configuration Options

The module provides several configuration options:

- **Working Directory**: Directory for temporary files and results
- **Resolution**: Preferred resolution for detection (default 0.5 cm/pix)
- **Debug Mode**: Enabling/disabling debugging information, cut-out objects and their coordinates in the working directory are added to the results of each prediction.
- **Max size tiles**: Maximum size of tiles for processing crop
- **Layer zones**: Polygons for forecasting
- **Model path**: The path to the YOLO model file (you can specify the pre-trained YOLO Ultralitycs model in the format `yolo11x-seg.pt ` - see all models for detection and segmentation on https://docs.ultralytics.com/ru/models /)
- **Detection Parameters**:
  - Confidence Threshold: Minimum confidence score for detections
  - IOU Threshold: The threshold of crossing over the union

### 2. Create YOLO Dataset

Access via: `Scripts > YOLO Tools > Create yolo dataset`

This tool helps you create datasets in YOLO format for training custom models:
- Export orthomosaic tiles with annotations
- Support for data augmentation
- Split data into training and validation sets
- Generate base YAML configuration files for YOLO training

## Configuration Options

The module provides several configuration options:

- **Working Directory**: Directory for temporary files and results
- **Resolution**: Preferred resolution for detection (default 0.5 cm/pix)
- **Debug Mode**: Enabling/disabling debugging information, tile images with annotations are added to the working directory.
- **Max size tiles**: Maximum size of tiles for processing crop
- **Layer zones**: Polygons annotations objects
- **Layer data**: Layer with annotations objects
- **Splitting data**: Specify the percentage of train sets (default train 0.8, val 0.2)
- **Proportion background**: Specify the percentage of background images without objects.
- **Augment data**: Add augmentation. Seven basic image transformations are used:
  - Rotate by 90 degrees
  - Rotate 180 degrees 
  - Rotate 270 degrees
  - Mirror image
  - Mirror image and rotate 90
  - Mirror image and rotate 180
  - Mirror image and rotate 270
- **Random augment colors**: Random augment colors: Random color conversion, adding noise:
  - HueSaturationValue(hue_shift_limit=360, sat_shift_limit=30, val_shift_limit=20)
  - ISONoise(p=0.5)
  - RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0, p=0.5)
- **Mode**: Use boxes or outlines of annotated objects

**Dataset directory structure**
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

**Annotation format**:
- For object detection:
```
     <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```
- For segmentation:
```
     <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
   where `class_id` is the class identifier, and the coordinates are normalized by the size of the image.

**Data.yaml file**:
```yaml
   train: train/images
   val: val/images
   nc: <number_of_classes>
   names: [<class_name_1>, <class_name_2>, ...]
```

## Notes

- The orthomosaic should have a resolution of 10 cm/pixel or better for optimal results
- GPU acceleration is recommended for faster processing
- Custom models can be trained using the dataset creation tool and the Ultralytics YOLO framework

### Processing of large orthomosaic

Orthomosaic created from aerial photography can be huge (tens of thousands of pixels in each dimension). For effective processing of such images, the module uses the following approaches:

1. **Tile division**: The orthomosaic is divided into fixed-size tiles, which are processed separately.
2. **Tile Overlap**: Tiles overlap so that objects located on tile borders are fully visible in at least one tile.
3. **Processing of overlapping detections**: A non-maximum suppression algorithm is applied to eliminate duplicate detections in overlapping areas.

### Coordinate transformation

The module performs several coordinate transformations:

1. **From world coordinates to pixel coordinates**: To extract tiles from an orthomosaic.
2. **From pixel coordinates of the tile to pixel coordinates of the orthomosaic**: To combine detection results from different tiles.
3. **From pixel coordinates of the orthomosaic to world coordinates**: To create vector objects in Metashape.

## Credits

This module is based on:

- [Agisoft Metashape Scripts](https://github.com/agisoft-llc/metashape-scripts/blob/master/src/detect_objects.py)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)