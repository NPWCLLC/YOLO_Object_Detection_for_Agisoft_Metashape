import sys

import Metashape

from PySide2 import QtCore, QtWidgets
import pathlib, os, time

from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import shutil
import yaml
import random
from collections import OrderedDict

IMAGE_FORMATS = ('.jpg', '.jpeg', '.png')
SUBSET_NAMES = ('train', 'val', 'test', 'valid')

def getShapeVertices(shape):
    """
    Gets the vertices of the given shape.

    This function computes and returns a list of vertices for the specified shape. It retrieves marker positions for attached
    shapes or directly uses coordinate values for detached shapes. Transformations are applied to convert marker positions
    to the desired coordinate system when working with attached shapes.

    Parameters:
    shape: Metashape.Shape
        The shape from which vertices are to be extracted. The shape can either be attached or detached.

    Returns:
    list
        A list of vertex points representing the shape's geometry. The points are either transformed marker positions
        (for attached shapes) or coordinate values directly extracted from the shape (for detached shapes).

    Raises:
    Exception
        If the active chunk is null.
    Exception
        If any marker position is invalid within the given shape.
    """
    chunk = Metashape.app.document.chunk
    if chunk is None:
        raise Exception("Null chunk")

    T = chunk.transform.matrix
    result = []

    if shape.is_attached:
        assert (len(shape.geometry.coordinates) == 1)
        for key in shape.geometry.coordinates[0]:
            for marker in chunk.markers:
                if marker.key == key:
                    if (not marker.position):
                        raise Exception("Invalid shape vertex")

                    point = T.mulp(marker.position)
                    point = Metashape.CoordinateSystem.transform(point, chunk.world_crs, chunk.shapes.crs)
                    result.append(point)
    else:
        assert (len(shape.geometry.coordinates) == 1)
        for coord in shape.geometry.coordinates[0]:
            result.append(coord)

    return result

def ensure_unique_directory(base_dir):
    """
    Generates a unique directory name by appending a numeric suffix to the provided base directory
    name if a directory with the same base name already exists. If the base directory does not exist,
    it is returned unchanged.

    Args:
        base_dir (str): The base directory name to check and ensure uniqueness for.

    Returns:
        str: A unique directory name. If the base directory does not exist, the same directory name is
        returned. If it does exist, a unique name with an appended numeric suffix is returned.
    """
    if not os.path.exists(base_dir):
        return base_dir

    counter = 1
    new_dir = f"{base_dir}_{counter}"
    while os.path.exists(new_dir):
        counter += 1
        new_dir = f"{base_dir}_{counter}"

    return new_dir

def remove_directory(directory_path):
    """
    Removes the specified directory and its contents if it exists. If the directory
    doesn't exist, the function does nothing.

    Args:
        directory_path (str): The path to the directory to remove.

    Raises:
        Exception: If an unexpected error occurs while attempting to remove the
        directory.
    """
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    else:
        pass

def count_annotations(subset_path):
    """
    Counts annotations in a dataset subset by analyzing image and label files.
    
    This function processes all images in a subset directory, reads their corresponding
    annotation files, and counts annotations per class as well as empty annotations.
    
    Parameters:
        subset_path (str): Path to the dataset subset directory containing 'images' and 'labels' folders.
    
    Returns:
        tuple: A tuple containing:
            - image_files (list): List of image filenames found in the subset
            - class_counts (dict): Dictionary mapping class IDs to their annotation counts
            - total_annotations (int): Total number of annotations (including empty)
            - empty_annotations (int): Number of images with no annotations
            - empty_annotation_ratio (float): Ratio of empty annotations to total annotations
    """

    class_counts = {}
    total_annotations = 0
    empty_annotations = 0
    image_files = [f for f in os.listdir(os.path.join(subset_path, "images")) if f.endswith(IMAGE_FORMATS)]

    for image_file in image_files:
        annotation_file = f"{pathlib.Path(image_file).stem}.txt"
        annotation_path = os.path.join(subset_path, "labels", annotation_file)

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    empty_annotations += 1
                else:
                    total_annotations += len(lines)
                    for line in lines:
                        class_id = int(line.split()[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                        else:
                            class_counts[class_id] = 1

    total_annotations += empty_annotations

    empty_annotation_ratio = empty_annotations / total_annotations if total_annotations > 0 and empty_annotations > 0  else 0

    return image_files, class_counts, total_annotations, empty_annotations, empty_annotation_ratio

def print_statistics(dataset_dir):
    """
    Prints comprehensive statistics for a YOLO dataset.
    
    This function analyzes and displays statistics for train, validation, and test subsets
    including image counts, annotation counts per class, empty annotations, and ratios
    between different subsets.
    
    Parameters:
        dataset_dir (str): Path to the root directory of the YOLO dataset containing
                          train/val/test subdirectories.
    """

    dataset_folder = pathlib.Path(dataset_dir).name
    print(f"\nFolder dataset: {dataset_folder}")

    empty_annotations_train = 0
    train_non_empty_annotations = 0
    val_non_empty_annotations = 0
    train_images_count = 0
    val_images_count = 0

    for subset in SUBSET_NAMES:
        subset_path = os.path.join(dataset_dir, subset)
        if os.path.exists(subset_path):
            image_files, class_counts, total_annotations, empty_annotations, empty_annotation_ratio = count_annotations(
                subset_path)

            images_count = len(image_files)

            if subset == "train":
                train_non_empty_annotations = total_annotations
                train_images_count = images_count
                empty_annotations_train = empty_annotations
            elif subset in ("val", "valid"):
                val_non_empty_annotations = total_annotations
                val_images_count = images_count

            print(f"\nStatistics {subset}:")
            print(f"Count images: {images_count}")
            print(f"Count images of non-empty annotations: {images_count-empty_annotations}")
            print(f"Total number of annotations: {total_annotations}")
            print(f"Number of empty annotations: {empty_annotations} ({empty_annotation_ratio:.2%})")
            print("Annotations by class:")


            for class_id, count in class_counts.items():
                print(f"Class {class_id}: {count} annotations")


    if train_non_empty_annotations > 0 and val_images_count > 0:
        val_to_train_annotation_ratio = val_non_empty_annotations / train_non_empty_annotations
        print(f"The proportion of non-empty annotations in val relative to train: {val_to_train_annotation_ratio:.2%}")


    if train_images_count > 0 and val_images_count > 0:
        val_to_train_image_ratio = val_images_count / train_images_count
        print(f"The proportion of images in val relative to train: {val_to_train_image_ratio:.2%}")


    print("-"*15)

class NoAliasDumper(yaml.Dumper):
    """
    A YAML dumper class to disable the generation of YAML aliases.

    This class extends `yaml.Dumper` to override its behavior and
    ensure that no aliases are generated while dumping YAML data.
    This is especially useful when dealing with large data
    structures where aliases can compromise clarity and
    maintainability.

    Attributes
    ----------
    None
    """
    def ignore_aliases(self, data):
        return True

class YOLODatasetConverter:
    """
    Converts annotated datasets to YOLO format with train/val split.
    
    This class handles the conversion of annotation data into YOLO-compatible format,
    including splitting the dataset into training and validation subsets, organizing
    files, and generating the required data.yaml configuration file.
    """
    
    def __init__(self, all_annotations, input_dir_images=None, output_dir=None, split_ratios=None, empty_ratio=None,
                 allow_empty_annotations=True, callback=None, mode_converter=None):
        """
        Initializes the YOLO dataset converter with necessary parameters.
        
        Parameters:
            all_annotations (dict): Dictionary containing all annotation data
            input_dir_images (str, optional): Directory containing input images
            output_dir (str, optional): Directory where converted dataset will be saved
            split_ratios (dict, optional): Dictionary with 'train' and 'val' split ratios (default: 0.8/0.2)
            empty_ratio (dict, optional): Dictionary with ratios for empty annotations distribution
            allow_empty_annotations (bool, optional): Whether to include images without annotations (default: True)
            callback (callable, optional): Callback function for logging messages
            mode_converter (str, optional): Conversion mode ('segmentation' for polygon annotations)
        """
        self.all_annotations = all_annotations
        self.input_dir_images = input_dir_images
        self.output_dir = output_dir
        self.split_ratios = split_ratios if split_ratios else {"train": 0.8, "val": 0.2}
        self.empty_ratio = empty_ratio if empty_ratio else {"train": 0.8, "val": 0.2}
        self.allow_empty_annotations = allow_empty_annotations

        self.train_dir = os.path.join(self.output_dir, "train")
        self.val_dir = os.path.join(self.output_dir, "valid")

        self.callback = callback
        self.mode_converter = mode_converter

    def _log(self, message=None):
        """
        Logs a message using callback function or prints to console.
        
        Parameters:
            message (str, optional): Message to log
        """

        if self.callback:
            self.callback(message)
        else:
            print(message)

    def prepare_directories(self):
        """
        Creates the necessary directory structure for YOLO dataset.
        
        Creates train and validation directories with their respective
        'images' and 'labels' subdirectories.
        """

        os.makedirs(os.path.join(self.train_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.val_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.val_dir, "images"), exist_ok=True)

    def visualize_annotations_for_image(self, image, annotations, target_dir):
        """
        Visualizes annotations on an image by drawing bounding boxes and segmentation contours.
        
        This method loads an image, draws bounding boxes and polygon segmentations (if in
        segmentation mode), and saves the visualized image to the target directory.
        
        Parameters:
            image (dict): Image information dictionary with 'id' and 'file_name' keys
            annotations (list): List of annotation dictionaries for all images
            target_dir (str): Directory where visualized image will be saved
        """
        os.makedirs(target_dir, exist_ok=True)
        img_path = os.path.join(self.input_dir_images, image['file_name'])
        output_path = os.path.join(target_dir, image['file_name'])

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}. Skipping.")
            return

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}. Skipping.")
            return

        image_id = image['id']
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

        for annotation in image_annotations:
            bbox = annotation['bbox']  # [xmin, ymin, width, height]
            category_id = annotation['category_id']
            segmentation = annotation['segmentation']

            if bbox:
                x_min, y_min, width, height = bbox
                x_max, y_max = int(x_min + width), int(y_min + height)
                x_min, y_min = int(x_min), int(y_min)

                color = (0, 255, 0)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

                cv2.putText(
                    img,
                    f"Class: {category_id}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

            if self.mode_converter == "segmentation" and segmentation:
                segmentation_points = segmentation
                try:
                    if isinstance(segmentation_points, list) and len(segmentation_points) % 2 == 0 and len(
                            segmentation_points) > 2:

                            points = np.array(segmentation_points, dtype=np.int32).reshape((-1, 2))

                            contour_color = (255, 0, 0)

                            cv2.polylines(img, [points], isClosed=True, color=contour_color, thickness=2)

                    else:
                        print(f"{segmentation_points}")
                except Exception as e:
                    print(f"{e} >>>> {segmentation}")

        cv2.imwrite(output_path, img)

    @staticmethod
    def write_yolo_annotation(output_path, yolo_lines):
        """
        Writes YOLO format annotations to a file.
        
        Parameters:
            output_path (str): Path to the output annotation file
            yolo_lines (list): List of YOLO format annotation strings to write
        """
        with open(output_path, "a") as f:
            for line in yolo_lines:
                f.write(line + "\n")

    def split_dataset(self):
        """
        Splits the dataset into training and validation subsets.
        
        This method separates images with annotations from empty ones, performs
        train/val split, and optionally includes empty images in training set
        based on the specified ratio.
        """

        image_dict = {image['id']: image for image in self.all_annotations['images']}

        annotated_images = set(annotation['image_id'] for annotation in self.all_annotations['annotations'])
        images_with_annotations = [image for image in self.all_annotations['images'] if image['id'] in annotated_images]
        images_without_annotations = [image for image in self.all_annotations['images'] if
                                      image['id'] not in annotated_images]

        print(f"Images with annotations: {len(images_with_annotations)}")
        print(f"Images without annotations: {len(images_without_annotations)}")

        train_with_annotations, val_with_annotations = train_test_split(
            images_with_annotations,
            test_size=self.split_ratios["val"],
            random_state=42
        )

        if self.allow_empty_annotations and images_without_annotations:
            max_train_empty = int(len(train_with_annotations) * self.empty_ratio["train"])



            random.shuffle(images_without_annotations)

            train_empty = images_without_annotations[:min(max_train_empty, len(images_without_annotations))]

            # print(f"empty_ratio['train']: {self.empty_ratio['train']}")
            # print(f"train_with_annotations: {len(train_with_annotations)}")
            # print(f"max_train_empty:{max_train_empty}")
            # print(f"train_empty:{len(train_empty)}")

        else:
            train_empty, val_empty = [], []


        self.train_images = train_with_annotations + train_empty
        self.val_images = val_with_annotations

        print(f"train_images: {len(self.train_images)}")
        print(f"val_images: {len(self.val_images)}")

        self.train_annotations = []
        self.val_annotations = []

        for annotation in self.all_annotations['annotations']:
            image_id = annotation['image_id']
            image_info = image_dict[image_id]

            if image_info in self.train_images:
                self.train_annotations.append(annotation)
            elif image_info in self.val_images:
                self.val_annotations.append(annotation)

    def move_images(self, images, target_dir):
        """
        Copies images from input directory to target directory.
        
        Parameters:
            images (list): List of image dictionaries with 'file_name' keys
            target_dir (str): Target directory where images will be copied
        """
        for image in images:
            source_path = os.path.join(self.input_dir_images, image['file_name'])
            target_path = os.path.join(target_dir, "images", image['file_name'])
            if os.path.exists(source_path):
                shutil.copy(source_path, target_path)
            else:
                print(f"{source_path}")
            self._log()

    def convert_to_yolo_format(self, annotations, target_dir, images):
        """
        Converts annotations to YOLO format and writes them to label files.
        
        This method processes bounding box or segmentation annotations, normalizes
        coordinates, and writes them in YOLO format (class_id + normalized coords).
        
        Parameters:
            annotations (list): List of annotation dictionaries
            target_dir (str): Directory where label files will be saved
            images (list): List of image dictionaries for reference
        """
        image_dict = {image['id']: image for image in images}

        for annotation in annotations:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            image_info = image_dict[image_id]
            img_width = image_info['width']
            img_height = image_info['height']

            yolo_line = ""

            if self.mode_converter == "detect":
                if annotation['bbox']:
                    xmin, ymin, width, height = annotation['bbox']
                    x_center = (xmin + width / 2) / img_width
                    y_center = (ymin + height / 2) / img_height
                    width /= img_width
                    height /= img_height

                    yolo_line = f"{category_id} {x_center:.10f} {y_center:.10f} {width:.10f} {height:.10f}"
            elif self.mode_converter == "segmentation":
                if annotation['segmentation']:
                    segmentation_points = annotation['segmentation']

                    if len(segmentation_points) % 2 != 0 or len(segmentation_points) == 0:
                        print(f" {annotation}")
                        continue

                    normalized_points = [
                        f"{x / img_width:.10f} {y / img_height:.10f}" for x, y in
                        zip(segmentation_points[::2], segmentation_points[1::2])
                    ]
                    points_str = " ".join(normalized_points)
                    yolo_line = f"{category_id} {points_str}"

            output_file = os.path.join(target_dir, "labels", os.path.splitext(image_info['file_name'])[0] + ".txt")
            if os.path.isfile(output_file) and yolo_line == "":
                continue
            self.write_yolo_annotation(output_file, [yolo_line])

            self._log()

        annotation_image_id = [annotation['image_id'] for annotation in annotations]
        for image in images:
            if image['id'] not in annotation_image_id:
                empty_file = os.path.join(target_dir, "labels", os.path.splitext(image['file_name'])[0] + ".txt")
                open(empty_file, "w").close()

    def create_data_yaml(self):
        """
        Creates the data.yaml configuration file for YOLO training.
        
        This file contains paths to train/val directories, number of classes,
        and class names. Required by YOLO for dataset configuration.
        """
        categories = self.all_annotations['categories']
        names = [name for name, index in sorted(categories.items(), key=lambda item: int(item[1]))]
        nc = len(categories)

        data = OrderedDict([
            ("train", "../train/images"),
            ("val", "../valid/images"),
            ("nc", nc),
            ("names", names)
        ])

        yaml_path = os.path.join(self.output_dir, "data.yaml")

        with open(yaml_path, "w") as yaml_file:
            yaml.dump(dict(data), yaml_file, Dumper=NoAliasDumper, default_flow_style=None, sort_keys=False)
            print(f"The data file.yaml has been created successfully: {yaml_path}")

    @staticmethod
    def analyze_txt_files(directory):
        """
        Analyzes annotation text files in a directory and prints statistics.
        
        Counts total, empty, and non-empty label files, and calculates the total
        number of annotation rows across all files.
        
        Parameters:
            directory (str): Directory containing label .txt files to analyze
        """

        total_files = 0
        empty_files = 0
        non_empty_files = 0
        total_sum = 0


        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    total_files += 1
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                            if not lines:
                                empty_files += 1
                            else:
                                non_empty_files += 1

                                try:
                                    total_sum += len(lines)
                                except ValueError:
                                    continue
                    except Exception as e:
                        print(f" {file_path}: {e}")

        empty_percentage = (empty_files / total_files * 100) if total_files > 0 else 0
        non_empty_percentage = (non_empty_files / total_files * 100) if total_files > 0 else 0

        print(f"Total labels: {total_files}")
        print(f"Empty labels: {empty_files} ({empty_percentage:.2f}%)")
        print(f"Total labels with data: {non_empty_files} ({non_empty_percentage:.2f}%)")
        if non_empty_files > 0:
            print(f"Sum all rows in labels: {total_sum}")

    def convert(self):
        """
        Executes the complete dataset conversion pipeline.
        
        This method orchestrates the entire conversion process: creates directories,
        splits the dataset, copies images, converts annotations to YOLO format,
        and creates the data.yaml configuration file.
        """
        self._log("Mode converter: {}".format(self.mode_converter))

        self.prepare_directories()

        self.split_dataset()

        self.move_images(self.train_images, self.train_dir)

        self.move_images(self.val_images, self.val_dir)

        self.convert_to_yolo_format(self.train_annotations, self.train_dir, self.train_images)

        self.convert_to_yolo_format(self.val_annotations, self.val_dir, self.val_images)

        self.create_data_yaml()

        self._log(f"The dataset has been successfully split and converted. The result is saved in: {self.output_dir}")
        self._log(f"Train images: {len(self.train_images)} | Val images: {len(self.val_images)}")

class WindowCreateYoloDataset(QtWidgets.QDialog):
    """
    Dialog window for creating YOLO datasets from orthomosaic images.
    
    This class provides a GUI for configuring and generating YOLO-format datasets
    from Agisoft Metashape orthomosaic data with annotated shapes.
    """

    def __init__(self, parent):
        """
        Initializes the YOLO dataset creation dialog window.
        
        Sets up default parameters, loads settings from application configuration,
        initializes the GUI, and prepares the working environment.
        
        Parameters:
            parent: Parent widget for this dialog.
        """

        QtWidgets.QDialog.__init__(self, parent)
        self.shapes = []
        self.setWindowTitle("Create Yolo dataset on orthomosaic")

        self.patch_size = None
        self.orthomosaic_resolution = None
        self.patch_inner_border = None
        self.train_on_user_data_enabled = None
        self.train_data = None
        self.train_zones = None
        self.force_small_patch_size = True
        self.max_image_size = None
        self.train_percentage = 0.8
        self.percent_empty_limit = 0.4

        self.augment_colors = False
        self.cleanup_working_dir = False
        self.isDebugMode = False
        self.selected_mode = "detect"

        self.preferred_patch_size = 640
        self.preferred_resolution = 0.005
        self.isAugmentData = False

        self.prefer_original_resolution = False

        self.expected_layer_name_train_zones = "Zone"
        self.expected_layer_name_train_data = "Data"

        if len(Metashape.app.document.path) > 0:
            self.working_dir = str(pathlib.Path(Metashape.app.document.path).parent / "dataset_yolo")
        else:
            self.working_dir = ""

        self.chunk = Metashape.app.document.chunk
        self.create_gui()

        max_image_size = Metashape.app.settings.value("scripts/create_yolo_dataset/max_image_size")
        self.max_image_size = int(max_image_size) if max_image_size else 640
        self.maxSizeImageSpinBox.setValue(self.max_image_size)

        percent_empty_limit = Metashape.app.settings.value("scripts/create_yolo_dataset/percent_empty_limit")
        self.percent_empty_limit = float(percent_empty_limit) if percent_empty_limit else 0.4
        self.proportionBackgroundSpinBox.setValue(self.percent_empty_limit)

        train_percentage = Metashape.app.settings.value("scripts/create_yolo_dataset/train_percentage")
        self.train_percentage = float(train_percentage) if train_percentage else 0.8
        self.separationDataSpinBox.setValue(self.train_percentage)

        self.exec()

    def stop(self):
        """
        Stops the running process by setting the stopped state to True.
        """
        self.stopped = True

    def check_stopped(self):
        """
        Checks if stop was requested and raises an exception if so.
        
        Raises:
            InterruptedError: If the stopped attribute is True.
        """
        if self.stopped:
            raise InterruptedError("Stop was pressed")

    def run_process(self):
        """
        Executes the main dataset creation process.
        
        This method orchestrates the entire workflow: loading parameters, preparing
        directories, exporting orthomosaic, creating dataset from user data, and
        displaying results.
        """
        try:
            self.stopped = False
            self.btnRun.setEnabled(False)
            self.btnStop.setEnabled(True)

            time_start = time.time()

            self.load_params()

            self.prepare()

            print("Script started...")

            self.export_orthomosaic()

            if self.chunk.shapes is None:
                self.chunk.shapes = Metashape.Shapes()
                self.chunk.shapes.crs = self.chunk.crs

            if self.train_on_user_data_enabled:
                self.create_on_user_data()
            else:
                msg_box = QtWidgets.QMessageBox()
                msg_box.setIcon(QtWidgets.QMessageBox.Warning)
                msg_box.setWindowTitle("")
                msg_box.setText("Please select the zone layer")
                # Execute dialog
                msg_box.exec_()



            self.results_time_total = time.time() - time_start
            self.show_results_dialog()
        except Exception as ex:
            print(ex)
            if self.stopped:
                Metashape.app.messageBox("Processing was stopped.")
            else:
                Metashape.app.messageBox("Something gone wrong.\n"
                                         "Please check the console.")
                raise
        finally:
            remove_directory(self.dir_inner_data)
            self.btnStop.setEnabled(False)
            self.btnRun.setEnabled(True)
            self.reject()

        print("Script finished.")
        return True

    def prepare(self):
        """
        Prepares the working directory structure for dataset creation.
        
        Creates necessary subdirectories for tiles, datasets, inner data, and
        debug outputs if debug mode is enabled.
        
        Raises:
            Exception: If working directory is not specified.
        """
        if self.working_dir == "":
            raise Exception("You should specify working directory (or save .psx project)")

        os.makedirs(self.working_dir, exist_ok=True)
        print("Working dir: {}".format(self.working_dir))

        self.cleanup_working_dir = False

        self.dir_tiles = self.working_dir + "/tiles/"
        self.dir_data = ensure_unique_directory(self.working_dir + f"/datasets_{self.selected_mode}")
        self.dir_inner_data = os.path.join(self.dir_data, "inner")
        self.dir_inner_images = os.path.join(self.dir_inner_data, "images")

        self.dir_train_subtiles_debug_dataset = os.path.join(self.dir_data, "debug_dataset")

        create_dirs = [self.dir_tiles, self.dir_data, self.dir_inner_images,]
        if self.isDebugMode:
            create_dirs.append(self.dir_train_subtiles_debug_dataset)

        for subdir in create_dirs:
            os.makedirs(subdir, exist_ok=True)

    def export_orthomosaic(self):
        """
        Exports the orthomosaic as tiles for dataset creation.
        
        This method exports the orthomosaic from Metashape into tiles, handles
        existing tiles by asking user preference, and builds tile-to-world
        transformation matrices for coordinate mapping.
        
        Raises:
            Exception: If no tiles are found after export.
        """


        print("Preparing orthomosaic...")

        kwargs = {}
        if not self.prefer_original_resolution:
            kwargs["resolution"] = self.preferred_resolution
        else:
            print("no resolution downscaling required")

        tiles = os.listdir(self.dir_tiles)
        if tiles:
            # Tiles already exist, ask user what to do
            buttons = [
                ("Use Existing Tiles", QtWidgets.QMessageBox.YesRole),
                ("Delete and Create New", QtWidgets.QMessageBox.NoRole)
            ]
            choice = self.show_question_dialog(
                title="Existing Tiles Found",
                text="Tiles already exist in the directory.",
                informative_text="Do you want to use existing tiles or delete them and create new ones?",
                buttons=buttons,
                default_button_index=0
            )

            if choice == 1:  # "Delete and Create New" was clicked
                # Delete existing tiles
                print("Deleting existing tiles...")
                for tile in tiles:
                    tile_path = os.path.join(self.dir_tiles, tile)
                    try:
                        os.remove(tile_path)
                    except Exception as e:
                        print(f"Error deleting tile {tile}: {e}")

                # Create new tiles
                self.chunk.exportRaster(path=self.dir_tiles + "tile.jpg",
                                        source_data=Metashape.OrthomosaicData,
                                        image_format=Metashape.ImageFormat.ImageFormatJPEG,
                                        save_alpha=False,
                                        white_background=True,
                                        save_world=True,
                                        split_in_blocks=True,
                                        block_width=self.patch_size,
                                        block_height=self.patch_size,
                                        **kwargs)
            else:
                print("Using existing tiles...")
        else:
            # No tiles exist, create them
            self.chunk.exportRaster(path=self.dir_tiles + "tile.jpg",
                                    source_data=Metashape.OrthomosaicData,
                                    image_format=Metashape.ImageFormat.ImageFormatJPEG,
                                    save_alpha=False,
                                    white_background=True,
                                    save_world=True,
                                    split_in_blocks=True,
                                    block_width=self.patch_size,
                                    block_height=self.patch_size,
                                    **kwargs)

        tiles = os.listdir(self.dir_tiles)
        if not tiles:
            raise Exception("No tiles found in the directory.")

        self.tiles_paths = {}
        self.tiles_to_world = {}

        for tile in sorted(tiles):
            assert tile.startswith("tile-")

            _, tile_x, tile_y = tile.split(".")[0].split("-")
            tile_x, tile_y = map(int, [tile_x, tile_y])
            if tile.endswith(".jgw") or tile.endswith(".pgw"):  # https://en.wikipedia.org/wiki/World_file
                with open(self.dir_tiles + tile, "r") as file:
                    matrix2x3 = list(map(float, file.readlines()))
                matrix2x3 = np.array(matrix2x3).reshape(3, 2).T
                self.tiles_to_world[tile_x, tile_y] = matrix2x3
            elif tile.endswith(".jpg"):
                self.tiles_paths[tile_x, tile_y] = self.dir_tiles + tile

        assert (len(self.tiles_paths) == len(self.tiles_to_world))
        assert (self.tiles_paths.keys() == self.tiles_to_world.keys())

        self.tile_min_x = min([key[0] for key in self.tiles_paths.keys()])
        self.tile_max_x = max([key[0] for key in self.tiles_paths.keys()])
        self.tile_min_y = min([key[1] for key in self.tiles_paths.keys()])
        self.tile_max_y = max([key[1] for key in self.tiles_paths.keys()])
        print("{} tiles, tile_x in [{}; {}], tile_y in [{}; {}]".format(len(self.tiles_paths), self.tile_min_x,
                                                                        self.tile_max_x, self.tile_min_y,
                                                                        self.tile_max_y))

    def add_boxes_zone_tiles(self, tiles_data, shapes_group):
        """
        Adds bounding box shapes to the shapes group for debugging visualization.
        
        Parameters:
            tiles_data (list): List of dictionaries containing tile bbox data
            shapes_group: Metashape shapes group to add boxes to
        """


        for row in tiles_data:
            xmin, ymin, xmax, ymax, label, zone_to_world = (int(row["xmin"]),
                                                            int(row["ymin"]),
                                                            int(row["xmax"]),
                                                            int(row["ymax"]),
                                                            row["label"],
                                                            row["zone_to_world"])

            corners = []
            for x, y in [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]:
                # x, y = zone_to_world @ np.array([x + 0.5, y + 0.5, 1]).reshape(3, 1)
                # p = Metashape.Vector([x, y])
                vec = zone_to_world @ np.array([x + 0.5, y + 0.5, 1])
                p = Metashape.Vector([vec[0].item(), vec[1].item()])

                p = Metashape.CoordinateSystem.transform(p, self.chunk.orthomosaic.crs, self.chunk.shapes.crs)
                corners.append([p.x, p.y])

            shape = self.chunk.shapes.addShape()
            shape.group = shapes_group
            shape.geometry = Metashape.Geometry.Polygon(corners)
            shape.label = label

    def custom_callback(self, message=None):
        """
        Custom callback for progress updates and interruption checks.
        
        Prints message, updates Metashape application, processes Qt events,
        and checks if stop was requested.
        
        Parameters:
            message (str, optional): Message to print
        """
        app = QtWidgets.QApplication.instance()
        if message:
            print(message)
        Metashape.app.update()
        app.processEvents()
        self.check_stopped()

    def create_on_user_data(self):

        from concurrent.futures import ThreadPoolExecutor



        random.seed(2391231231324531)

        app = QtWidgets.QApplication.instance()

        training_start = time.time()

        num_cores = os.cpu_count()
        executor = ThreadPoolExecutor(max_workers=num_cores)

        print(f'Max size tiles: {self.max_image_size}')

        tales_boxes_data = []
        nannotations = 1

        self.train_zones_on_ortho = []

        n_train_zone_shapes_out_of_orthomosaic = 0
        for zone_i, shape in enumerate(self.train_zones):
            shape_vertices = getShapeVertices(shape)
            zone_from_world = None
            zone_from_world_best = None
            zone_to_world = None
            for tile_x in range(self.tile_min_x, self.tile_max_x + 1):
                for tile_y in range(self.tile_min_y, self.tile_max_y + 1):
                    if (tile_x, tile_y) not in self.tiles_paths:
                        continue
                    to_world = self.tiles_to_world[tile_x, tile_y]
                    from_world = self.invert_matrix_2x3(to_world)
                    for p in shape_vertices:
                        p = Metashape.CoordinateSystem.transform(p, self.chunk.shapes.crs,
                                                                 self.chunk.orthomosaic.crs)
                        p_in_tile = from_world @ [p.x, p.y, 1]
                        distance2_to_tile_center = np.linalg.norm(
                            p_in_tile - [self.patch_size / 2, self.patch_size / 2])
                        if zone_from_world_best is None or distance2_to_tile_center < zone_from_world_best:
                            zone_from_world_best = distance2_to_tile_center
                            zone_from_world = self.invert_matrix_2x3(
                                self.add_pixel_shift(to_world, -tile_x * self.patch_size,
                                                     -tile_y * self.patch_size))
                            zone_to_world = self.add_pixel_shift(to_world, -tile_x * self.patch_size,
                                                                 -tile_y * self.patch_size)
            if zone_from_world_best > 1.1 * (self.patch_size / 2) ** 2:
                n_train_zone_shapes_out_of_orthomosaic += 1

            zone_from = None
            zone_to = None
            for p in shape_vertices:
                p = Metashape.CoordinateSystem.transform(p, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                p_in_ortho = np.int32(np.round(zone_from_world @ [p.x, p.y, 1]))
                if zone_from is None:
                    zone_from = p_in_ortho
                if zone_to is None:
                    zone_to = p_in_ortho
                zone_from = np.minimum(zone_from, p_in_ortho)
                zone_to = np.maximum(zone_to, p_in_ortho)
            train_size = zone_to - zone_from
            train_size_m = np.int32(np.round(train_size * self.orthomosaic_resolution))
            if np.any(train_size < self.patch_size):
                print("Zone #{} {}x{} pixels ({}x{} meters) is too small - each side should be at least {} meters"
                      .format(zone_i + 1, train_size[0], train_size[1], train_size_m[0], train_size_m[1],
                              self.patch_size * self.orthomosaic_resolution), file=sys.stderr)
                self.train_zones_on_ortho.append(None)
            else:
                print("Zone #{}: {}x{} orthomosaic pixels, {}x{} meters".format(zone_i + 1, train_size[0],
                                                                                train_size[1], train_size_m[0],
                                                                                train_size_m[1]))
                self.train_zones_on_ortho.append((zone_from, zone_to, zone_from_world, zone_to_world))
        assert len(self.train_zones_on_ortho) == len(self.train_zones)

        if n_train_zone_shapes_out_of_orthomosaic > 0:
            print(f"Warning, {n_train_zone_shapes_out_of_orthomosaic} train zones shapes are out of orthomosaic")

        area_threshold = 0.3

        all_annotations = {"images": [],
                           "annotations": [],
                           "categories": {}
                           }

        image_id = 1

        self.train_nannotations_in_zones = 0
        id_label = 0
        classes = {}

        for zone_i, shape in enumerate(self.train_zones):
            if self.train_zones_on_ortho[zone_i] is None:
                continue

            self.txtPBar.setText(f"Create dataset (zones {zone_i + 1} of {len(self.train_zones)}):")
            app.processEvents()
            self.check_stopped()

            zone_from, zone_to, zone_from_world, zone_to_world = self.train_zones_on_ortho[zone_i]
            annotations = []
            annotations_boxes = []

            for annotation in self.train_data:
                app.processEvents()
                self.check_stopped()

                annotation_vertices = getShapeVertices(annotation)
                annotation_from = None
                annotation_to = None
                poly = []
                for p in annotation_vertices:
                    p = Metashape.CoordinateSystem.transform(p, self.chunk.shapes.crs, self.chunk.orthomosaic.crs)
                    p_in_ortho = np.int32(np.round(zone_from_world @ [p.x, p.y, 1]))

                    if annotation_from is None:
                        annotation_from = p_in_ortho
                    if annotation_to is None:
                        annotation_to = p_in_ortho

                    poly.append(p_in_ortho)
                    annotation_from = np.minimum(annotation_from, p_in_ortho)
                    annotation_to = np.maximum(annotation_to, p_in_ortho)

                bbox_from, bbox_to = self.intersect(zone_from, zone_to, annotation_from, annotation_to)
                if self.area(bbox_from, bbox_to) > self.area(annotation_from, annotation_to) * area_threshold:
                    _label = annotation.label
                    if _label not in classes:
                        classes[_label] = id_label
                        id_label += 1

                    try:
                        annotations_boxes.append({"box": (annotation_from, annotation_to), "category": classes[_label]})
                        annotations.append(poly)
                    except ValueError:
                        raise ValueError(f"The label annotation of {_label} was not found in the classes list")

            all_annotations['categories'] = classes
            self.train_nannotations_in_zones += len(annotations)
            print(f"\nCreate dataset zone #{zone_i + 1}: {len(annotations)} annotations inside")

            border = self.patch_inner_border
            inner_path_size = self.patch_size - 2 * border

            zone_size = zone_to - zone_from
            assert np.all(zone_size >= self.patch_size)
            nx_tiles, ny_tiles = np.int32((zone_size - 2 * border + inner_path_size - 1) // inner_path_size)
            assert nx_tiles >= 1 and ny_tiles >= 1
            xy_step = np.int32(np.round((zone_size + [nx_tiles, ny_tiles] - 1) // [nx_tiles, ny_tiles]))

            out_of_orthomosaic_train_tile = 0
            total_steps = nx_tiles * ny_tiles

            for x_tile in range(0, nx_tiles):
                for y_tile in range(0, ny_tiles):

                    current_step = x_tile * ny_tiles + y_tile + 1
                    progress = (current_step * 100) / total_steps
                    self.progressBar.setValue(progress)
                    app.processEvents()

                    tile_to = zone_from + self.patch_size + xy_step * [x_tile, y_tile]
                    if x_tile == nx_tiles - 1 and y_tile == ny_tiles - 1:
                        assert np.all(tile_to >= zone_to)
                    tile_to = np.minimum(tile_to, zone_to)
                    tile_from = tile_to - self.patch_size
                    if x_tile == 0 and y_tile == 0:
                        assert np.all(tile_from == zone_from)
                    assert np.all(tile_from >= zone_from)

                    tile = self.read_part(tile_from, tile_to)
                    label_tile = f"{(zone_i + 1)}-{x_tile}-{y_tile}"

                    assert tile.shape == (self.patch_size, self.patch_size, 3)

                    white_pixels_fraction = np.sum(np.all(tile == 255, axis=-1)) / (
                            tile.shape[0] * tile.shape[1])
                    if np.all(tile == 255) or white_pixels_fraction >= 0.90:
                        out_of_orthomosaic_train_tile += 1
                        continue

                    tales_boxes_data.append(
                        {"xmin": tile_from[0], "ymin": tile_from[1], "xmax": tile_to[0], "ymax": tile_to[1],
                         "label": label_tile, "zone_to_world": zone_to_world})

                    tile_annotations_boxes = []
                    tile_annotations = []

                    for idx, ann in enumerate(annotations_boxes):
                        annotation_from, annotation_to = ann["box"]
                        category = ann["category"]
                        bbox_from, bbox_to = self.intersect(tile_from, tile_to, annotation_from, annotation_to)
                        if self.area(bbox_from, bbox_to) > self.area(annotation_from,
                                                                     annotation_to) * area_threshold:
                            tile_annotations_boxes.append(
                                {"box": (bbox_from - tile_from, bbox_to - tile_from), "category": category})
                            tile_annotations.append(np.array(annotations[idx]) - np.array(tile_from))


                    transformations = [
                        (False, 0),
                        (False, 1),
                        (False, 2),
                        (False, 3),
                        (True, 0),
                        (True, 1),
                        (True, 2),
                        (True, 3)
                    ]
                    if self.isAugmentData:
                        transformations = transformations[0:8]
                    else:
                        transformations = transformations[0:1]

                    version_i = 0
                    for is_mirrored, n90rotation in transformations:
                        tile_version = tile.copy()
                        tile_annotations_version = [item["box"] for item in tile_annotations_boxes]
                        tile_annotations_mask_version = tile_annotations.copy()

                        if is_mirrored:
                            if tile_annotations:
                                tile_annotations_version, tile_annotations_mask_version = self.flip_annotations(
                                    tile_annotations_version, tile_annotations_mask_version, tile_version)
                            tile_version = cv2.flip(tile_version, 1)

                        for _ in range(n90rotation):
                            if tile_annotations:
                                tile_annotations_version, tile_annotations_mask_version = self.rotate90clockwise_annotations(
                                    tile_annotations_version, tile_annotations_mask_version, tile_version
                                )
                            tile_version = cv2.rotate(tile_version, cv2.ROTATE_90_CLOCKWISE)

                        if self.augment_colors:
                            tile_version = self.random_augmentation(tile_version)

                        h, w, cn = tile_version.shape

                        tile_name = f"{(zone_i + 1)}-{x_tile}-{y_tile}-{version_i}_{image_id}.jpg"

                        row_img = {"id": image_id, "file_name": tile_name,
                                   "height": h, "width": w}
                        all_annotations["images"].append(row_img)

                        boxes = []
                        masks = []


                        if tile_annotations_version:
                            for idx_an, (_min, _max) in enumerate(tile_annotations_version):

                                contour = [[point[0], point[1]] for point in tile_annotations_mask_version[idx_an]]
                                contour = self.correct_contour(contour, w, h)
                                xmin, ymin, xmax, ymax = self.get_bounding_box(contour)

                                coords_array = np.array(contour)
                                flat_coords = coords_array.flatten().tolist()

                                boxes.append([xmin, ymin, xmax, ymax])
                                masks.append(flat_coords)

                                row_annatation = {"image_id": image_id,
                                                  "id": nannotations,
                                                  "category_id": tile_annotations_boxes[idx_an]["category"],
                                                  "bbox": [xmin, ymin, xmax-xmin, ymax-ymin],
                                                  "segmentation": flat_coords if flat_coords else None,
                                                  "area": (xmax - xmin) * (ymax - ymin),
                                                  "iscrowd": 0}
                                all_annotations["annotations"].append(row_annatation)
                                nannotations += 1


                        else:
                            # row_annatation = {"image_id": image_id,
                            #                   "id": nannotations,
                            #                   "category_id": 0,
                            #                   "bbox": [],
                            #                   "segmentation": [],
                            #                   "area": 0,
                            #                   "iscrowd": 0}
                            # all_annotations["annotations"].append(row_annatation)
                            # nannotations += 1
                            pass

                        image_id += 1
                        executor.submit(cv2.imwrite, os.path.join(self.dir_inner_images, tile_name), tile_version)

                        self.check_stopped()
                        if self.isDebugMode:
                            tile_with_entity = self.debug_draw_objects(tile_version,
                                                                    boxes,
                                                                    masks)
                            os.makedirs(os.path.join(self.dir_train_subtiles_debug_dataset,"inner"), exist_ok=True)
                            executor.submit(cv2.imwrite, os.path.join(self.dir_train_subtiles_debug_dataset,"inner",tile_name),
                                                 tile_with_entity)


                    self.check_stopped()

            if out_of_orthomosaic_train_tile == total_steps:
                raise RuntimeError(
                    f"It seems that zone #{zone_i + 1} has no orthomosaic data, please check zones, orthomosaic and its Outer Boundary.")


        executor.shutdown(wait=True)

        random.shuffle(all_annotations["images"])
        random.shuffle(all_annotations["annotations"])


        print(f"Total images in annotations: {len(all_annotations['images'])}")
        print(f"Total annotations: {len(all_annotations['annotations'])}")
        print(f"Classes: {all_annotations['categories']}")

        if self.isDebugMode:
            detected_shapes_layer = self.chunk.shapes.addGroup()
            detected_shapes_layer.label = "Tiles Boxes"
            detected_shapes_layer.show_labels = False
            self.add_boxes_zone_tiles(tales_boxes_data, detected_shapes_layer)

        print("\n>>> Create dataset...")

        val_pr = 1-self.train_percentage
        converter = YOLODatasetConverter(
            all_annotations,
            input_dir_images=self.dir_inner_images,
            output_dir=self.dir_data,
            split_ratios={"train": self.train_percentage, "val": val_pr if val_pr > 0 else None},
            empty_ratio={"train": self.percent_empty_limit, "val": 1-self.percent_empty_limit},
            allow_empty_annotations=True,
            callback=self.custom_callback,
            mode_converter=self.selected_mode,

        )


        Metashape.app.update()
        app.processEvents()
        self.check_stopped()

        converter.convert()

        if self.isDebugMode:
            for i, image in enumerate(converter.train_images):
                converter.visualize_annotations_for_image(
                    image=image,
                    annotations=converter.train_annotations,
                    target_dir=os.path.join(self.dir_train_subtiles_debug_dataset, "train_annotated_images")

                )

            for i, image in enumerate(converter.val_images):
                converter.visualize_annotations_for_image(
                    image=image,
                    annotations=converter.val_annotations,
                    target_dir=os.path.join(self.dir_train_subtiles_debug_dataset, "val_annotated_images")
                )


        # print("Dataset prepared:")
        # print("Train:")
        # converter.analyze_txt_files(os.path.join(converter.train_dir, "labels"))
        # print("Val:")
        # converter.analyze_txt_files(os.path.join(converter.val_dir, "labels"))

        print_statistics(self.dir_data)

        self.results_time_training = time.time() - training_start

    def show_question_dialog(self, title, text, informative_text, buttons, default_button_index=0):
        """
        Displays a universal question dialog with custom buttons.

        This method creates and displays a QMessageBox with a question icon, allowing
        the user to choose between multiple options. It's a reusable utility method
        for showing dialogs throughout the application.

        Parameters:
            title (str): The window title of the message box.
            text (str): The main text displayed in the message box.
            informative_text (str): Additional informative text displayed below the main text.
            buttons (list of tuple): A list of tuples where each tuple contains:
                - button_text (str): The text to display on the button
                - button_role (QtWidgets.QMessageBox.ButtonRole): The role of the button
            default_button_index (int, optional): The index of the button to set as default.
                Defaults to 0 (first button).

        Returns:
            int: The index of the clicked button in the buttons list.

        """

        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Question)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setInformativeText(informative_text)

        # Add all buttons and store references
        button_widgets = []
        for button_text, button_role in buttons:
            btn = msg_box.addButton(button_text, button_role)
            button_widgets.append(btn)

        # Set default button
        if 0 <= default_button_index < len(button_widgets):
            msg_box.setDefaultButton(button_widgets[default_button_index])

        # Execute dialog
        msg_box.exec_()

        # Find which button was clicked and return its index
        clicked_button = msg_box.clickedButton()
        for i, btn in enumerate(button_widgets):
            if btn == clicked_button:
                return i

        # Should not reach here, but return 0 as fallback
        return 0

    @staticmethod
    def flip_annotations(bboxes, contours, img):
        h, w, _ = img.shape
        flipped_entity = []
        flipped_contours = []

        for bbox_from, bbox_to in bboxes:
            (xmin, ymin), (xmax, ymax) = bbox_from, bbox_to
            flipped_entity.append(((w - xmax, ymin), (w - xmin, ymax)))


        for contour in contours:
            flipped_contour = []
            for x, y in contour:
                flipped_contour.append((w - x, y))
            flipped_contours.append(flipped_contour)

        return flipped_entity, flipped_contours

    def rotate90clockwise_annotations(self, entity, contours, img):
        h, w, _ = img.shape
        rotated_entity = []
        rotated_contours = []

        for bbox_from, bbox_to in entity:
            (xmin, ymin), (xmax, ymax) = bbox_from, bbox_to
            xmin2, ymin2 = self.rotate90clockwise_point(xmin, ymin, w, h)
            xmax2, ymax2 = self.rotate90clockwise_point(xmax, ymax, w, h)
            rotated_entity.append(((xmin2, ymin2), (xmax2, ymax2)))

        for contour in contours:
            rotated_contour = []
            for x, y in contour:
                x_new, y_new = self.rotate90clockwise_point(x, y, w, h)
                rotated_contour.append((x_new, y_new))
            rotated_contours.append(rotated_contour)

        return rotated_entity, rotated_contours

    def rotate90clockwise_point(self, x, y, w, h):
        return h - y, x

    @staticmethod
    def get_bounding_box(points):
        if not points:
            return []

        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for (x, y) in points:
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        return [min_x, min_y, max_x, max_y]

    def correct_contour(self, contour, width, height):
        contour = self.interpolate_contour(contour)
        corrected_contour = []

        def is_inside(_x, _y):
            return 0 <= _x < width and 0 <= _y < height

        def correct_point(_x, _y, _w, _h):
            corrected_x = min(max(_x, 0), _w)
            corrected_y = min(max(_y, 0), _h)
            return corrected_x, corrected_y

        inside_state = False

        for (x, y) in contour:
            if is_inside(x, y):
                corrected_contour.append((x, y))
                inside_state = True
            else:
                if inside_state:
                    corrected_contour.append(correct_point(x, y, width, height))
                    inside_state = False

                if x > width and y > height:
                    if (width, height) not in corrected_contour:
                        corrected_contour.append((width, height))
                elif x > width and y < 0:
                    if (width, 0) not in corrected_contour:
                        corrected_contour.append((width, 0))
                elif x < 0 and y > height:
                    if (0, height) not in corrected_contour:
                        corrected_contour.append((0, height))
                elif x < 0 and y < 0:
                    if (0, 0) not in corrected_contour:
                        corrected_contour.append((0, 0))

        if corrected_contour and corrected_contour[0] != corrected_contour[-1]:
            corrected_contour.append(corrected_contour[0])

        return corrected_contour

    @staticmethod
    def interpolate_contour(contour, step=0.5):

        interpolated_contour = []

        for i in range(len(contour)):
            x1, y1 = contour[i]
            x2, y2 = contour[(i + 1) % len(contour)]

            interpolated_contour.append((x1, y1))

            dx = x2 - x1
            dy = y2 - y1
            distance = np.hypot(dx, dy)

            if distance > step:
                num_points = int(np.floor(distance / step))
                new_x = np.linspace(x1, x2, num=num_points, endpoint=False)
                new_y = np.linspace(y1, y2, num=num_points, endpoint=False)

                for nx, ny in zip(new_x[1:], new_y[1:]):
                    interpolated_contour.append((int(round(nx)), int(round(ny))))

        return interpolated_contour

    def random_augmentation(self, img):
        import albumentations as A

        stages = []
        if self.augment_colors:
            stages.append(
                A.HueSaturationValue(hue_shift_limit=360, sat_shift_limit=30, val_shift_limit=20))
            stages.append(A.ISONoise(p=0.5))
            stages.append(A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0, p=0.5))

            random.shuffle(stages)

            transform = A.Compose(stages)

            img = transform(image=img)["image"]
        return img

    def read_part(self, res_from, res_to):

        res_size = res_to - res_from
        assert np.all(res_size >= [self.patch_size, self.patch_size])
        res = np.zeros((res_size[1], res_size[0], 3), np.uint8)
        res[:, :, :] = 255

        tile_xy_from = np.int32(res_from // self.patch_size)
        tile_xy_upto = np.int32((res_to - 1) // self.patch_size)
        assert np.all(tile_xy_from <= tile_xy_upto)
        for tile_x in range(tile_xy_from[0], tile_xy_upto[0] + 1):
            for tile_y in range(tile_xy_from[1], tile_xy_upto[1] + 1):
                if (tile_x, tile_y) not in self.tiles_paths:
                    continue
                part = cv2.imread(self.tiles_paths[tile_x, tile_y])
                part = cv2.copyMakeBorder(part, 0, self.patch_size - part.shape[0], 0, self.patch_size - part.shape[1],
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])
                part_from = np.int32([tile_x, tile_y]) * self.patch_size - res_from
                part_to = part_from + self.patch_size

                res_inner_from = np.int32([max(0, part_from[0]), max(0, part_from[1])])
                res_inner_to = np.int32([min(part_to[0], res_size[0]), min(part_to[1], res_size[1])])

                part_inner_from = res_inner_from - part_from
                part_inner_to = part_inner_from + res_inner_to - res_inner_from

                res[res_inner_from[1]:res_inner_to[1], res_inner_from[0]:res_inner_to[0], :] = part[part_inner_from[1]:
                                                                                                    part_inner_to[1],
                                                                                               part_inner_from[0]:
                                                                                               part_inner_to[0], :]

        return res

    def intersect(self, a_from, a_to, b_from, b_to):

        c_from = np.maximum(a_from, b_from)
        c_to = np.minimum(a_to, b_to)
        if np.any(c_from >= c_to):
            return c_from, c_from
        else:
            return c_from, c_to

    def area(self, a_from, a_to):
        a_size = a_to - a_from
        return a_size[0] * a_size[1]

    def add_pixel_shift(self, to_world, dx, dy):
        to_world = to_world.copy()
        to_world[0, 2] = to_world[0, :] @ [dx, dy, 1]
        to_world[1, 2] = to_world[1, :] @ [dx, dy, 1]
        return to_world

    def invert_matrix_2x3(self, to_world):

        to_world33 = np.vstack([to_world, [0, 0, 1]])
        from_world = np.linalg.inv(to_world33)

        assert (from_world[2, 0] == from_world[2, 1] == 0)
        assert (from_world[2, 2] == 1)
        from_world = from_world[:2, :]

        return from_world

    def show_results_dialog(self):
        message = "Finished in {:.2f} sec:\n".format(self.results_time_total)

        print(message)
        Metashape.app.messageBox(message)

    def create_gui(self):
        locale = QtCore.QLocale(QtCore.QLocale.C)
        self.labelTrainZonesLayer = QtWidgets.QLabel("Layer zones:")
        self.trainZonesLayer = QtWidgets.QComboBox()
        self.labelTrainDataLayer = QtWidgets.QLabel("Layer data:")
        self.trainDataLayer = QtWidgets.QComboBox()

        self.noTrainDataChoice = (None, "Please select...", True)
        self.layers = [self.noTrainDataChoice]

        if self.chunk.shapes is None:
            print("No shapes")
        else:
            for layer in self.chunk.shapes.groups:
                key, label, enabled = layer.key, layer.label, layer.enabled
                if not enabled:
                    continue
                print("Shape layer: key={}, label={}, enabled={}".format(key, label, enabled))
                if label == '':
                    label = 'Layer'
                self.layers.append((key, label, layer.enabled))

        for key, label, enabled in self.layers:
            self.trainZonesLayer.addItem(label)
            self.trainDataLayer.addItem(label)

        self.trainZonesLayer.setCurrentIndex(0)
        self.trainDataLayer.setCurrentIndex(0)

        for i, (key, label, enabled) in enumerate(self.layers):

            if  self.expected_layer_name_train_zones.lower() in label.lower():
                self.trainZonesLayer.setCurrentIndex(i)
            if self.expected_layer_name_train_data.lower() in label.lower():
                self.trainDataLayer.setCurrentIndex(i)

        self.chkUse5mmResolution = QtWidgets.QCheckBox("Process with 0.50 cm/pix resolution")
        self.chkUse5mmResolution.setToolTip(
            "Process with downsampling to 0.50 cm/pix instad of original orthomosaic resolution.")
        self.chkUse5mmResolution.setChecked(not self.prefer_original_resolution)

        self.groupBoxGeneral = QtWidgets.QGroupBox("General")
        generalLayout = QtWidgets.QGridLayout()

        self.labelWorkingDir = QtWidgets.QLabel()
        self.labelWorkingDir.setText("Working dir:")
        self.workingDirLineEdit = QtWidgets.QLineEdit()
        self.workingDirLineEdit.setText(self.working_dir)
        self.workingDirLineEdit.setPlaceholderText("Path to dir for intermediate data")
        self.workingDirLineEdit.setToolTip("Path to dir for intermediate data")
        self.btnWorkingDir = QtWidgets.QPushButton("...")
        self.btnWorkingDir.setFixedSize(25, 25)

        QtCore.QObject.connect(self.btnWorkingDir, QtCore.SIGNAL("clicked()"), lambda: self.choose_working_dir())
        generalLayout.addWidget(self.labelWorkingDir, 0, 0)
        generalLayout.addWidget(self.workingDirLineEdit, 0, 1)
        generalLayout.addWidget(self.btnWorkingDir, 0, 2)
        generalLayout.addWidget(self.chkUse5mmResolution, 1, 1)

        self.debugModeCbox = QtWidgets.QCheckBox("Debug mode")
        generalLayout.addWidget(self.debugModeCbox, 4, 1, 1, 2)

        self.maxSizeImageSpinBox = QtWidgets.QSpinBox(self)
        self.maxSizeImageSpinBox.setMaximumWidth(150)
        self.maxSizeImageSpinBox.setMinimum(256)
        self.maxSizeImageSpinBox.setMaximum(2048)
        self.maxSizeImageSpinBox.setSingleStep(256)
        self.maxSizeImageSpinBox.setValue(1024)
        self.maxSizeImageLabel = QtWidgets.QLabel("Max size tiles:")
        generalLayout.addWidget(self.maxSizeImageLabel, 5, 0)
        generalLayout.addWidget(self.maxSizeImageSpinBox, 5, 1, 1, 2)

        self.groupBoxGeneral.setLayout(generalLayout)

        self.tabWidget = QtWidgets.QTabWidget()

        self.tabDataSetTraining = QtWidgets.QWidget()
        self.tabWidget.addTab(self.tabDataSetTraining, "Dataset")
        # Additional dataset training
        self.groupBoxDataSetTraining = QtWidgets.QGroupBox("Additional dataset training")
        trainingDataSetLayout = QtWidgets.QGridLayout()

        # Separation of annotations
        trainingDataSetLayout.addWidget(self.labelTrainZonesLayer, 0, 0)
        trainingDataSetLayout.addWidget(self.trainZonesLayer, 0, 1,1,2)
        trainingDataSetLayout.addWidget(self.labelTrainDataLayer, 1, 0)
        trainingDataSetLayout.addWidget(self.trainDataLayer, 1, 1, 1, 2)

        self.separationDataLable = QtWidgets.QLabel("Splitting data (train):")

        self.separationDataSpinBox = CustomDoubleSpinBox()
        self.separationDataSpinBox.setMaximumWidth(150)
        self.separationDataSpinBox.setRange(0.1, 1)
        self.separationDataSpinBox.setSingleStep(0.1)
        self.separationDataSpinBox.setDecimals(1)
        self.separationDataSpinBox.setValue(self.train_percentage)
        self.separationDataSpinBox.setLocale(locale)

        trainingDataSetLayout.addWidget(self.separationDataLable, 2, 0)
        trainingDataSetLayout.addWidget(self.separationDataSpinBox, 2, 1)

        self.labelProportionBackground = QtWidgets.QLabel("Proportion background:")

        self.proportionBackgroundSpinBox = CustomDoubleSpinBox()
        self.proportionBackgroundSpinBox.setMaximumWidth(150)
        self.proportionBackgroundSpinBox.setRange(0.1, 1)
        self.proportionBackgroundSpinBox.setSingleStep(0.1)
        self.proportionBackgroundSpinBox.setDecimals(1)
        self.proportionBackgroundSpinBox.setValue(self.percent_empty_limit)
        self.proportionBackgroundSpinBox.setLocale(locale)
        trainingDataSetLayout.addWidget(self.labelProportionBackground, 3, 0)
        trainingDataSetLayout.addWidget(self.proportionBackgroundSpinBox, 3, 1)


        self.labelAugmentedAnnotations = QtWidgets.QLabel("Augmented data (x8):")
        trainingDataSetLayout.addWidget(self.labelAugmentedAnnotations, 4, 0)

        self.augmentCbox = QtWidgets.QCheckBox("Off")
        trainingDataSetLayout.addWidget(self.augmentCbox, 4, 1)

        self.labelAugmentColorsCbox = QtWidgets.QLabel("Random augment colors:")
        trainingDataSetLayout.addWidget(self.labelAugmentColorsCbox, 5, 0)

        self.augmentColorsCbox = QtWidgets.QCheckBox("Off")
        trainingDataSetLayout.addWidget(self.augmentColorsCbox, 5, 1)

        self.labelMode = QtWidgets.QLabel("Mode:")
        self.cbxDetect = QtWidgets.QCheckBox("Detect (Boxes)")
        self.cbxDetect.setChecked(True)
        self.cbxSegmentation = QtWidgets.QCheckBox("Segmentation (Contours)")
        self.cbxSegmentation.setDisabled(False)

        self.buttonGroup = QtWidgets.QButtonGroup(self)
        self.buttonGroup.setExclusive(True)
        self.buttonGroup.addButton(self.cbxDetect, 0)
        self.buttonGroup.addButton(self.cbxSegmentation, 1)

        trainingDataSetLayout.addWidget(self.labelMode, 6, 0)
        trainingDataSetLayout.addWidget(self.cbxDetect, 6, 1)
        trainingDataSetLayout.addWidget(self.cbxSegmentation, 7, 1)

        self.btnRun = QtWidgets.QPushButton("Run")
        self.btnRun.setMaximumWidth(100)
        self.btnRun.setEnabled(True)

        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)
        self.btnStop.setMaximumWidth(100)

        layout = QtWidgets.QGridLayout()
        row = 0

        layout.addWidget(self.groupBoxGeneral, row, 0, 1, 3)
        row += 1

        self.tabDataSetTraining.setLayout(trainingDataSetLayout)
        layout.addWidget(self.tabWidget, row, 0, 1, 3)
        row += 1

        self.txtInfoPBar = QtWidgets.QLabel()
        self.txtInfoPBar.setText("")
        layout.addWidget(self.txtInfoPBar, row, 1, 1, 3)
        row += 1

        self.txtPBar = QtWidgets.QLabel()
        self.txtPBar.setText("Progress:")
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setTextVisible(True)
        layout.addWidget(self.txtPBar, row, 0)
        layout.addWidget(self.progressBar, row, 1, 1, 2)
        row += 1

        layout.addWidget(self.btnRun, row, 0)
        layout.addWidget(self.btnStop, row, 3)
        row += 1

        self.setLayout(layout)

        QtCore.QObject.connect(self.btnRun, QtCore.SIGNAL("clicked()"), lambda: self.run_process())
        QtCore.QObject.connect(self.btnStop, QtCore.SIGNAL("clicked()"), lambda: self.stop())

        self.buttonGroup.buttonClicked[int].connect(self.check_selection_mode)

        self.debugModeCbox.stateChanged.connect(self.change_debug_mode)
        self.augmentColorsCbox.stateChanged.connect(self.change_augment_colors)
        self.augmentCbox.stateChanged.connect(self.change_augment_data)

    def check_selection_mode(self, button_id):
        if button_id == 0:
            self.selected_mode = "detect"
        elif button_id == 1:
            self.selected_mode = "segmentation"

    def change_debug_mode(self, value):
        self.isDebugMode = value
        print(f"Debug mode: {'On' if value else 'Off'}")

    def change_augment_colors(self, value):
        self.augment_colors = value
        self.augmentColorsCbox.setText('On' if value else 'Off')

    def change_augment_data(self, value):
        self.isAugmentData = value
        self.augmentCbox.setText('On' if value else 'Off')

    def choose_working_dir(self):
        working_dir = Metashape.app.getExistingDirectory()
        self.workingDirLineEdit.setText(working_dir)

    def load_params(self):

        app = QtWidgets.QApplication.instance()

        self.prefer_original_resolution = not self.chkUse5mmResolution.isChecked()

        self.percent_empty_limit = self.proportionBackgroundSpinBox.value()
        self.max_image_size = self.maxSizeImageSpinBox.value()
        self.preferred_patch_size = self.max_image_size

        self.working_dir = self.workingDirLineEdit.text()
        self.train_percentage = self.separationDataSpinBox.value()

        Metashape.app.settings.setValue("scripts/create_yolo_dataset/percent_empty_limit",
                                        str(self.percent_empty_limit))
        Metashape.app.settings.setValue("scripts/create_yolo_dataset/max_image_size", str(self.max_image_size))
        Metashape.app.settings.setValue("scripts/create_yolo_dataset/train_percentage", str(self.train_percentage))

        if not self.prefer_original_resolution:
            self.orthomosaic_resolution = self.preferred_resolution
            self.patch_size = self.preferred_patch_size
        else:
            self.orthomosaic_resolution = self.chunk.orthomosaic.resolution
            if self.orthomosaic_resolution > 0.105:
                raise Exception("Orthomosaic should have resolution <= 10 cm/pix.")
            if self.force_small_patch_size:
                patch_size_multiplier = 1
            else:
                patch_size_multiplier = max(1, min(4, self.preferred_resolution / self.orthomosaic_resolution))
            self.patch_size = round(self.preferred_patch_size * patch_size_multiplier)

        self.patch_inner_border = self.patch_size // 8
        print("Using resolution {} m/pix with patch {}x{}".format(self.orthomosaic_resolution, self.patch_size,
                                                                  self.patch_size))


        trainZonesLayer = self.layers[self.trainZonesLayer.currentIndex()]
        trainDataLayer = self.layers[self.trainDataLayer.currentIndex()]

        if trainZonesLayer == self.noTrainDataChoice or trainDataLayer == self.noTrainDataChoice:
            self.train_on_user_data_enabled = False
            print("Additional dataset disabled")
        else:
            self.train_on_user_data_enabled = True
            print("Additional dataset expected on key={} layer data w.r.t. key={} layer zones".format(
                trainDataLayer[0], trainZonesLayer[0]))

        loading_train_shapes_start = time.time()
        self.shapes = self.chunk.shapes
        self.train_zones = []
        self.train_data = []

        print(f"All shapes chunk: {len(self.shapes)}")

        train_zones_key = trainZonesLayer[0]
        train_data_key = trainDataLayer[0]

        print("Grouping shapes by key: {} and {}".format(train_zones_key, train_data_key))
        for i, shape in enumerate(self.shapes):
            if shape.group.key == train_zones_key:
                self.train_zones.append(shape)
            elif shape.group.key == train_data_key:
                self.train_data.append(shape)

            i+=1
            self.progressBar.setValue(int((100 * i + 1) / len(self.shapes)))
            app.processEvents()
            self.check_stopped()



        print("{} zones and {} data loaded in {:.2f} sec".format(len(self.train_zones),
                                                                             len(self.train_data),
                                                                             time.time() - loading_train_shapes_start))

    def debug_draw_objects(self, img, bboxes, contours):
        """
        Draws bounding boxes and contours on an image for debugging visualization.
        
        Parameters:
            img: Input image to draw on
            bboxes (list): List of bounding boxes in format [xmin, ymin, xmax, ymax]
            contours (list): List of contours to draw
            
        Returns:
            numpy.ndarray: Image with drawn bounding boxes and contours
        """


        img = img.copy()
        h, w, cn = img.shape

        for tree in bboxes:
            # Convert to integers
            xmin, ymin, xmax, ymax = map(int, map(round, tree))

            # Check bbox values
            assert np.all(np.array([xmin, ymin]) >= np.int32([0, 0])), \
                f"Bounding box values out of bounds: {xmin}, {ymin}"
            assert np.all(np.array([xmax, ymax]) <= np.int32([w, h])), \
                f"Bounding box values out of bounds: {xmax}, {ymax} (image size: {w}, {h})"
            # if np.all(np.array([xmin, ymin]) >= np.int32([0, 0])):
            #     xmin = max(xmin, 0)
            #     ymin = max(ymin, 0)
            # if np.all(np.array([xmax, ymax]) <= np.int32([w, h])):
            #     xmax = min(xmax, w)
            #     ymax = min(ymax, h)

            # Convert to numpy array
            (xmin, ymin), (xmax, ymax) = np.array([xmin, ymin]), np.array([xmax, ymax])

            # Draw rectangle
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Draw contours
        for contour in contours:
            # corrected_contour = []
            # for (x, y) in contour:
            #     x = min(max(x, 0), w)
            #     y = min(max(y, 0), h)
            #     corrected_contour.append([x, y])
            contour = np.array(contour, dtype=np.int32).reshape(-1, 2)
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        return img

class CustomDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """
    Custom implementation of QDoubleSpinBox to modify text representation.

    This class provides a customized text representation for the numerical
    values of a QDoubleSpinBox. It removes trailing zeroes and unnecessary
    decimal points from the displayed value for cleaner visualization.

    Attributes
    ----------
    Inherited from QDoubleSpinBox.
    """
    def textFromValue(self, value):
        """
        Converts a numeric value to its string representation with trailing zeros removed.
        
        This method overrides the default QDoubleSpinBox text representation to provide
        cleaner formatting by removing unnecessary trailing zeros and decimal points.
        
        Parameters:
            value (float): The numeric value to convert to text.
            
        Returns:
            str: Formatted string representation without trailing zeros.
        """
        import re
        text = super(CustomDoubleSpinBox, self).textFromValue(value)
        return re.sub(r'0*$', '', re.sub(r'\.0*$', '', text))





