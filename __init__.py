import os

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