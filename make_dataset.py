import os
from crop_function import crop
import glob
from pathlib import Path


class_names = os.listdir("Violence")
for class_name in class_names:
    path_classes = os.path.join("Violence", class_name)
    output = os.path.join("Violence_output", class_name)

    file_paths = glob.glob(path_classes + "/*.avi")

    for file_path in file_paths:
        file_name = (Path(file_path)).stem
        print("Process video in " + file_path)
        crop(path_in=file_path, path_out=output + "/" + file_name + ".avi")

