import time
from typing import List, Tuple, Dict
import utilities
import os
import json
import glob
from collections import defaultdict
import SensorCalibration
from parameters import CAMERA_MODEL, FILE_EXTENSION, ISO_LIST

'''
    Step 4: This builds a calibration JSON file - mostly stores dark current file paths.
'''

DESIRED_DARK_CURRENT_EXPOSURE_TIME = 1.0
DATA_PATH = "../data"
TEST_SHOTS_DIR = "test_shots"
OUTPUT_DIR = "results"
DARK_FRAME_DIR = "dark_frame"


def iso_exposure_time(path: str) -> Tuple[int, float]:
    tags = utilities.exif_tags(path)
    exposure_time = utilities.get_exposure_time(tags)
    iso = utilities.get_iso(tags)
    return iso, exposure_time


def find_dark_current_files(calibration: SensorCalibration, iso: int):
    base_path = os.path.join(os.path.join(DATA_PATH, CAMERA_MODEL), TEST_SHOTS_DIR)
    dark_frame_directory = os.path.join(os.path.join(base_path, f"ISO_{iso}"), DARK_FRAME_DIR)
    paths = sorted(glob.glob(os.path.join(dark_frame_directory, f'*.{FILE_EXTENSION}')))
    if len(paths) == 0:
        raise ValueError(f"no paths found for ISO = {iso}")

    # load the paths and sort by exposure time
    result = defaultdict(list)
    for path in paths:
        (file_iso, file_exposure_time) = iso_exposure_time(path)
        if file_iso != iso:
            raise ValueError(f"ISO does not match. Found {file_iso}; wanted {iso}")
        result[file_exposure_time].append(path)

    calibration.dark_current_dict = result


def run():
    iso_calibration_dict: Dict[int, SensorCalibration.SensorCalibrationISO] = {}
    output_calibration = SensorCalibration.SensorCalibration()

    for iso in ISO_LIST:
        print(f"Calibrating ISO = {iso}")
        calibration = SensorCalibration.SensorCalibrationISO()
        calibration.iso = iso
        find_dark_current_files(calibration, iso)

        # write it out
        iso_calibration_dict[iso] = calibration

    output_calibration.iso_calibration_dict = iso_calibration_dict
    output_json_dict = output_calibration.json_dictionary()

    output_directory = os.path.join(os.path.join(os.path.join(DATA_PATH, CAMERA_MODEL), OUTPUT_DIR))
    output_file = os.path.join(output_directory, "calibration.json")
    with open(output_file, 'w') as f:
        json.dump(output_json_dict, f, indent=2)


if __name__ == '__main__':
    start = time.time()
    run()
    print(f"Calibration run time = {time.time() - start}")

