import os
import glob
from pathlib import Path
import utilities
import json
import time
from parameters import CAMERA_MODEL, ISO_LIST_TO_INJECT, BIT_DEPTH, NO_AUTO_SCALE, NO_AUTO_BRIGHT

DATA_PATH = "../data"
OUTPUT_DIR = "results"
NOISE_MODEL_DIR = f"P-G Models"
NOISE_MODEL_NAME = f"{CAMERA_MODEL}_noise_models (DNG).json"
TEST_SHOTS_DIR = "test_shots"
VALIDATION_DIR = "validation"
INJECTED_DIR = "injected"
CLEAN_IMAGE_DIR = "clean"
NOISY_IMAGE_DIR = "noisy"
IMAGE_FORMAT = "tiff"
BPS = 16         # bits per pixel (PNG can be 8 or 16, JPEG is only 8)

'''
    This script add noise using a DNG noise model, which you can make by pulling data from DNG files.
'''

def run(single_directory: str,
        no_auto_bright: bool, no_auto_scale: bool):
    base_path = os.path.join(DATA_PATH, CAMERA_MODEL)
    output_path = os.path.join(base_path, OUTPUT_DIR)

    model_path = os.path.join(os.path.join(output_path, NOISE_MODEL_DIR), NOISE_MODEL_NAME)
    with open(model_path, 'r') as f:
        models_by_iso = json.load(f)

    input_directory = os.path.join(os.path.join(os.path.join(base_path, TEST_SHOTS_DIR), VALIDATION_DIR))
    if single_directory is None:
        directories_to_process = sorted(glob.glob(os.path.join(input_directory, "*")))
    else:
        directories_to_process = [os.path.join(os.path.join(input_directory, single_directory))]

    injection_directory = os.path.join(os.path.join(os.path.join(base_path, OUTPUT_DIR), VALIDATION_DIR), INJECTED_DIR)

    for directory in directories_to_process:
        clean_input_directory = os.path.join(directory, CLEAN_IMAGE_DIR)
        clean_path = glob.glob(os.path.join(clean_input_directory, "*"))[0]
        stem = Path(clean_path).stem
        output_directory = os.path.join(injection_directory, stem)
        os.makedirs(output_directory, exist_ok=True)
        print(f"Generating uncalibrated noisy images based on '{stem}'")

        for iso in ISO_LIST_TO_INJECT:
            approaches = ["P-G"]
            for approach in approaches:
                noisy_file_name = f"{approach}_ISO_{iso}.{IMAGE_FORMAT}"
                output_path = os.path.join(output_directory, noisy_file_name)
                print(f"Generating noisy image '{Path(output_path).stem}' at iso = {iso}")
                utilities.write_synth_noisy_image(clean_path, models_by_iso, iso, output_path,
                                                  BPS, approach, BIT_DEPTH, no_auto_scale = no_auto_scale, no_auto_bright=no_auto_bright)


def go():
    print(f"Injecting uncalibrated Noise for {CAMERA_MODEL}")
    start = time.time()
    run(single_directory=None, no_auto_bright=NO_AUTO_BRIGHT, no_auto_scale=NO_AUTO_SCALE)
    print(f"Noise injection run time = {time.time() - start}")


if __name__ == "__main__":
    go()