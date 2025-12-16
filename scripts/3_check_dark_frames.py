import glob
import os
from collections import defaultdict
import utilities
import time
from typing import Dict, List

'''
    Step 3: This script checks to see if you have the correct set of dark frames.
    The script will tell you if you need more images and what kind and which images can be deleted.
'''

FILE_EXTENSION = "DNG"
DARK_FRAME_DIR_NAME = "ISO_3200/dark_frame"  # dark frames are per-ISO, so check each ISO
CAMERA_MODEL = "iPhone_11_Pro"
TEST_SHOTS_DIR = 'test_shots'
DATA_PATH = "../data"
OUTPUT_DIR = "results"
FULL_LOGGING = False
WARNINGS = True
DESIRED_IMAGE_COUNT = 10


class DarkFrameImage:
    """
    Loads a raw file and stores its exposure time, ISO, and path.
    """

    def __init__(self, path: str):
        self.path = path
        self.exposure_time = 0.0
        self.iso = 0
        self._analyze_image()

    def _analyze_image(self):
        tags = utilities.exif_tags(self.path)
        self.exposure_time = utilities.get_exposure_time(tags)
        self.iso = utilities.get_iso(tags)


# need to organize by ISO, then exposure time, with a count of images at that exposure time
# returns a dictionary of (ISO, dictionary of (exposure_time, image count))
def sort_images(images: List[DarkFrameImage]) -> Dict[int, Dict[float, List[DarkFrameImage]]]:
    iso_dict = defaultdict(lambda: defaultdict(list))
    for image in images:
        iso = int(image.iso)
        exposure = image.exposure_time
        iso_dict[iso][exposure].append(image)
    return iso_dict


def summarize_coverage(image_dictionary: Dict[int, Dict[float, List[DarkFrameImage]]]):
    has_complete_set = True
    iso_count = len(image_dictionary.keys())

    for iso, exposure_time_dict in image_dictionary.items():
        has_complete_set_for_this_iso = True
        for exposure_time, image_list in exposure_time_dict.items():
            count = len(image_list)
            if count < DESIRED_IMAGE_COUNT:
                print(f"ISO {iso}, exposure_time: {exposure_time}, missing {DESIRED_IMAGE_COUNT - count} image(s)")
                has_complete_set = False
                has_complete_set_for_this_iso = False
            elif count > DESIRED_IMAGE_COUNT:
                extra_image_list = image_list[DESIRED_IMAGE_COUNT:]
                print(f"ISO {iso}, exposure_time: {exposure_time}, EXTRA {count - DESIRED_IMAGE_COUNT} image(s)")
                has_complete_set = False
                has_complete_set_for_this_iso = False
                for extra_image in extra_image_list:
                    print(f"\tYou can delete '{extra_image.path}'")
        if has_complete_set_for_this_iso:
            print(f"complete set of {DESIRED_IMAGE_COUNT} images in {len(exposure_time_dict.keys())} exposure times")

    if has_complete_set:
        print(f"complete set of {DESIRED_IMAGE_COUNT} images in {iso_count} ISOs")


def run(rename: bool = False):
    start = time.time()
    path = os.path.join(os.path.join(DATA_PATH, CAMERA_MODEL), TEST_SHOTS_DIR)
    path = os.path.join(path, DARK_FRAME_DIR_NAME)
    paths = sorted(glob.glob(os.path.join(path, f'*.{FILE_EXTENSION}')))

    # load up all the images
    image_list = [DarkFrameImage(path) for path in paths]
    print(f"loaded {len(image_list)} raw images")

    image_dict = sort_images(image_list)
    summarize_coverage(image_dict)
    print(f"total time = {time.time() - start}")

    for iso, exposure_time_dict in sorted(image_dict.items()):
        for exposure_time, images in sorted(exposure_time_dict.items()):
            exp_str = f"{exposure_time:.0f}s" if exposure_time >= 1.0 else f"1_{1.0 / exposure_time:.0f}s"
            for idx, image in enumerate(images):
                if image.exposure_time != exposure_time:
                    raise ValueError(f"mismatching exposure time")

                print(f"ISO {iso}; time = {exp_str}; {image.path}")
                if rename:
                    new_file_name = f"ISO_{iso}-{exp_str}-dark_frame_{idx}.{FILE_EXTENSION}"
                    new_path = os.path.join(path, new_file_name)
                    os.rename(image.path, new_path)


if __name__ == '__main__':
    start = time.time()
    run(rename = False)
    print(f"run time = {time.time() - start}")
