import os
import glob
import time
from pathlib import Path
import utilities
import numpy as np
import json
from SensorCalibration import SensorCalibration, SensorCalibrationISO
from typing import Dict, Tuple
import rawpy
from collections import defaultdict
from parameters import CAMERA_MODEL, ISO_LIST_TO_INJECT, NO_AUTO_SCALE, \
    NO_AUTO_BRIGHT, USE_DARK_CURRENT_DATABASE, FILE_EXTENSION
import dng_parameters as dngp
import struct
import re

'''
    This script add noise using the noise model and dark current database.
'''

DATA_PATH = "../data"
OUTPUT_DIR = "results"
CALIBRATION_NAME = "calibration.json"
TEST_SHOTS_DIR = "test_shots"
VALIDATION_DIR = "validation"
CLEAN_IMAGE_DIR = "clean"
NOISY_IMAGE_DIR = "noisy"
IMAGE_FORMAT = "tiff"
INJECTED_DIR = "injected"
NOISE_MODEL_NAME = f"{CAMERA_MODEL}_noise_models.json"
RAW_DATA_FILE_EXTENSION = "rawdata"
RAW_METADATA_FILE_EXTENSION = "json"
BPS = 16         # bits per pixel (PNG and TIFF can be 8 or 16, JPEG is only 8)


def load_calibration() -> SensorCalibration:
    base_path = os.path.join(DATA_PATH, CAMERA_MODEL)
    calibration_path = os.path.join(os.path.join(os.path.join(base_path, OUTPUT_DIR)), CALIBRATION_NAME)

    with open(calibration_path, 'r') as f:
        calibration_json_dict = json.load(f)        # this is a dictionary with one key (the ISO)
        sensor_calibration = SensorCalibration.newWithJSON(calibration_json_dict)
        return sensor_calibration


# returns {channel: [CFA_data]} for this exposure time
# dark current buffers for a single channel
def load_dark_current_database(calibration: SensorCalibrationISO, exposure_time: float):
    available_exposures = list(calibration.dark_current_dict.keys())
    available_exposures_float = [float(exp) for exp in available_exposures]

    # deal with very short exposure times (which have little noise anyway)
    min_exposure_time = min(available_exposures_float)
    scale = 1.0
    if exposure_time < min_exposure_time:
        closest_exposure = min_exposure_time
        scale = exposure_time / closest_exposure
    else:
        differences = [abs(exposure_time - exp) for exp in available_exposures_float]
        closest_idx = np.argmin(differences)
        closest_exposure = float(available_exposures[closest_idx])
        tolerance = 0.5
        relative_error = abs(closest_exposure - exposure_time) / exposure_time

        if relative_error > tolerance:
            raise(ValueError, f"Warning: Closest exposure {closest_exposure}s is {relative_error * 100:.1f}% different from ideal {exposure_time}s")
        elif closest_exposure != exposure_time:
            # print(f"...no exact exposure time found. Looking for {exposure_time} and will match to {closest_exposure}")
            scale = exposure_time / closest_exposure

    paths = calibration.dark_current_dict[str(closest_exposure)]
    result = defaultdict(list)
    for path in paths:
        with rawpy.imread(path) as raw:
            raw_data = raw.raw_image.astype(np.float32)
            pattern_str = utilities.pattern_to_string(raw.raw_pattern)
            channels = utilities.split_cfa_channels(raw_data,
                                                    pattern_str,
                                                    raw.black_level_per_channel,
                                                    raw.white_level)

            tags = utilities.exif_tags(path)
            file_exposure_time = utilities.get_exposure_time(tags)
            file_iso = utilities.get_iso(tags)

            # iPhones report one ISO, but store a different on at the extremes
            assert(calibration.iso == file_iso)

            assert(closest_exposure == file_exposure_time)
            for ch, cfa_data in channels.items():
                result[ch].append(cfa_data * scale)
    return result


def _sample_per_pixel_vectorized(
        dark_frames: np.ndarray,
        output_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Efficient per-pixel sampling using advanced NumPy indexing.
    """
    frame_count = len(dark_frames)
    frame_h, frame_w = dark_frames[0].shape
    out_h, out_w = output_shape

    # Verify output fits within dark frame dimensions
    if out_h > frame_h or out_w > frame_w:
        raise ValueError(f"Output shape {output_shape} exceeds dark frame shape ({frame_h}, {frame_w})")

    # For each pixel, randomly choose which frame to sample from
    # Shape: (out_h, out_w) with values in [0, num_frames)
    frame_indices = np.random.randint(0, frame_count, size=(out_h, out_w))

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:out_h, 0:out_w]

    # Advanced indexing: for each (y, x), get dark_frames[frame_indices[y,x], y, x]
    dark_frames = np.array(dark_frames)  # needs to be np.array for the next line.
    dark_sampled = dark_frames[frame_indices, y_coords, x_coords]

    return dark_sampled


def _sample_patches(
        dark_frames: np.ndarray,
        output_shape: Tuple[int, int],
        patch_size: int
) -> np.ndarray:
    """
    Sample patches from random dark frames at random locations.

    Fills output by tiling patches sampled from the dark frame database.
    Preserves spatial correlation within each patch.
    """
    frame_count = len(dark_frames)
    frame_h, frame_w = dark_frames[0].shape
    out_h, out_w = output_shape

    # Initialize output
    dark_sampled = np.zeros(output_shape, dtype=dark_frames[0].dtype)

    y = 0
    while y < out_h:
        x = 0
        while x < out_w:
            # Randomly select a frame
            frame_idx = np.random.randint(0, frame_count)
            dark_frame = dark_frames[frame_idx]

            # Randomly select patch location in the frame
            max_y = frame_h - patch_size
            max_x = frame_w - patch_size

            if max_y > 0 and max_x > 0:
                patch_y = np.random.randint(0, max_y)
                patch_x = np.random.randint(0, max_x)
            else:
                # If patch is larger than frame, just use (0, 0)
                patch_y = 0
                patch_x = 0

            # Extract patch (handle edges)
            patch_h = min(patch_size, frame_h - patch_y, out_h - y)
            patch_w = min(patch_size, frame_w - patch_x, out_w - x)

            patch = dark_frame[patch_y:patch_y + patch_h, patch_x:patch_x + patch_w]

            # Place in output
            dark_sampled[y:y + patch_h, x:x + patch_w] = patch

            x += patch_size
        y += patch_size

    return dark_sampled


def sample_from_dark_current_database(dark_current_database: dict, apply_crop: bool,
                                      channel: str,
                                      output_shape: Tuple[int, int], sample_mode: str):
    dark_frame_cfa_list = dark_current_database[channel]
    patch_size = 256

    if sample_mode == 'per_pixel':
        return _sample_per_pixel_vectorized(dark_frame_cfa_list, output_shape)
    elif sample_mode == 'per-patch':
        return _sample_patches(dark_frame_cfa_list, output_shape, patch_size)
    else:
        raise ValueError(f"Unknown mode: {sample_mode}")


def synthesize_noise(clean_channels: Dict[str, np.ndarray],
                     target_iso: int,
                     target_exposure_time: float,
                     calibration: SensorCalibration,
                     loaded_noise_model: str,
                     use_dark_current_database: bool):

    noisy_channels = dict()
    calib_target = calibration.iso_calibration_dict[str(target_iso)]
    target_iso_str = str(target_iso)

    if target_iso_str not in loaded_noise_model:
        raise ValueError(f"No noise model available for ISO {target_iso}")

    # loads the dark current images for this ISO and exposure time
    if use_dark_current_database:
        dark_current_database = load_dark_current_database(calib_target, target_exposure_time)

    for ch in ['R', 'G1', 'G2', 'B']:
        a = loaded_noise_model[target_iso_str][ch]['a']
        b = loaded_noise_model[target_iso_str][ch]['b']
        brightness_scale = loaded_noise_model[target_iso_str][ch].get("brightness_scale")
        if brightness_scale is None:
            brightness_scale = 1.0

        clean = clean_channels[ch] * brightness_scale
        noisy = utilities.add_noise_to_channel_poisson_gaussian(clean, a, b)

        if use_dark_current_database:
            dark_current_sample_mode = 'per_pixel'      # 'per-patch' or 'per-pixel'
            dark_current = sample_from_dark_current_database(dark_current_database, False,
                                                             channel=ch,
                                                             output_shape=clean.shape,
                                                             sample_mode=dark_current_sample_mode)
            noisy = noisy + dark_current

        noisy_channels[ch] = noisy
    return noisy_channels


# returns rgb data
def load_and_inject(clean_path: str, sensor_calibration: SensorCalibration, output_iso: int,
                    bps: int, loaded_noise_model: dict, use_dark_current_database: bool,
                    no_auto_scale: bool, no_auto_bright: bool,
                    raw_data_output_path: str = None,
                    raw_metadata_output_path: str = None):
    with rawpy.imread(clean_path) as raw:
        pattern_str = utilities.pattern_to_string(raw.raw_pattern)
        raw_data = raw.raw_image.astype(np.float32)
        channels = utilities.split_cfa_channels(raw_data,
                                                pattern_str,
                                                raw.black_level_per_channel,
                                                raw.white_level)

        tags = utilities.exif_tags(clean_path)
        clean_exposure_time = utilities.get_exposure_time(tags)
        clean_iso = utilities.get_iso(tags)

        # trying to match the exposure time of a real noisy image, so use that exposure time
        # (for an image with similar brightness - note the aperture value should be the same)
        target_exposure_time = clean_exposure_time * (clean_iso / output_iso)

        noisy_cfa = synthesize_noise(clean_channels = channels,
                                     target_iso = output_iso,
                                     target_exposure_time = target_exposure_time,
                                     calibration = sensor_calibration,
                                     loaded_noise_model = loaded_noise_model,
                                     use_dark_current_database=use_dark_current_database)

        noisy_raw = utilities.merge_cfa_channels(noisy_cfa,
                                                 pattern_str,
                                                 raw.black_level_per_channel,
                                                 raw.white_level)

        # Inject into rawpy object
        raw.raw_image[:] = noisy_raw

        # have to do this before leaving the 'with' statement. Otherwise the raw data is invalid
        if raw_data_output_path is not None:
            noise_profile = loaded_noise_model[str(output_iso)]
            do_write_raw_data_files(raw, noisy_raw,
                                    raw_data_output_path, raw_metadata_output_path,
                                    tags, output_iso, target_exposure_time, noise_profile)

        # Demosaic and postprocess. Other parameters for conversion if you want linear images
        # gamma = (1, 1),  # disables gamma correction
        # output_color = rawpy.ColorSpace.raw  # keeps raw color space
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=no_auto_bright,
            no_auto_scale=no_auto_scale,
            output_bps=bps
        )
    return rgb


def do_write_raw_data_files(raw_object, raw_data,
                            raw_data_output_path, raw_metadata_output_path,
                            exif_tags, output_iso, output_exposure_time,
                            noise_profile):
    # write binary data to disk
    arr_uint16 = raw_data.astype(np.uint16)
    with open(raw_data_output_path, "wb") as f:
        f.write(struct.pack("II", raw_data.shape[0], raw_data.shape[1]))
        f.write(arr_uint16.tobytes())

    # write json for metadata
    white_balance = raw_object.camera_whitebalance
    black_level_per_channel_tuple = raw_object.black_level_per_channel,
    white_level = raw_object.white_level
    wb_base_value = white_balance[1]        # green
    white_balance = white_balance[:3]       # only want RGB, not extra values
    as_shot_neutral = [wb_base_value / v for v in white_balance]
    aperture_value = utilities.get_aperture(exif_tags)
    focal_length = utilities.get_focal_length(exif_tags)

    # noise profile - write (a, b) per channel
    noise_profile_list = list()
    # must be produced in order as RGGB
    for ch, noise_dict in noise_profile.items():
        a = noise_dict['a']
        b = noise_dict['b']
        noise_profile_list.append(a)
        noise_profile_list.append(b)

    black_level_per_channel = [x for x in black_level_per_channel_tuple]

    json_dict = {"iso": output_iso,
                 "shutter_speed": output_exposure_time,
                 "as_shot": as_shot_neutral,
                 "aperture": aperture_value,
                 "focal_length": focal_length,
                 "white_level": white_level,
                 "black_level_per_channel": black_level_per_channel[0],
                 "noise_profile": noise_profile_list,
                 "bayer_pattern": dngp.DNG_BAYER_PATTERN,
                 "camera_make": dngp.DNG_CAMERA_MAKE,
                 "camera_model": dngp.DNG_CAMERA_MODEL_NAME,
                 "illuminant1": dngp.DNG_ILLUMINANT1,
                 "color_matrix1": dngp.DNG_COLOR_MATRIX1,
                 "illuminant2": dngp.DNG_ILLUMINANT2,
                 "color_matrix2": dngp.DNG_COLOR_MATRIX2,
                 "active_area": dngp.DNG_ACTIVE_AREA,
                 "crop_origin": dngp.DNG_CROP_ORIGIN,
                 "crop_size": dngp.DNG_CROP_SIZE,
                 "bounds": dngp.DNG_BOUNDS,
                 "baseline_exposure": dngp.DNG_BASELINE_EXPOSURE,
    }
    if dngp.DNG_FORWARD_MATRIX1 is not None:
        json_dict["forward_matrix1"] = dngp.DNG_FORWARD_MATRIX1

    if dngp.DNG_FORWARD_MATRIX2 is not None:
        json_dict["forward_matrix2"] = dngp.DNG_FORWARD_MATRIX2

    with open(raw_metadata_output_path, "w") as f2:
        json.dump(json_dict, f2, indent=2)


def run(write_real_noisy: bool, use_dark_current_database: bool,
        no_auto_bright: bool, no_auto_scale: bool,
        single_directory: str = None, use_noise_model: bool = False,
        write_raw_data_files: bool = False,
        write_clean_raw: bool = False,
        use_dataset_naming_policy: bool = False,
        override_input_directory: str = None):
    sensor_calibration = load_calibration()

    base_path = os.path.join(DATA_PATH, CAMERA_MODEL)
    models_by_iso = None

    if use_noise_model:
        noise_model_folder = os.path.join(base_path, OUTPUT_DIR)
        noise_model_path = os.path.join(os.path.join(noise_model_folder, "P-G models"), NOISE_MODEL_NAME)
        print(f"USING NOISE MODEL AT {noise_model_path}")
        with open(noise_model_path, 'r') as f:
            models_by_iso = json.load(f)

    input_directory = os.path.join(os.path.join(os.path.join(base_path, TEST_SHOTS_DIR), VALIDATION_DIR))
    if override_input_directory:
        input_directory = override_input_directory

    if single_directory is None:
        input_directory_path = Path(input_directory)
        directories_to_process = sorted([p for p in input_directory_path.rglob("*") if p.is_dir()])
    else:
        directories_to_process = [os.path.join(os.path.join(input_directory, single_directory))]

    injection_directory = os.path.join(os.path.join(os.path.join(base_path, OUTPUT_DIR), VALIDATION_DIR), INJECTED_DIR)
    if override_input_directory:
        injection_directory = override_input_directory

    for idx, directory in enumerate(directories_to_process):
        # find the clean image
        clean_input_directory = directory
        if use_dataset_naming_policy is False:
            clean_input_directory = os.path.join(directory, CLEAN_IMAGE_DIR)

        clean_paths = sorted(glob.glob(os.path.join(clean_input_directory, f"*.{FILE_EXTENSION}")))
        if len(clean_paths) == 0:
            continue
        print(f"Processing {directory}; {len(clean_paths)} source files; {idx+1} of {len(directories_to_process)}")
        dir_start = time.time()
        for clean_path in clean_paths:
            stem = Path(clean_path).stem
            ext = Path(clean_path).suffix

            if use_dataset_naming_policy:
                orig_file_name = f"{stem}.{IMAGE_FORMAT}"
                output_directory = clean_input_directory
            else:
                orig_file_name = f"Clean.{IMAGE_FORMAT}"
                output_directory = os.path.join(injection_directory, stem)
                os.makedirs(output_directory, exist_ok=True)

            orig_output_path = os.path.join(output_directory, orig_file_name)
            utilities.write_clean_image(clean_path, orig_output_path, BPS, no_auto_bright=no_auto_bright, no_auto_scale=no_auto_scale)

            tags = utilities.exif_tags(clean_path)
            clean_aperture = utilities.get_aperture(tags)
            clean_exposure_time = utilities.get_exposure_time(tags)
            clean_iso = utilities.get_iso(tags)

            if write_raw_data_files and write_clean_raw:
                if use_dataset_naming_policy:
                    raw_data_file_name = f"{stem}.{RAW_DATA_FILE_EXTENSION}"
                    raw_metadata_file_name = f"{stem}.{RAW_METADATA_FILE_EXTENSION}"
                else:
                    raw_data_file_name = f"Clean.{RAW_DATA_FILE_EXTENSION}"
                    raw_metadata_file_name = f"Clean.{RAW_METADATA_FILE_EXTENSION}"

                raw_data_output_path = os.path.join(output_directory, raw_data_file_name)
                raw_metadata_output_path = os.path.join(output_directory, raw_metadata_file_name)
                with rawpy.imread(clean_path) as raw:
                    raw_data = raw.raw_image.astype(np.float32)
                    noise_profile = models_by_iso[str(clean_iso)]
                    do_write_raw_data_files(raw, raw_data,
                                            raw_data_output_path, raw_metadata_output_path,
                                            tags, clean_iso, clean_exposure_time, noise_profile)

            if models_by_iso is None:
                noisy_tag_name = "NB_H_Z"
            elif use_dark_current_database:
                noisy_tag_name = "NB_Z"
            else:   # just calibrated
                noisy_tag_name = "NB"

            parts = stem.rsplit("clean", 1)  # split from the right, max 1 split
            dataset_noisy_name_base = "noisy".join(parts)

            for iso in ISO_LIST_TO_INJECT:
                # print(f"generating {noisy_tag_name} noisy image based on '{Path(clean_path).stem}' at iso = {iso}")
                raw_data_output_path: str = None
                raw_metadata_output_path: str = None
                iso_str = f"{iso:05d}"  # "00200"
                dataset_noisy_name_iso = re.sub(r"ISO\d{5}", f"ISO{iso_str}", dataset_noisy_name_base)

                if write_raw_data_files:
                    if use_dataset_naming_policy:
                        raw_data_file_name = f"{dataset_noisy_name_iso}.{RAW_DATA_FILE_EXTENSION}"
                        raw_metadata_file_name = f"{dataset_noisy_name_iso}.{RAW_METADATA_FILE_EXTENSION}"
                    else:
                        raw_data_file_name = f"{noisy_tag_name}_ISO_{iso}.{RAW_DATA_FILE_EXTENSION}"
                        raw_metadata_file_name = f"{noisy_tag_name}_ISO_{iso}.{RAW_METADATA_FILE_EXTENSION}"
                    raw_data_output_path = os.path.join(output_directory, raw_data_file_name)
                    raw_metadata_output_path = os.path.join(output_directory, raw_metadata_file_name)

                noisy_rgb = load_and_inject(clean_path, sensor_calibration, iso, bps=BPS,
                                            loaded_noise_model = models_by_iso,
                                            use_dark_current_database=use_dark_current_database,
                                            no_auto_bright=no_auto_bright, no_auto_scale=no_auto_scale,
                                            raw_data_output_path=raw_data_output_path,
                                            raw_metadata_output_path=raw_metadata_output_path)

                if use_dataset_naming_policy:
                    noisy_file_name = f"{dataset_noisy_name_iso}.{IMAGE_FORMAT}"
                else:
                    noisy_file_name = f"{noisy_tag_name}_ISO_{iso}.{IMAGE_FORMAT}"

                output_path = os.path.join(output_directory, noisy_file_name)
                utilities.write_rgb(noisy_rgb, output_path, BPS)

                if write_real_noisy:
                    # decode and write out decoded real noisy image(s) to compare
                    noisy_input_directory = os.path.join(directory, NOISY_IMAGE_DIR)
                    for idx in range(2):
                        noisy_file_name = f"{stem}_noisy_ISO_{iso}_{idx+1}{ext}"
                        noisy_path = os.path.join(noisy_input_directory, noisy_file_name)
                        noisy_output_path = os.path.join(output_directory, f"Real_{idx+1}_ISO_{iso}.{IMAGE_FORMAT}")
                        noisy_tags = utilities.exif_tags(noisy_path)
                        if noisy_tags is not None:
                            noisy_aperture = utilities.get_aperture(noisy_tags)
                            target_exposure_time = clean_exposure_time * (clean_iso / iso)
                            noisy_exposure_time = utilities.get_exposure_time(noisy_tags)
                            if noisy_aperture != clean_aperture:
                                print(f"...WARNING for ISO = {iso}, Aperture mismatch: Clean = {clean_aperture}, Noisy = {noisy_aperture}")
                            if abs(noisy_aperture - clean_aperture) >= 1.0:        # this must be maintained to avoid issues with lens distortion / shading
                                raise ValueError(f"noisy_aperture - clean_aperture > 1.0: {abs(noisy_aperture - clean_aperture)}")

                            # if abs(noisy_exposure_time - target_exposure_time) >= 0.01:
                            #     raise ValueError(f"abs(noisy_exposure_time - target_exposure_time) > 0.01: {abs(noisy_exposure_time - target_exposure_time)}")
                            utilities.write_clean_image(noisy_path, noisy_output_path, BPS, no_auto_bright=no_auto_bright, no_auto_scale=no_auto_scale)
        print(f"Processing took {time.time() - dir_start}")

def go(write_jpg: bool = False,
       use_dark_current_database = True,
       write_raw_data_files: bool = False,
       write_real_noisy: bool = True,
       use_dataset_naming_policy: bool = False,
       override_input_directory: str = None):
    print(f"Injecting Calibrated Noise for {CAMERA_MODEL}")
    start = time.time()

    if write_jpg:
        global BPS
        global IMAGE_FORMAT
        BPS = 8
        IMAGE_FORMAT = "jpg"

    did_write_real_noisy = not write_real_noisy

    # P-G + Dark Current
    if use_dark_current_database:
        print("Poisson-Gaussian (Calibrated) + Dark Current")
        run(write_real_noisy = not did_write_real_noisy,            # make TIFFs from real noisy image
            use_dark_current_database=True,     # inject dark current (from calibration.json)
            no_auto_bright=NO_AUTO_BRIGHT,
            no_auto_scale=NO_AUTO_SCALE,
            single_directory=None,
            use_noise_model=True,
            write_raw_data_files=write_raw_data_files,
            use_dataset_naming_policy=use_dataset_naming_policy,
            override_input_directory=override_input_directory)              # use P-G noise model (if False, then uses calibration.json)
        did_write_real_noisy = True

    elif use_dataset_naming_policy is False:
        print("Poisson-Gaussian (Calibrated) only")
        run(write_real_noisy = not did_write_real_noisy,            # make TIFFs from real noisy image
            use_dark_current_database=False,     # inject dark current (from calibration.json)
            no_auto_bright=NO_AUTO_BRIGHT,
            no_auto_scale=NO_AUTO_SCALE,
            single_directory=None,
            use_noise_model=True,
            write_raw_data_files=write_raw_data_files,
            use_dataset_naming_policy=use_dataset_naming_policy,
            override_input_directory=override_input_directory)              # use P-G noise model (if False, then uses calibration.json)

    print(f"noise injection run time = {time.time() - start}")


if __name__ == "__main__":
    go(write_jpg = False,
       use_dark_current_database=USE_DARK_CURRENT_DATABASE,
       write_raw_data_files=True,
       write_real_noisy=True,
       use_dataset_naming_policy=False,
       override_input_directory=None)
