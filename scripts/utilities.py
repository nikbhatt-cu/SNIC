import rawpy
import numpy as np
import json
import os
import imageio
from PIL import Image
from PIL import ImageCms
from typing import Dict, Any, Tuple
import tifffile
DATA_PATH = "../data"
CAMERA_MODEL = "Sony_A7R_III"
OUTPUT_DIR = "results"
NOISE_MODEL_NAME = f"{CAMERA_MODEL}_noise_models.json"


def load_noise_model():
    # Load all ISO-specific noise models
    base_path = os.path.join(DATA_PATH, CAMERA_MODEL)

    output_path = os.path.join(os.path.join(base_path, OUTPUT_DIR))
    model_path = os.path.join(output_path, NOISE_MODEL_NAME)
    with open(model_path, 'r') as f:
        return json.load(f)


def pattern_to_string(pattern):
    color_map = {0: 'R', 1: 'G', 2: 'B', 3: 'G'}
    return ''.join(color_map[pattern[i, j]] for i in range(2) for j in range(2))


def adjust_pattern(pattern, row_offset, col_offset):
    shifted = pattern.copy()
    if row_offset % 2 == 1:
        shifted = shifted[[1, 0], :]
    if col_offset % 2 == 1:
        shifted = shifted[:, [1, 0]]
    return shifted


def split_cfa_channels(raw_array, pattern, black_levels, white_level):
    # Ensure even dimensions
    h, w = raw_array.shape
    raw_array = raw_array[:h - h % 2, :w - w % 2]

    def normalize_channel(data, color_char):
        """Normalize a channel with its specific black level."""
        # Map color to black level index
        color_to_idx = {'R': 0, 'G': 1, 'B': 2}
        black_idx = color_to_idx[color_char]
        black = black_levels[black_idx]

        normalized = (data - black) / (white_level - black)
        return np.clip(normalized, 0, 1)

    # Map pattern to channels
    if pattern == 'RGGB':
        R  = raw_array[0::2, 1::2]
        G1 = raw_array[0::2, 0::2]
        G2 = raw_array[1::2, 1::2]
        B  = raw_array[1::2, 0::2]
    elif pattern == 'BGGR':
        B  = raw_array[0::2, 0::2]
        G1 = raw_array[0::2, 1::2]
        G2 = raw_array[1::2, 0::2]
        R  = raw_array[1::2, 1::2]
    elif pattern == 'GRBG':
        G1 = raw_array[0::2, 0::2]
        R  = raw_array[0::2, 1::2]
        B  = raw_array[1::2, 0::2]
        G2 = raw_array[1::2, 1::2]
    elif pattern == 'GBRG':
        G1 = raw_array[0::2, 0::2]
        B  = raw_array[0::2, 1::2]
        R  = raw_array[1::2, 0::2]
        G2 = raw_array[1::2, 1::2]
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")

    result = {'R': normalize_channel(R, 'R'),
              'G1': normalize_channel(G1, 'G'),
              'G2': normalize_channel(G2, 'G'),
              'B': normalize_channel(B, 'B')}

    return result


def merge_cfa_channels(channels, pattern, black_levels, white_level):
    h, w = channels['R'].shape[0] * 2, channels['R'].shape[1] * 2
    merged = np.zeros((h, w), dtype=np.uint16)  # Use uint16 for raw DN values

    def denormalize_channel(normalized_data, color_char):
        """Denormalize a channel: DN = normalized × (white - black) + black"""
        color_to_idx = {'R': 0, 'G': 1, 'B': 2}
        black_idx = color_to_idx[color_char]
        black = black_levels[black_idx]

        # Denormalize
        dn_values = normalized_data * (white_level - black) + black

        # Clip to valid range and convert to uint16
        dn_values = np.clip(dn_values, 0, white_level).astype(np.uint16)

        return dn_values

    # Denormalize each channel
    R_dn = denormalize_channel(channels['R'], 'R')
    G1_dn = denormalize_channel(channels['G1'], 'G')
    G2_dn = denormalize_channel(channels['G2'], 'G')
    B_dn = denormalize_channel(channels['B'], 'B')

    # Map pattern to positions
    if pattern == 'RGGB':
        merged[0::2, 1::2] = R_dn
        merged[0::2, 0::2] = G1_dn
        merged[1::2, 1::2] = G2_dn
        merged[1::2, 0::2] = B_dn
    elif pattern == 'BGGR':
        merged[0::2, 0::2] = B_dn
        merged[0::2, 1::2] = G1_dn
        merged[1::2, 0::2] = G2_dn
        merged[1::2, 1::2] = R_dn
    elif pattern == 'GRBG':
        merged[0::2, 0::2] = G1_dn
        merged[0::2, 1::2] = R_dn
        merged[1::2, 0::2] = B_dn
        merged[1::2, 1::2] = G2_dn
    elif pattern == 'GBRG':
        merged[0::2, 0::2] = G1_dn
        merged[0::2, 1::2] = B_dn
        merged[1::2, 0::2] = R_dn
        merged[1::2, 1::2] = G2_dn
    else:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")

    return merged


def load_image_channels(path: str) -> Tuple[dict, int, float]:
    with rawpy.imread(path) as raw:
        pattern_str = pattern_to_string(raw.raw_pattern)

        # Split into channels and normalize with per-channel black levels
        channels = split_cfa_channels(
            raw,
            pattern_str,
            raw.black_level_per_channel,
            raw.white_level
        )
        tags = exif_tags(path)
        exposure_time = get_exposure_time(tags)
        iso = get_iso(tags)
        return channels, iso, exposure_time


def add_noise_to_channel_gaussian(channel, a, b, bit_depth):
    sigma = np.sqrt(a * channel + b)
    noise = np.random.randn(*channel.shape) * sigma
    return np.clip(channel + noise, 0, 1)


def add_awgn_noise_to_channel(channel, iso, base_sigma, base_iso = 100.0):
    """
    This method does a slightly less naive AWGN model scales sigma based on ISO
    """
    sigma = base_sigma * np.sqrt(iso / base_iso)
    noise = np.random.normal(loc=0, scale=sigma, size=channel.shape)
    noisy = channel + noise
    return np.clip(noisy, 0, 1)


def add_awgn_noise_to_channel_weighted(channel, iso, base_sigma, signal_weight = 0.01):
    """
    This method does a slightly less naive AWGN model scales sigma based on ISO
    """
    sigma = base_sigma * np.sqrt(iso / 100.0)
    sigma_total = np.sqrt((signal_weight * channel) ** 2 + sigma ** 2)
    noise = np.random.normal(loc=0, scale=sigma_total, size=channel.shape)
    noisy = channel + noise
    return np.clip(noisy, 0, 1)


def add_noise_to_channel_poisson_gaussian(channel, a, b):
    # Scale to simulate photon counts
    photons_per_DN = 1.0 / a
    photons = channel * photons_per_DN
    noisy_photons = np.random.poisson(np.maximum(photons, 0))

    # Convert back to DN
    shot_noise = noisy_photons / photons_per_DN

    # Add read noise
    read_noise = np.random.normal(0, np.sqrt(b), channel.shape)

    noisy = shot_noise + read_noise
    return np.clip(noisy, 0, 1)


# injects noise at a specific ISO and returns the RAW object with the noise inserted
def inject_noise(raw, models_by_iso, iso_noise_to_inject, approach: str, bit_depth: int):
    iso = str(iso_noise_to_inject)
    pattern_str = pattern_to_string(raw.raw_pattern)
    raw_data = raw.raw_image.astype(np.float32)
    channels = split_cfa_channels(raw_data,
                                  pattern_str,
                                  raw.black_level_per_channel,
                                  raw.white_level)

    if iso not in models_by_iso:
        raise ValueError(f"No noise model available for ISO {iso}")

    noisy_channels = {}

    for ch in channels:
        if approach == 'G':
            a = models_by_iso[iso][ch]['a']
            b = models_by_iso[iso][ch]['b']
            noisy_channels[ch] = add_noise_to_channel_gaussian(channels[ch], a, b, bit_depth)
        elif approach == 'P-G':
            a = models_by_iso[iso][ch]['a']
            b = models_by_iso[iso][ch]['b']
            noisy_channels[ch] = add_noise_to_channel_poisson_gaussian(channels[ch], a, b)
        elif approach == 'AWGN-Pure':
            noisy_channels[ch] = add_awgn_noise_to_channel(channels[ch], int(iso), base_sigma=0.005)
        elif approach == 'AWGN':
            noisy_channels[ch] = add_awgn_noise_to_channel_weighted(channels[ch], int(iso), base_sigma=0.005)
        else:
            raise ValueError(f"unexpected approach = {approach}")

    # Convert back to uint16 using original black and white levels
    noisy_raw = merge_cfa_channels(noisy_channels,
                                   pattern_str,
                                   raw.black_level_per_channel,
                                   raw.white_level)
    raw.raw_image[:] = noisy_raw
    return raw


# use a clean image (ISO 100, for example) - don't use a high ISO input image that has noise already
# returns an RGB buffer. Pass None for iso_noise_to_inject to just decode the RAW
def process_raw_to_rgb_inject(raw_path, models_by_iso,
                              iso_noise_to_inject, bps, approach: str,
                              bit_depth: int,
                              no_auto_scale: bool,  # turns the image green
                              no_auto_bright: bool):
    with rawpy.imread(raw_path) as raw:
        if iso_noise_to_inject is not None:
            raw = inject_noise(raw, models_by_iso, iso_noise_to_inject, approach, bit_depth)

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


# convenience
def process_raw_to_rgb(raw_path, bps, approach, bit_depth, no_auto_scale: bool, no_auto_bright: bool):
    return process_raw_to_rgb_inject(raw_path, None, None, bps, approach, bit_depth,
                                     no_auto_bright=no_auto_bright, no_auto_scale=no_auto_scale)


def write_rgb(rgb, output_file_path, bps):
    if bps == 8:
        imageio.imwrite(output_file_path, rgb)
    else:
        # write a 16-bit tiff with sRGB color profile embedded
        image_uint16 = rgb.astype(np.uint16)
        # imageio.imwrite(output_file_path, image_uint16)

        with open('/System/Library/ColorSync/Profiles/sRGB Profile.icc', "rb") as f:  # replace with your path
            srgb_profile = f.read()

        tifffile.imwrite(
            output_file_path,
            image_uint16,
            photometric='rgb',
            extratags=[(34675, 1, len(srgb_profile), srgb_profile, False)]
        )


def write_synth_noisy_image(clean_path, models_by_iso, iso, output_path, bps, approach, bit_depth,
                            no_auto_scale: bool, no_auto_bright: bool):
    rgb = process_raw_to_rgb_inject(clean_path, models_by_iso, iso, bps,
                                    approach=approach, bit_depth=bit_depth,
                                    no_auto_scale=no_auto_scale,
                                    no_auto_bright=no_auto_bright)
    write_rgb(rgb, output_path, bps)


def write_clean_image(clean_path, output_path, bps, no_auto_scale: bool, no_auto_bright: bool):
    rgb = process_raw_to_rgb(clean_path, bps, approach=None, bit_depth=None,
                             no_auto_scale= no_auto_scale, no_auto_bright= no_auto_bright)
    write_rgb(rgb, output_path, bps)


# pass tags from exif_tags()
def get_exposure_time(tags) -> float:
    """
    Extract exposure time from RAW metadata.
    """
    if 'EXIF ExposureTime' in tags:
        shutter_speed_tag = tags['EXIF ExposureTime']
        # The value might be a Fraction object, convert to float for calculation
        shutter_speed_value = float(shutter_speed_tag.values[0])
        if shutter_speed_value != 0:
            return shutter_speed_value
    else:
        print("ExposureTime tag not found.")

    # Alternatively, try to get ShutterSpeedValue and convert from APEX
    if 'EXIF ShutterSpeedValue' in tags:
        shutter_speed_apex_tag = tags['EXIF ShutterSpeedValue']
        # APEX value is usually a Fraction or float
        shutter_speed_apex = float(shutter_speed_apex_tag.values[0])
        # Convert from APEX to seconds: t = 2^(-Tv)
        shutter_speed_seconds = 2 ** (-shutter_speed_apex)
        if shutter_speed_seconds != 0:
            return shutter_speed_seconds
    else:
        raise ValueError(f"Could not extract exposure time from RAW file")


# pass tags from exif_tags()
def get_iso(tags) -> int:
    iso_tag = tags.get('EXIF ISOSpeedRatings')
    if iso_tag:
        iso_speed = int(iso_tag.values[0])
        return iso_speed
    raise ValueError(f"Could not extract ISO from RAW file")

import exifread

def get_aperture_exifread(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
        fnumber = tags.get('EXIF FNumber')
        if fnumber:
            return float(fnumber.values[0].num) / float(fnumber.values[0].den)
    return None


def get_aperture(tags) -> int:
    the_tag = tags.get('EXIF FNumber')
    if the_tag:
        return float(the_tag.values[0].num) / float(the_tag.values[0].den)
    raise ValueError(f"Could not extract aperture from file'")


def get_focal_length(tags) -> int:
    the_tag = tags.get("EXIF FocalLength")
    if the_tag:
        return float(the_tag.values[0].num) / float(the_tag.values[0].den)
    raise ValueError(f"Could not extract focal length from file'")


def exif_tags(path) -> Dict[str, Any]:
    if os.path.exists(path) is False:
        return None
    with open(path, 'rb') as f:
        # Get EXIF tags
        tags = exifread.process_file(f)
        return tags
