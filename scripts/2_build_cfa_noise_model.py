import rawpy
import numpy as np
import glob
import os
import json
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
import utilities
from typing import Tuple, List
from collections import defaultdict
from itertools import combinations

'''
    Step 2: This builds a noise models using pairs of calibration images. The goal is to 
    take these pairs at each exposure time to get a range of brightness values
    using a gray card. This script produces a model with the suffix "_base". You will
    then tune it to have correct ISO scaling for noise.
    
    IMPORTANT: For this and most later scripts, you configure it by changing the parameters.py script (see below import)
'''

from parameters import CAMERA_MODEL, FILE_EXTENSION, ISO_LIST, CROP_X, CROP_Y, CROP_HEIGHT, CROP_WIDTH

# Config
DATA_PATH = "../data"
TEST_SHOTS_DIR = "test_shots"
OUTPUT_DIR = "results"
VALIDATION_DIR = "validation"
MODELS_DIR = "P-G models"
GRAY_CARD_DIR_NAME = "gray_card"
DARK_FRAME_DIR_NAME = "dark_frame"
CLEAN_DIR = "clean"
NOISY_DIR = "noisy"
CALIBRATION_PLOT_DIR_NAME = "calibration_plots"

PATCH_SIZE = 32    # about 36-50 patches per image
STD_THRESH = 0.03  # flatness threshold


def extract_patches(arr, P, thresh=None):
    h, w = arr.shape
    # thresh = STD_THRESH                 # ADJUST BY ISO; maybe 0.02 or 0.03 at high ISO or multiply by ISO somehow
    for i in range(0, h - P + 1, P):
        for j in range(0, w - P + 1, P):
            patch = arr[i:i+P, j:j+P]
            keep = True
            std = patch.std()
            var = patch.var()
            mean = patch.mean()
            max = patch.max()

            if mean < 0.001 or mean > 0.90 or max > 0.95:
                keep = False
                # print("skipping due to too dark or too bright")
            else:
                # Also check if variance is suspiciously low for signal level
                expected_min_var = mean * 0.0001  # Very rough heuristic
                if var < expected_min_var * 0.1:  # 10× below expected
                    print("skipping due to variance")
                    keep = False

            if thresh is not None:
                if std >= thresh:    # discard patches have too much texture / gradients
                    keep = False

            if keep:
                yield patch


def process_image(path):
    with rawpy.imread(path) as raw:
        pattern = raw.raw_pattern
        adjusted = utilities.adjust_pattern(pattern, CROP_Y, CROP_X)
        pattern_str = utilities.pattern_to_string(adjusted)

        raw_crop = raw.raw_image[CROP_Y:CROP_Y + CROP_HEIGHT, CROP_X:CROP_X + CROP_WIDTH].astype(np.float32)

        # Split into channels and normalize with per-channel black levels
        channels = utilities.split_cfa_channels(
            raw_crop,
            pattern_str,
            raw.black_level_per_channel,
            raw.white_level
        )
        return channels


def gather_stats_paired(image_pair_paths, remove_clipped = True):
    """
    Use image pairs to compute variance from difference.
    More robust than single-image variance.
    """
    stats = {'R': [], 'G1': [], 'G2': [], 'B': []}

    for path1, path2 in image_pair_paths:  # pairs at same exposure
        channels1 = process_image(path1)
        channels2 = process_image(path2)

        # remove pairs that have clipped data
        okay_to_use_pair = True
        for idx, buff in enumerate([channels1, channels2]):
            all_saturated = all([ch.mean() > 0.98 for ch in buff.values()])
            all_black = all([ch.mean() < 0.02 for ch in buff.values()])

            if all_saturated:
                path = [path1, path2][idx]
                exposure_time = utilities.get_exposure_time(utilities.exif_tags(path))
                print(f"{path} @ {exposure_time}: (all channels saturated)")
                if remove_clipped:
                    okay_to_use_pair = False

            if all_black:
                path = [path1, path2][idx]
                exposure_time = utilities.get_exposure_time(utilities.exif_tags(path))
                print(f"{path} @ {exposure_time}: (all channels black)")
                if remove_clipped:
                    okay_to_use_pair = False

        if okay_to_use_pair is False:
            print("skipping pair")
            continue

        for ch_name in channels1.keys():
            ch1 = channels1[ch_name]
            ch2 = channels2[ch_name]

            patches1 = list(extract_patches(ch1, PATCH_SIZE))
            patches2 = list(extract_patches(ch2, PATCH_SIZE))
            the_zip = zip(patches1, patches2)
            for patch1, patch2 in the_zip:
                mean = (patch1.mean() + patch2.mean()) / 2
                # Variance from difference: var = E[(I1-I2)²] / 2
                diff = patch1 - patch2
                var = (diff ** 2).mean() / 2
                stats[ch_name].append((mean, var))

    return stats


def fit_channel_stats_clamped(stats, clamp=True, useHuber = True):
    models = {}
    for ch, pairs in stats.items():
        X = np.array([m for m, v in pairs])
        y = np.array([v for m, v in pairs])

        if clamp:
            # Remove invalid values
            valid = np.isfinite(X) & np.isfinite(y)
            valid &= (y > 0)  # variance must be positive

            # Keep low signal for read noise estimation
            # But remove literal zeros (no information)
            valid &= (X >= 0)

            # Remove saturation region (non-linear)
            # Be conservative: remove top 5%
            if valid.sum() > 0:
                saturation = np.percentile(X[valid], 95)
                valid &= (X < saturation * 0.98)

            # Remove extremely low signals (dominated by quantization)
            min_signal = 0.02  # adjust based on normalization
            valid &= (X > min_signal)

            # Optional: Remove outliers using robust statistics
            if valid.sum() > 10:
                if useHuber:
                    X_valid = X[valid]
                    y_valid = y[valid]
                    # Fit preliminary model
                    from sklearn.linear_model import HuberRegressor
                    reg_robust = HuberRegressor().fit(X_valid.reshape(-1, 1), y_valid)
                    residuals = y_valid - reg_robust.predict(X_valid.reshape(-1, 1))
                    outlier_threshold = 3 * np.median(np.abs(residuals))
                    inliers = np.abs(residuals) < outlier_threshold
                    valid_indices = np.where(valid)[0]
                    valid[valid_indices[~inliers]] = False
                else:
                    # Fit once, remove outliers, refit
                    X_temp = X[valid].reshape(-1, 1)
                    y_temp = y[valid]
                    reg_temp = LinearRegression().fit(X_temp, y_temp)
                    residuals = y_temp - reg_temp.predict(X_temp)
                    outliers = np.abs(residuals) > 3 * np.std(residuals)
                    valid[np.where(valid)[0][outliers]] = False

            X = X[valid]
            y = y[valid]

        if len(X) < 10:
            print(f"Warning: Only {len(X)} valid samples for channel {ch}")

        X = X.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)

        models[ch] = {
            'a': reg.coef_[0].item(),
            'b': max(0, reg.intercept_.item()),  # variance can't be negative
            'sigma_read': np.sqrt(max(0, reg.intercept_.item())),
            'n_samples': len(X),
            'r2': reg.score(X, y)
        }

    return models


def plot_calibration(iso, stats, models, output_directory, channel='R'):
    pairs = stats[channel]
    X = np.array([m for m, v in pairs])
    y = np.array([v for m, v in pairs])

    model = models[channel]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    plt.scatter(X, y, alpha=0.8, s=1)

    x_line = np.linspace(0, X.max(), 100)
    y_line = model['a'] * x_line + model['b']
    plt.plot(x_line, y_line, 'r-', linewidth=2,
             label=f'Fit: σ² = {model["a"]:.4e}·μ + {model["b"]:.4e}')

    ax.set_xlabel('Mean Signal (μ)', fontsize=20)
    ax.set_ylabel(f'Variance (σ²)', fontsize=20)
    ax.set_title(f'ISO {iso} Photon Transfer Curve - {channel} Channel\n' + f'R² = {model["r2"]:.4f}',
                 fontsize=20)
    ax.tick_params(axis='both', labelsize = 18)
    plt.legend(fontsize = 20)
    plt.grid(alpha=0.3)
    calib_plot_dir = os.path.join(output_directory, CALIBRATION_PLOT_DIR_NAME)
    os.makedirs(calib_plot_dir, exist_ok=True)
    output_file = os.path.join(calib_plot_dir, f"ISO_{iso}.jpg")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def make_path_pairs(paths) -> List[Tuple[str, str]]:
    exposure_time_dict = defaultdict(list)
    for path in paths:
        tags = utilities.exif_tags(path)
        exposure_time = utilities.get_exposure_time(tags)
        exposure_time_dict[exposure_time].append(path)

    pairs = []
    for file_paths in exposure_time_dict.values():
        if len(file_paths) >= 2:
            pairs.extend(combinations(file_paths, 2))

    sorted_pairs = sorted(pairs)
    return sorted_pairs


if __name__ == '__main__':
    start_time = time.time()
    output_directory = os.path.join(os.path.join(DATA_PATH, CAMERA_MODEL), OUTPUT_DIR)
    all_iso_output = dict()
    os.makedirs(output_directory, exist_ok=True)

    for iso in ISO_LIST:
        iso_start_time = time.time()

        # set some globals
        ISO_DIR = f"ISO_{iso}"
        base_path = os.path.join(os.path.join(DATA_PATH, CAMERA_MODEL), TEST_SHOTS_DIR)
        iso_path = os.path.join(base_path, ISO_DIR)
        gray_path = os.path.join(os.path.join(iso_path, GRAY_CARD_DIR_NAME), f"*.{FILE_EXTENSION}")
        gray_paths = glob.glob(gray_path)

        dark_path = os.path.join(os.path.join(iso_path, DARK_FRAME_DIR_NAME), f"*.{FILE_EXTENSION}")
        dark_paths = glob.glob(dark_path)

        # stats
        path_pairs = make_path_pairs(gray_paths)
        gray_stats = gather_stats_paired(path_pairs)
        models_gray = fit_channel_stats_clamped(gray_stats)
        plot_calibration(iso = iso, stats=gray_stats, models=models_gray, output_directory = output_directory)

        # Print results if desired
        '''
        for ch, params in models_gray.items():
            print(f'ISO {iso}: Gray: {ch}: σ² = {params["a"]:.3e}·I + {params["b"]:.3e}')

        for ch, params in models_dark.items():
            print(f'ISO {iso}: Dark: {ch}: σ² = {params["a"]:.3e}·I + {params["b"]:.3e}')
        '''

        # prepare for json
        for ch, params in models_gray.items():
            params["a"] = float(params["a"])
            params["b"] = float(params["b"])
            models_gray[ch] = params

        # final numbers
        for ch, params in models_gray.items():
            print(f'ISO {iso}:\t{ch}:\tσ² = \t{params["a"]:.3e}\t·I + \t{params["b"]:.3e}')

        all_iso_output[iso] = models_gray

    # output a JSON to describe the calibrated noise model. Run the next script to tune the calibration
    # based on real noisy images (of regular scenes, not flat fields)
    model_directory = os.path.join(output_directory, MODELS_DIR)
    os.makedirs(model_directory, exist_ok=True)
    output_file_name = f'{CAMERA_MODEL}_noise_models_base.json'
    output_path = os.path.join(model_directory, output_file_name)
    with open(output_path, 'w') as f:
        json.dump(all_iso_output, f, indent=2)
    print(f"wrote {output_path}; took {time.time() - iso_start_time}")
    print(f"total run time {time.time() - start_time}")
