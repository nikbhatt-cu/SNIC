import rawpy
import numpy as np
import matplotlib.pyplot as plt
import os
import utilities

'''
    Step 1: This script determines which portion of the gray card should be used.
    You can pick one yourself by setting the y_min etc, or set USE_BEST_CROP
    (image portion will be camera dependent due to focal length and sensor).
    Recommendation: shoot at the lowest true ISO and a midpoint aperture and focal length.
    Have a controlled lighting setup (with a known white balance).
    Also set the exposure time so that the image is primarily mid-tones, not white or black.
    A good crop size is 512x512. 
'''


# these constants are here are set to defaults, but you can override them by calling driver() and passing
# parameters. main just calls the run() function with these defaults.

CAMERA_MODEL = "iPhone_11_Pro"
ISO = 32   # use the lowest true ISO
GRAY_CARD_FILE_NAME = "TETH7418_ISO_32_1_500s.DNG"

GRAY_CARD_DIR_NAME = "gray_card"
DATA_PATH = "../data"
TEST_SHOTS_DIR = "test_shots"
OUTPUT_DIR = "results"
STATS_DIR = "crop"

# recommended Std. Dev is 5 or less; recommended CV is 0.02 or less; note those will rise as ISO rises.
CROP_Y = 2000
CROP_X = 3300
HEIGHT = 512
WIDTH = 512
Y_MIN = 0
X_MIN = 0

USE_BEST_CROP = True        # True = override above crop rect
BEST_CROP_HEIGHT = 512      # overrides the crop height and width above.
BEST_CROP_WIDTH = 512       # overrides the crop height and width above.
STRIDE = 100


def find_best_crop(image, crop_height=HEIGHT, crop_width=WIDTH, stride=STRIDE):
    best_cv = float('inf')
    best_crop = None

    # Slide window across image
    for y in range(0, image.shape[0] - crop_height, stride):
        for x in range(0, image.shape[1] - crop_width, stride):
            crop = image[y:y + crop_height, x:x + crop_width]

            mean = np.mean(crop)
            std = np.std(crop)
            cv = std / mean

            # Also check it's not too dark/bright (card is properly exposed)
            if cv < best_cv:
                best_cv = cv
                best_crop = crop
                best_crop_rect = (y, x, crop_height, crop_width)

    return best_crop, best_crop_rect, best_cv


def check_crop_uniformity_rgb(crop_image, print_stats=True):
    h, w = crop_image.shape[:2]
    center = crop_image[h//3:2*h//3, w//3:2*w//3].mean()  # Green channel
    edges = [
        crop_image[0:h//4, w//3:2*w//3].mean(),  # Top
        crop_image[3*h//4:h, w//3:2*w//3].mean(),  # Bottom
        crop_image[h//3:2*h//3, 0:w//4].mean(),  # Left
        crop_image[h//3:2*h//3, 3*w//4:w].mean()  # Right
    ]
    variation = (max(edges) - min(edges)) / center

    if print_stats:
        print(f"Center: {center:.1f}")
        print(f"Edge variation: {variation * 100:.2f}%")

        if variation < 0.05:
            print(f"Good uniformity: {variation}")
        elif variation < 0.10:
            print(f"Acceptable, but could be better: {variation}")
        else:
            print(f"Too much variation - adjust lighting: {variation}")

    return variation


def check_crop_uniformity_per_channel(bayer, colors, crop_rect, print_stats=True):
    y, x, w, h = crop_rect
    crop_bayer = bayer[y:y + h, x:x + w]
    crop_colors = colors[y:y + h, x:x + w]

    variations = {}
    for i, ch_name in enumerate(['R', 'G', 'B', 'G2']):
        ch_data = crop_bayer[crop_colors == i]

        if len(ch_data) == 0:
            continue

        # Split into 3×3 grid, check variation
        ch_2d = crop_bayer.copy()
        ch_2d[crop_colors != i] = np.nan

        grid_means = []
        for row in range(3):
            for col in range(3):
                r_start = row * h // 3
                r_end = (row + 1) * h // 3
                c_start = col * w // 3
                c_end = (col + 1) * w // 3

                grid_section = ch_2d[r_start:r_end, c_start:c_end]
                grid_mean = np.nanmean(grid_section)
                grid_means.append(grid_mean)

        # Check variation across grid
        variation = (max(grid_means) - min(grid_means)) / np.mean(grid_means)

        if print_stats:
            print(f"{ch_name} channel:")
            print(f"  Grid variation: {variation * 100:.2f}%")
            print(f"  Center vs edges: {(grid_means[4] - np.mean(grid_means[:3])) / grid_means[4] * 100:.2f}%")

            if variation < 0.03:  # <3% variation
                print(f"  Excellent uniformity: {variation}")
            elif variation < 0.05:  # <5%
                print(f"  Good uniformity {variation}")
            elif variation < 0.10:  # <10%
                print(f"  Acceptable but not ideal: {variation}")
            else:
                print(f"  Too much variation - reduce crop size: {variation}")
            print()

        variations[ch_name] = variation

    return variations


def find_best_crop_checking_uniformity(bayer_data, raw_colors, rgb, crop_height=HEIGHT, crop_width=WIDTH, stride=STRIDE, print_stats=True):
    min_channel_variation = 100000.0
    min_rgb_variation = 100000.0

    best_origin_channel = (0, 0)        # y, x
    best_origin_rgb = (0, 0)        # y, x
    for y in range(0, rgb.shape[0] - crop_height, stride):
        for x in range(0, rgb.shape[1] - crop_width, stride):
            crop_rect = (y, x, crop_height, crop_width)
            channel_variations = check_crop_uniformity_per_channel(bayer_data, raw_colors, crop_rect, print_stats)
            max_iter_channel_variation = max(channel_variations.values())
            if max_iter_channel_variation < min_channel_variation:
                best_origin_channel = (y, x)
                min_channel_variation = max_iter_channel_variation

            crop_image = rgb[y:y + crop_height, x:x + crop_width]
            max_iter_rgb_variation = check_crop_uniformity_rgb(crop_image, print_stats)
            if max_iter_rgb_variation < min_rgb_variation:
                best_origin_rgb = (y, x)
                min_rgb_variation = max_iter_rgb_variation

    print(f"final channel stats: {best_origin_channel} x ({crop_height}, {crop_width}); variation = {min_channel_variation}")
    print(f"final rgb stats: {best_origin_rgb} x ({crop_height}, {crop_width}); variation = {min_rgb_variation}")
    crop_rect = (best_origin_channel[0], best_origin_channel[1], crop_height, crop_width)
    check_crop_uniformity_per_channel(bayer_data, raw_colors, crop_rect, True)

    return best_origin_channel[0], best_origin_channel[1], crop_height, crop_width


def run():
    global Y_MIN
    global X_MIN
    global HEIGHT
    global WIDTH

    input_path = os.path.join(os.path.join(DATA_PATH, CAMERA_MODEL), TEST_SHOTS_DIR)
    input_path = os.path.join(os.path.join(input_path, f"ISO_{ISO}"), GRAY_CARD_DIR_NAME)
    input_file_path = os.path.join(input_path, GRAY_CARD_FILE_NAME)

    output_path = os.path.join(os.path.join(DATA_PATH, CAMERA_MODEL), OUTPUT_DIR)
    output_path = os.path.join(os.path.join(os.path.join(output_path, f"ISO_{ISO}")), STATS_DIR)
    os.makedirs(output_path, exist_ok=True)

    # Load RAW image
    print(f"Analyzing {input_file_path}")
    with rawpy.imread(input_file_path) as raw:
        bayer_data = raw.raw_image.astype(np.float32)
        raw_colors = raw.raw_colors
        rgb = raw.postprocess(use_camera_wb=True,
                              no_auto_bright=True
                              )

    gray = np.mean(rgb, axis=2)  # Convert to grayscale

    crop_rect = (Y_MIN, X_MIN, HEIGHT, WIDTH)
    if USE_BEST_CROP:
        print("Looking for the best crop based on RGB")
        best = find_best_crop(gray, crop_height=BEST_CROP_HEIGHT, crop_width=BEST_CROP_WIDTH)
        crop_rect = best[1]
        print(f"find_best_crop returns rect = {crop_rect}")

        crop_rect = find_best_crop_checking_uniformity(bayer_data, raw_colors, rgb,
                                                       crop_height=BEST_CROP_HEIGHT, crop_width=BEST_CROP_WIDTH,
                                                       print_stats=False)
        print(f"find_best_crop with uniformity returns rect = {crop_rect}")

    cropped = gray[crop_rect[0]:crop_rect[0] + crop_rect[2], crop_rect[1]:crop_rect[1] + crop_rect[3]]    # cols:rows

    # stats to write out
    crop_loc = f"final crop_rect = {crop_rect}"
    print(crop_loc)

    check_crop_uniformity_per_channel(bayer_data, raw_colors, crop_rect)
    check_crop_uniformity_rgb(cropped)

    min_val = int(cropped.min())
    max_val = int(cropped.max())
    bins = np.arange(min_val, max_val + 2, 1)
    plt.hist(cropped.flatten(), bins=bins, edgecolor='white', linewidth=0.8, color='steelblue')
    ax = plt.gca()

    title = f"ISO {ISO} Intensity Histogram ({crop_rect[2]} x {crop_rect[3]})"
    ax.tick_params(axis='both', labelsize = 17)
    ax.set_xlabel('Pixel Intensity', fontsize=20)
    ax.set_ylabel(f'Number of Pixels', fontsize=20)
    plt.tight_layout()
    output_file_path = os.path.join(output_path, f"{title}.jpg")
    plt.savefig(output_file_path)
    plt.show()

    # Compute stats
    mean = np.mean(cropped)
    std = np.std(cropped)
    cv = std / mean  # Coefficient of variation

    # compute the stats in 100x100 tiles
    min_cv = 1000.0
    max_cv = 0.0
    for i in range(int(crop_rect[2] / 100)):
        for j in range(int(crop_rect[3] / 100)):
            region = cropped[i*100:(i+1)*100, j*100:(j+1)*100]
            mean_ij = np.mean(region)
            std_ij = np.std(region)
            cv_ij = std_ij / mean_ij
            min_cv = min(min_cv, cv_ij)
            max_cv = max(max_cv, cv_ij)
    print(f"CV range is [{min_cv:.4f}, {max_cv:.4f}]")      # want a small range

    stats = f"Mean: {mean:.2f}, Std Dev: {std:.2f}, CV: {cv:.4f}"
    print(stats)

    title = f"ISO {ISO} Stats ({crop_rect[0]} x {crop_rect[1]}, {crop_rect[2]} x {crop_rect[3]})"
    output_file_path = os.path.join(output_path, f"{title}.txt")
    with open(output_file_path, 'w') as file_object:
        file_object.write(crop_loc + "\n")
        file_object.write(stats)

    # if this happens - shoot a different image.
    if mean >= 255.0:
        raise ValueError(f"ERROR: Bad Image or crop area: needs to be a mid-tone gray")

    # show cropped image
    im = plt.imshow(cropped, cmap='gray')
    ax = plt.gca()
    title = f"ISO {ISO} Size {crop_rect[2]} x {crop_rect[3]}; Std Dev: {std:.2f}, CV: {cv:.4f}"
    title = f"Cropped Calibration Area"
    ax.tick_params(axis='both', labelsize = 18)
    ax.set_xlabel('X Coordinate', fontsize=20)
    ax.set_ylabel(f'Y Coordinate', fontsize=20)
    width = cropped.shape[1]  # or crop_rect[2]
    height = cropped.shape[0]  # or crop_rect[3]
    ax.set_xticks(range(0, width, 100))
    ax.set_yticks(range(0, height, 100))
    plt.tight_layout()

    output_file_path = os.path.join(output_path, f"{title}.jpg")
    plt.savefig(output_file_path)
    plt.show()


def driver(data_path: str = DATA_PATH,
           camera_model: str = CAMERA_MODEL,
           test_shots_dir: str = TEST_SHOTS_DIR,
           gray_card_dir_name: str = GRAY_CARD_DIR_NAME,
           gray_card_file_name: str = GRAY_CARD_FILE_NAME,
           stats_dir: str = STATS_DIR,
           output_dir: str = OUTPUT_DIR,
           iso: int = ISO,
           use_best_crop: bool = USE_BEST_CROP,
           best_crop_height: int = BEST_CROP_HEIGHT,
           best_crop_width: int = BEST_CROP_WIDTH,
           y_min: int = 0,
           x_min: int = 0,
           height: int = 0,
           width: int = 0,
           ):

    global ISO
    global DATA_PATH
    global CAMERA_MODEL
    global TEST_SHOTS_DIR
    global OUTPUT_DIR
    global STATS_DIR
    global GRAY_CARD_DIR_NAME
    global GRAY_CARD_FILE_NAME

    global Y_MIN
    global X_MIN
    global HEIGHT
    global WIDTH

    global USE_BEST_CROP
    global BEST_CROP_WIDTH
    global BEST_CROP_HEIGHT

    ISO = iso
    DATA_PATH = data_path
    CAMERA_MODEL = camera_model
    TEST_SHOTS_DIR = test_shots_dir
    OUTPUT_DIR = output_dir
    STATS_DIR = stats_dir
    GRAY_CARD_DIR_NAME = gray_card_dir_name
    GRAY_CARD_FILE_NAME = gray_card_file_name

    HEIGHT = height
    WIDTH = width

    USE_BEST_CROP = use_best_crop  # True = override crop rect
    BEST_CROP_WIDTH = best_crop_width  # overrides the crop height and width above.
    BEST_CROP_HEIGHT = best_crop_height  # overrides the crop height and width above.
    run()


if __name__ == '__main__':
    # if no parameters, it will find the best crop
    driver()

    # or pass parameters for a specific crop rect
    #driver(y_min = 1500, x_min = 3300, height = 512, width = 512, use_best_crop = False)
