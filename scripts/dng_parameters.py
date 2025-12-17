from parameters import CAMERA_MODEL

if CAMERA_MODEL == "iPhone_11_Pro":
    DNG_BAYER_PATTERN = 1  # 0=GRBG, 1=RGGB, 2=GBRG, 3=BGGR
    DNG_CAMERA_MAKE = "APPLE"
    DNG_CAMERA_MODEL_NAME = "iPhone12,3 back camera"

    DNG_ILLUMINANT1 = 17
    DNG_FORWARD_MATRIX1 = None
    DNG_COLOR_MATRIX1 = [1.233062, -0.5465536, -0.2566419,
                         -0.4623892, 1.508173, 0.00938271,
                         -0.02246051, 0.1456095, 0.5877981]

    DNG_ILLUMINANT2 = 21
    DNG_FORWARD_MATRIX2 = None
    DNG_COLOR_MATRIX2 = [0.8899321, -0.302455, -0.1254793,
                         -0.4658667, 1.333127, 0.105414,
                         -0.1035988, 0.2399584, 0.4020588]

    DNG_ACTIVE_AREA = [0, 0, 3024, 4032]    # top, left, bottom, right
    DNG_CROP_ORIGIN = [0, 0]  # X, Y
    DNG_CROP_SIZE = [4032, 3024]  # Width, Height
    DNG_BOUNDS = DNG_ACTIVE_AREA
    DNG_BASELINE_EXPOSURE = 0.0

elif CAMERA_MODEL == "iPhone_11_Pro_tele":
    DNG_BAYER_PATTERN = 1           # 0=GRBG, 1=RGGB, 2=GBRG, 3=BGGR
    DNG_CAMERA_MAKE = "APPLE"
    DNG_CAMERA_MODEL_NAME = "iPhone12,3 back telephoto camera"

    DNG_ILLUMINANT1 = 17
    DNG_FORWARD_MATRIX1 = None
    DNG_COLOR_MATRIX1 = [1.199428, -0.4989913, -0.2554508,
                         -0.4223687, 1.454964, 0.03593826,
                         -0.01624137, 0.1555983, 0.6283963]

    DNG_ILLUMINANT2 = 21
    DNG_FORWARD_MATRIX2 = None
    DNG_COLOR_MATRIX2 = [0.8642834, -0.2705879, -0.1183693,
                         -0.4209237, 1.274108, 0.1193669,
                         -0.09458346, 0.2356555, 0.4472266]

    DNG_ACTIVE_AREA = [0, 0, 3024, 4032]    # top, left, bottom, right
    DNG_CROP_ORIGIN = [0, 0]  # X, Y
    DNG_CROP_SIZE = [4032, 3024]  # Width, Height
    DNG_BOUNDS = DNG_ACTIVE_AREA
    DNG_BASELINE_EXPOSURE = 0.0

elif CAMERA_MODEL == "Sony_A7R_III":
    DNG_BAYER_PATTERN = 1           # 0=GRBG, 1=RGGB, 2=GBRG, 3=BGGR
    DNG_CAMERA_MAKE = "SONY"
    DNG_CAMERA_MODEL_NAME = "Sony ILCE-7RM3"

    DNG_ILLUMINANT1 = 17
    DNG_FORWARD_MATRIX1 = [0.5135, 0.3089, 0.1418,
                           0.2384, 0.7198, 0.0418,
                           0.0973, 0.0002, 0.7276]
    DNG_COLOR_MATRIX1 = [0.7683, -0.3276, 0.0299,
                         -0.363, 1.0892, 0.3161,
                         -0.0162, 0.0671, 0.7133]

    DNG_ILLUMINANT2 = 21
    DNG_FORWARD_MATRIX2 = [0.5493, 0.2579, 0.1572,
                           0.3238, 0.628, 0.0482,
                           0.1544, 0.0006, 0.6701]
    DNG_COLOR_MATRIX2 = [0.664, -0.1847, -0.0503,
                         -0.5238, 1.301, 0.2474,
                         -0.0993, 0.1673, 0.6527]

    DNG_ACTIVE_AREA = [0, 0, 5320, 8000]    # top, left, bottom, right
    DNG_CROP_ORIGIN = [8, 8]      # X, Y
    DNG_CROP_SIZE = [7952, 5304]       # Width, Height
    DNG_BOUNDS = DNG_ACTIVE_AREA
    DNG_BASELINE_EXPOSURE = 0.0

elif CAMERA_MODEL == "Sony_RX100_IV":
    DNG_BAYER_PATTERN = 1           # 0=GRBG, 1=RGGB, 2=GBRG, 3=BGGR
    DNG_CAMERA_MAKE = "SONY"
    DNG_CAMERA_MODEL_NAME = "Sony DSC-RX100M4"

    DNG_ILLUMINANT1 = 17
    DNG_FORWARD_MATRIX1 = [0.7978, 0.1352, 0.0313,
                           0.288, 0.7119, 0.0001,
                           0, 0, 0.8251]
    DNG_COLOR_MATRIX1 = [0.736600, -0.321300, 0.038000,
                         -0.360900, 1.112700, 0.285200,
                         -0.021800, 0.069400, 0.582100]

    DNG_ILLUMINANT2 = 21
    DNG_FORWARD_MATRIX2 = [0.7978, 0.1352, 0.0313,
                           0.288, 0.7119, 0.0001,
                           0, 0, 0.8251]
    DNG_COLOR_MATRIX2 = [0.659600, -0.207900, -0.056200,
                         -0.478200, 1.301600, 0.193300,
                         -0.097000, 0.158100, 0.518100]

    DNG_ACTIVE_AREA = [0, 0, 3672, 5504]    # top, left, bottom, right
    DNG_CROP_ORIGIN = [12, 12]      # X, Y
    DNG_CROP_SIZE = [5472, 3648]       # Width, Height
    DNG_BOUNDS = DNG_ACTIVE_AREA
    DNG_BASELINE_EXPOSURE = 0.0
