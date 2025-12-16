from typing import Dict, List
import numpy as np

all_channels = ['R', 'G1', 'G2', 'B']


def channel_dict_to_json_dict(channel_dict: dict) -> dict:
    result = dict()
    for channel in all_channels:
        sub_dict = dict()
        array = channel_dict[channel]
        data = array.tolist()
        dtype = str(array.dtype)
        shape = array.shape
        sub_dict["array"] = data
        sub_dict["dtype"] = dtype
        sub_dict["shape"] = shape
        result[channel] = sub_dict
    return result


def json_dict_to_channel_dict(json_dict: dict) -> dict:
    result = dict()
    for channel in json_dict:
        sub_dict = json_dict[channel]
        data = sub_dict["array"]
        dtype = sub_dict["dtype"]
        shape = tuple(sub_dict["shape"])
        array = np.array(data, dtype=dtype).reshape(shape)
        result[channel] = array
    return result


class SensorCalibrationISO:
    """
        this stores dark current files per ISO
    """

    def __init__(self, bit_depth: int = 14):
        self.iso = 0    # ISO for this model
        self.bit_depth = bit_depth
        self.dark_current_exposure_time = 0.0
        self.dark_current = None  # Dict[channel, ndarray]

        # this is the Zhang "database" -> {exposure_time : [path]}
        # that's because storing all the RAW data would be too large / slow
        self.dark_current_dict: Dict[float, List[int]] = None

    @staticmethod
    def newWithJSON(iso: int, calibration_dict: dict):
        obj = SensorCalibrationISO()
        obj.iso = int(calibration_dict["ISO"])
        assert(int(iso) == obj.iso)
        obj.bit_depth = int(calibration_dict["Bit Depth"])

        dc = calibration_dict.get("Dark Current")
        if dc is not None:
            obj.dark_current = json_dict_to_channel_dict(dc)

        obj.dark_current_exposure_time = calibration_dict.get("Dark Current Exposure Time")
        obj.dark_current_dict = calibration_dict.get("Dark Current Database")
        return obj

    def json_dictionary(self) -> dict:           # for output
        result = dict()
        result["ISO"] = self.iso
        result["Bit Depth"] = self.bit_depth

        # for the dark current, need to flatten the arrays
        if self.dark_current is not None:
            result["Dark Current"] = channel_dict_to_json_dict(self.dark_current)

        result["Dark Current Exposure Time"] = self.dark_current_exposure_time
        result["Dark Current Database"] = self.dark_current_dict
        return result

    def sample_dark_patches(self, channel: str,
                            target_exposure_time: float,
                            clean_iso: int,
                            target_iso: int,
                            output_shape: tuple):
        h, w = output_shape
        dark_current_channel = self.dark_current[channel]
        dark_h, dark_w = dark_current_channel.shape

        dark_full = np.zeros(output_shape)

        # Fill with random patches from dark_ref
        patch_size = min(dark_h, dark_w)

        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                # Random location in dark_ref
                dy = np.random.randint(0, max(1, dark_h - patch_size + 1))
                dx = np.random.randint(0, max(1, dark_w - patch_size + 1))

                # Extract patch
                patch_h = min(patch_size, h - y, dark_h - dy)
                patch_w = min(patch_size, w - x, dark_w - dx)
                patch = dark_current_channel[dy:dy + patch_h, dx:dx + patch_w]

                # Place in output
                dark_full[y:y + patch_h, x:x + patch_w] = patch

        # scale for output exposure time
        target_exposure = target_exposure_time * (clean_iso / target_iso)
        scale_factor = target_exposure / self.dark_current_exposure_time

        return dark_full * scale_factor


class SensorCalibration:
    # this uses numpy data structures
    def __init__(self):
        self.iso_calibration_dict = dict()

    # this loads JSON and turns it into numpy data structures
    @staticmethod
    def newWithJSON(json_dict: dict):
        obj = SensorCalibration()

        calib_dict = dict()
        for iso, iso_calibration in json_dict["Sensor ISO Calibration"].items():
            iso_calib = SensorCalibrationISO.newWithJSON(iso, iso_calibration)
            calib_dict[iso] = iso_calib
        obj.iso_calibration_dict = calib_dict
        return obj

    def json_dictionary(self) -> dict:           # for output
        output_dict = dict()
        iso_dict = dict()
        for iso, calibration in self.iso_calibration_dict.items():
            iso_dict[iso] = calibration.json_dictionary()

        output_dict["Sensor ISO Calibration"] = iso_dict
        return output_dict
