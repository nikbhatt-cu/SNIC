import json
import numpy as np
from parameters import CAMERA_MODEL
from pathlib import Path

'''
    Step 4: Tune the calibrated noise model. 

    IMPORTANT: For this and most later scripts, you configure it by changing the parameters.py script (see below import)
'''

DATA_PATH = "../data"
OUTPUT_DIR = "results"
MODEL_DIR = "P-G models"
ORIG_NOISE_MODEL_NAME = f"{CAMERA_MODEL}_noise_models_base.json"
OUTPUT_NOISE_MODEL_NAME = f"{CAMERA_MODEL}_noise_models.json"


def run():
    # Read the input JSON file
    base_dir = Path(DATA_PATH) / CAMERA_MODEL / OUTPUT_DIR / MODEL_DIR
    input_file = base_dir / ORIG_NOISE_MODEL_NAME
    output_file = base_dir / OUTPUT_NOISE_MODEL_NAME

    scale_factors_file = base_dir / "scale_factors.json"

    with open(input_file, 'r') as f:
        data = json.load(f)
        original_data = json.load(open(input_file, 'r'))  # Keep original for scale factor calculation

    # Dictionary to store scale factors
    scale_factors = {}

    # Get ISO values in order
    iso_values = sorted([int(iso) for iso in data.keys()])
    print(f"ISO values: {iso_values}")

    # The 2nd and 4th ISO values (indices 1 and 3)
    iso_2nd = iso_values[1]  # Should be 50
    iso_4th = iso_values[3]  # Should be 200

    print(f"\nUsing ISO {iso_2nd} and ISO {iso_4th} to define the line in log-log space")

    # Process each channel
    channels = ['R', 'G1', 'G2', 'B']

    for channel in channels:
        print(f"\n{channel} channel:")
        scale_factors[channel] = {}

        # Get the 'a' values at ISO 50 and ISO 200
        a_2nd = data[str(iso_2nd)][channel]['a']
        a_4th = data[str(iso_4th)][channel]['a']

        print(f"  Original: ISO {iso_2nd} -> a = {a_2nd:.10f}")
        print(f"  Original: ISO {iso_4th} -> a = {a_4th:.10f}")

        # Fit a line in log-log space: log(a) = m * log(ISO) + c
        # We have two points: (log(iso_2nd), log(a_2nd)) and (log(iso_4th), log(a_4th))
        log_iso_2nd = np.log(iso_2nd)
        log_iso_4th = np.log(iso_4th)
        log_a_2nd = np.log(a_2nd)
        log_a_4th = np.log(a_4th)

        # Calculate slope m
        m = (log_a_4th - log_a_2nd) / (log_iso_4th - log_iso_2nd)

        # Calculate intercept c
        c = log_a_2nd - m * log_iso_2nd

        print(f"  Power law: a = {np.exp(c):.10e} * ISO^{m:.6f}")

        # Update all ISO values with the fitted line and calculate scale factors
        for iso in iso_values:
            iso_str = str(iso)
            log_iso = np.log(iso)
            log_a_fitted = m * log_iso + c
            a_fitted = np.exp(log_a_fitted)

            original_a = original_data[iso_str][channel]['a']
            data[iso_str][channel]['a'] = a_fitted

            # Calculate scale factor: new_a / original_a
            scale_factor = a_fitted / original_a
            scale_factors[channel][iso_str] = scale_factor

            if iso == iso_2nd or iso == iso_4th:
                print(f"  ISO {iso:4d}: a = {a_fitted:.10f} (anchor, scale = {scale_factor:.6f})")
            else:
                print(f"  ISO {iso:4d}: a = {a_fitted:.10f} (was {original_a:.10f}, scale = {scale_factor:.6f})")

    # Write the updated JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Write the scale factors to a separate file
    with open(scale_factors_file, 'w') as f:
        json.dump(scale_factors, f, indent=2)

    print(f"\n" + "="*60)
    print(f"Updated noise model saved to: {output_file}")
    print(f"Scale factors saved to: {scale_factors_file}")
    print("="*60)

    # Print summary table of scale factors
    print("\nScale Factors Summary (new_a / original_a):")
    print("="*60)
    print(f"{'ISO':<8}", end="")
    for channel in channels:
        print(f"{channel:<15}", end="")
    print()
    print("-"*60)
    for iso in iso_values:
        iso_str = str(iso)
        print(f"{iso:<8}", end="")
        for channel in channels:
            sf = scale_factors[channel][iso_str]
            print(f"{sf:<15.6f}", end="")
        print()

if __name__ == '__main__':
    run()
