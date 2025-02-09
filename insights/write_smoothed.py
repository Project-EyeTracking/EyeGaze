# scripts/main_script.py
import sys
import os
import pandas as pd
import numpy as np

# Get the absolute path to the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)  # Add root to Python path

from gen_insights import apply_smoothing  # Now you can import directly




files = os.listdir(f"{root_dir}\output\processed_csv")


for filename in files:

    processed_coords = pd.read_csv(f"{root_dir}\output\processed_csv\{filename}")
    smoothed_coords = apply_smoothing(processed_coords["ScreenX"], processed_coords["ScreenY"], method="kalman", window_size=5)

    try:
        # ... (code for reading CSV, smoothing, etc.) ...

        # 1. Ensure 'ScreenX' and 'ScreenY' columns exist in processed_coords
        if 'ScreenX' not in processed_coords.columns or 'ScreenY' not in processed_coords.columns:
            raise KeyError("ScreenX or ScreenY columns not found in processed_coords DataFrame.")

        # 2. Handle different types of smoothed_coords and ensure correct length
        if isinstance(smoothed_coords, pd.Series):
            if smoothed_coords.name == 'x':
                if len(smoothed_coords) != len(processed_coords):
                    raise ValueError("Length of smoothed_coords (x) and processed_coords must be the same.")
                processed_coords['ScreenX'] = smoothed_coords.values
            elif smoothed_coords.name == 'y':
                if len(smoothed_coords) != len(processed_coords):
                    raise ValueError("Length of smoothed_coords (y) and processed_coords must be the same.")
                processed_coords['ScreenY'] = smoothed_coords.values
            else:
                raise ValueError("Smoothed coordinates series name is invalid.")
        elif isinstance(smoothed_coords, tuple) or isinstance(smoothed_coords, np.ndarray):  # Handle tuples or NumPy arrays
            if len(smoothed_coords) != 2:  # Check if it has two elements (x and y)
                raise ValueError("Smoothed coordinates must contain two arrays (x and y).")

            smoothed_x, smoothed_y = smoothed_coords # unpack the tuple or numpy array

            if len(smoothed_x) != len(processed_coords) or len(smoothed_y) != len(processed_coords):
                raise ValueError("Length of smoothed_x, smoothed_y and processed_coords must be the same.")

            processed_coords['ScreenX'] = smoothed_x
            processed_coords['ScreenY'] = smoothed_y

        else:
            raise TypeError("Unsupported type for smoothed_coords.  Must be pandas Series, tuple, or numpy array.")

        # 3. Save the updated DataFrame (optional)
        output_file_path = os.path.join(root_dir, f"insights\smoothed_coord\{filename}") # Construct path relative to root
        processed_coords.to_csv(output_file_path, index=False)  # Save to a new CSV

        print(f"Smoothed coordinates written to {output_file_path}")

    except (ValueError, TypeError, KeyError) as e:  # Catch specific errors
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e: # Catch other potential errors during file reading
        print(f"An error occurred: {e}")
        sys.exit(1)

