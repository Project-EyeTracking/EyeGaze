import json
import math
import os
from collections import defaultdict

import numpy as np

WIDTH_PIXELS = 2560
HEIGHT_PIXELS = 1440


def calculate_metrics_statistics(directory_path, output_file):
    # Calculate max distance for normalization (screen diagonal)
    max_distance = np.sqrt(WIDTH_PIXELS**2 + HEIGHT_PIXELS**2)

    # Dictionary to store metrics values for each metric type
    metrics_values = defaultdict(list)
    normalized_jitter_values = []
    file_count = 0

    # First pass: collect all values
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path) as file:
                    data = json.load(file)

                    # Store each metric value
                    for metric_name, value in data["metrics"].items():
                        metrics_values[metric_name].append(float(value))

                        # Calculate normalized jitter for Gaze_Jitter
                        if metric_name == "Gaze_Jitter":
                            normalized_jitter = float(value) / max_distance
                            normalized_jitter_values.append(normalized_jitter)

                    file_count += 1
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")

    # Calculate statistics
    metrics_stats = {}
    if file_count > 0:
        for metric_name, values in metrics_values.items():
            # Calculate mean
            mean = sum(values) / len(values)

            # Calculate standard deviation
            squared_diff_sum = sum((x - mean) ** 2 for x in values)
            std_dev = math.sqrt(squared_diff_sum / len(values))

            metrics_stats[metric_name] = {"mean": round(mean, 4), "std_dev": round(std_dev, 4)}

        # Add normalized jitter statistics
        if normalized_jitter_values:
            normalized_mean = sum(normalized_jitter_values) / len(normalized_jitter_values)
            normalized_squared_diff_sum = sum(
                (x - normalized_mean) ** 2 for x in normalized_jitter_values
            )
            normalized_std_dev = math.sqrt(
                normalized_squared_diff_sum / len(normalized_jitter_values)
            )

            metrics_stats["Normalized_Gaze_Jitter"] = {
                "mean": round(normalized_mean, 4),
                "std_dev": round(normalized_std_dev, 4),
            }

    # Prepare output data structure
    output_data = {
        "number_of_files_processed": file_count,
        "screen_resolution": {"width": 2560, "height": 1440, "diagonal": round(max_distance, 2)},
        "metrics_statistics": metrics_stats,
    }

    # Write results to output file
    try:
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Results successfully written to {output_file}")
    except Exception as e:
        print(f"Error writing to output file: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Replace with your directory path containing JSON files
    directory_path = "output/metrics/horizontal"
    output_file = "output/metrics/metrics_horizontal_average.json"

    calculate_metrics_statistics(directory_path, output_file)
