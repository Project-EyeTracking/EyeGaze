import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter


def load_data(file1, file2):
    """Load and extract coordinate data from two CSV files."""
    game_data = pd.read_csv(file1)
    processed_data = pd.read_csv(file2)

    game_coords = (game_data["GameX"], game_data["GameY"])
    processed_coords = (processed_data["ScreenX"], processed_data["ScreenY"])

    return game_coords, processed_coords


def apply_smoothing(x, y, method="moving_average", **kwargs):
    """Apply smoothing to coordinate data using specified method.

    Args:
        x (pd.Series): X coordinates
        y (pd.Series): Y coordinates
        method (str): Smoothing method ('moving_average', 'savgol', or 'kalman')
        **kwargs: Additional parameters for smoothing methods

    Returns:
        tuple: Smoothed x and y coordinates
    """
    if method == "moving_average":
        window_size = kwargs.get("window_size", 30)
        x_smooth = x.rolling(window=window_size, min_periods=1).mean()
        y_smooth = y.rolling(window=window_size, min_periods=1).mean()

    elif method == "savgol":
        window_size = kwargs.get("window_size", 29)  # Must be odd
        poly_order = kwargs.get("poly_order", 2)
        x_smooth = savgol_filter(x, window_length=window_size, polyorder=poly_order)
        y_smooth = savgol_filter(y, window_length=window_size, polyorder=poly_order)

    elif method == "kalman":
        window_size = kwargs.get("window_size", 10)

        # Initialize arrays to store smoothed coordinates
        x_smooth = np.zeros_like(x)
        y_smooth = np.zeros_like(y)

        # Initialize Kalman Filter for 2D Gaze Tracking
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: (x, y, vx, vy), Measurement: (x, y)

        # Define Transition Matrix F (Assuming Constant Velocity Model)
        dt = 1 / 30  # Time step (30 FPS)
        kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Define Measurement Matrix H (We only observe x, y)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Define Initial State and Covariance
        kf.x = np.array([x.iloc[0], y.iloc[0], 0, 0])  # Initial state (x, y, vx, vy)
        kf.P = np.eye(4) * 1000  # Initial uncertainty
        kf.R = np.array([[10, 0], [0, 10]])  # Measurement noise (adjust based on model accuracy)
        kf.Q = np.eye(4) * 0.01  # Process noise (adjust for smoothness)

        for i in range(len(x)):
            current_measurement = np.array([x.iloc[i], y.iloc[i]])

            if i % window_size == 0:
                # Predict step
                kf.predict()
                x_smooth[i] = kf.x[0]
                y_smooth[i] = kf.x[1]
            else:
                # Update step
                if not np.isnan(current_measurement).any():
                    kf.update(current_measurement)
                x_smooth[i] = kf.x[0]
                y_smooth[i] = kf.x[1]

    return x_smooth, y_smooth


def plot_coordinates(game_coords, processed_coords, smoothed_coords):
    """Create and display the coordinate plot."""
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(3, 1)

    # Extract coordinates
    game_x, game_y = game_coords
    proc_x, proc_y = processed_coords
    smooth_x, smooth_y = smoothed_coords

    # Create time array
    time = np.arange(len(game_x))

    # Plot X coordinates over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, game_x, "b-", label="Game Data", alpha=0.8)
    ax1.plot(time, smooth_x, "g--", label="Smoothed Data", alpha=0.8)
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("X Coordinate")
    ax1.set_title("X Coordinates Over Time")
    ax1.grid(True)
    ax1.legend()

    # Plot Y coordinates over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, game_y, "b-", label="Game Data", alpha=0.8)
    ax2.plot(time, smooth_y, "g--", label="Smoothed Data", alpha=0.8)
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Y Coordinate")
    ax2.set_title("Y Coordinates Over Time")
    ax2.grid(True)
    ax2.legend()

    # Plot X-Y trajectory
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(game_x, game_y, "b-", label="Game Data", alpha=0.8)
    ax3.plot(smooth_x, smooth_y, "g--", label="Smoothed Data", alpha=0.8)
    # plt.scatter(x1, y1, label='Processed Data', marker='x', alpha=0.8)
    ax3.set_xlabel("X Coordinate")
    ax3.set_ylabel("Y Coordinate")
    ax3.set_title("X-Y Trajectory")
    ax3.grid(True)
    ax3.legend()
    ax3.invert_yaxis()

    # Adjust layout
    plt.tight_layout()
    plt.savefig("output/trajectory_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    # # Horizontal movement data
    # file1 = "output/Game_Horizontal_Medium_1738079657.csv"
    # file2 = "output/processed_coordinates_1738275859.csv"

    # Vertical movement data
    file1 = "output/Game_Vertical_Medium_1738079713.csv"
    file2 = "output/processed_coordinates_1738079713.csv"

    game_coords, processed_coords = load_data(file1, file2)
    # print(f"{processed_coords=}")

    smoothed_coords = apply_smoothing(
        processed_coords[0], processed_coords[1], method="kalman", window_size=5
    )

    plot_coordinates(game_coords, processed_coords, smoothed_coords)


if __name__ == "__main__":
    main()
