import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import time


def load_data(file1, file2):
    """Load and extract coordinate data from two CSV files."""
    game_data = pd.read_excel(file1)
    processed_data = pd.read_csv(file2)
    
    global time_file
    time_file = str(file1).split('_')[-1]

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

    elif method == "ema":
        alpha = kwargs.get("alpha", 0.2)  # Smoothing factor (0 < alpha <= 1)
        x_smooth = x.ewm(alpha=alpha, adjust=False).mean()
        y_smooth = y.ewm(alpha=alpha, adjust=False).mean()

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
    plot_path = f'output/trajectory_plot_{time_file}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()


def compute_metrics(game_coords, smoothed_coords):
    """Compute tracking performance metrics."""
    game_x, game_y = np.array(game_coords[0]), np.array(game_coords[1])
    gaze_x, gaze_y = np.array(smoothed_coords[0]), np.array(smoothed_coords[1])

    # 1. Mean Absolute Error
    # Measures the average absolute difference between gaze-tracked coordinates and game object coordinates.
    # Lower values indicate better tracking accuracy
    mae = np.mean(np.abs(gaze_x - game_x) + np.abs(gaze_y - game_y))

    # 2. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((gaze_x - game_x) ** 2 + (gaze_y - game_y) ** 2))

    # 3. Cross-Correlation Between Gaze and Object Trajectory
    # Measures how well the gaze trajectory follows the object movement over time.
    # A high correlation (closer to 1) means the gaze movement aligns well with the object movement.
    corr_x, _ = pearsonr(game_x, gaze_x)
    corr_y, _ = pearsonr(game_y, gaze_y)

    # 4. Gaze Jitter (Variance of Position Changes)
    # Measures how much the gaze position fluctuates within short time intervals.
    # High jitter indicates noisy tracking, while low jitter suggests stable gaze tracking.
    jitter = np.mean(np.diff(gaze_x) ** 2 + np.diff(gaze_y) ** 2)

    # 5. Gaze Drift (Final Distance from Object)
    # Measures how far the gaze drifts away from the object over time.
    drift = euclidean((game_x[-1], game_y[-1]), (gaze_x[-1], gaze_y[-1]))

    # # 6. Percentage of Time on Target (PTT)
    # threshold = 50  # Define acceptable tracking error (in pixels)
    # within_threshold = np.sum(np.sqrt((gaze_x - game_x)**2 + (gaze_y - game_y)**2) < threshold)
    # ptt = (within_threshold / len(game_x)) * 100  # Percentage

    # Print results
    print("Tracking Metrics:")
    print(f"MAE: {mae:.2f} pixels")
    print(f"RMSE: {rmse:.2f} pixels")
    print(f"Cross-Correlation X: {corr_x:.2f}, Y: {corr_y:.2f}")
    print(f"Gaze Jitter: {jitter:.2f} pixels")
    print(f"Gaze Drift: {drift:.2f} pixels")
    # print(f"Percentage of Time on Target (PTT): {ptt:.2f}%")


# def main():
#     # # Horizontal movement data
#     # file1 = "output/Game_Horizontal_Medium_1738079657.csv"
#     # file2 = "output/processed_coordinates_1738275859.csv"

#     # Vertical movement data
#     file1 = "output/Game_Horizontal_Medium_1738678116.xlsx"
#     file2 = "output/processed_coordinates_1738678166.csv"

#     game_coords, processed_coords = load_data(file1, file2)
#     # print(f"{processed_coords=}")

#     smoothed_coords = apply_smoothing(
#         processed_coords[0], processed_coords[1], method="kalman", window_size=5
#     )

#     compute_metrics(game_coords, smoothed_coords)

#     plot_coordinates(game_coords, processed_coords, smoothed_coords)


# if __name__ == "__main__":
#     main()
