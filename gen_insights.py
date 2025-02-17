import json
import pathlib

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr


def load_constants(json_file_path):

    global WIDTH_PIXELS, HEIGHT_PIXELS

    try:
        with open(json_file_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    WIDTH_PIXELS = data.get("width_pixels", 0)
    HEIGHT_PIXELS = data.get("height_pixels", 0)


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

    elif method == "ema":
        alpha = kwargs.get("alpha", 0.2)  # Smoothing factor (0 < alpha <= 1)
        x_smooth = x.ewm(alpha=alpha, adjust=False).mean()
        y_smooth = y.ewm(alpha=alpha, adjust=False).mean()

    return x_smooth, y_smooth


def plot_coordinates(game_coords, processed_coords, smoothed_coords, title):
    """Create and display the coordinate plot."""
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1)

    # Extract coordinates
    game_x, game_y = game_coords
    proc_x, proc_y = processed_coords
    smooth_x, smooth_y = smoothed_coords

    # Create time array
    time = np.arange(len(game_x))

    # Plot X coordinates over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, game_x, "b-", label="Game Position", alpha=0.8)
    ax1.plot(time, smooth_x, "g--", label="Gaze Position (Smoothed)", alpha=0.8)
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("X Coordinate")
    ax1.set_title("X Coordinates Over Time")
    ax1.grid(True)
    ax1.legend()
    ax1.invert_yaxis()

    # Plot Y coordinates over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, game_y, "b-", label="Game Position", alpha=0.8)
    ax2.plot(time, smooth_y, "g--", label="Gaze Position (Smoothed)", alpha=0.8)
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Y Coordinate")
    ax2.set_title("Y Coordinates Over Time")
    ax2.grid(True)
    ax2.legend()
    ax2.invert_yaxis()

    # # Plot X-Y trajectory
    # ax3 = fig.add_subplot(gs[2, 0])
    # ax3.plot(game_x, game_y, "b-", label="Game Data", alpha=0.8)
    # ax3.plot(smooth_x, smooth_y, "g--", label="Smoothed Data", alpha=0.8)
    # # plt.scatter(x1, y1, label='Processed Data', marker='x', alpha=0.8)
    # ax3.set_xlabel("X Coordinate")
    # ax3.set_ylabel("Y Coordinate")
    # ax3.set_title("X-Y Trajectory")
    # ax3.grid(True)
    # ax3.legend()
    # ax3.invert_yaxis()

    # Adjust layout
    plt.tight_layout()
    plot_path = f"output/plot/trajectory_plot_{title}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    # plt.show()

    return fig


def animate_points(game_coords, processed_coords, timestamp):
    """Create animation of game coordinates and processed coordinates."""

    fig, ax = plt.subplots()
    ax.set_xlim(0, WIDTH_PIXELS)
    ax.set_ylim(0, HEIGHT_PIXELS)

    game_x, game_y = game_coords
    proc_x, proc_y = processed_coords

    # Plot instruction points (game coordinates)
    (game_line,) = ax.plot([], [], "ro", label="Game Position")

    # Plot observation points (processed coordinates)
    (proc_line,) = ax.plot([], [], "bo", label="Gaze Position (Smoothed)", alpha=0.5)

    def init():
        game_line.set_data([], [])
        proc_line.set_data([], [])
        return game_line, proc_line

    def update(frame):
        # Update points up to current frame
        game_line.set_data(game_x[:frame], game_y[:frame])
        proc_line.set_data(proc_x[:frame], proc_y[:frame])
        return game_line, proc_line

    # Create animation using the number of frames in the data
    n_frames = len(game_x)
    ani = animation.FuncAnimation(
        fig, update, frames=range(n_frames), init_func=init, blit=True, interval=33.3
    )

    # Add legend
    ax.legend()
    ax.invert_yaxis()
    ax.set_title("Gaze Tracking Trajectory")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)

    # Save the animation as a video file
    Writer = animation.writers["ffmpeg"]
    writer = Writer(
        fps=30, metadata=dict(artist="EyeTracker"), bitrate=2000, extra_args=["-vcodec", "libx264"]
    )

    video_dir = pathlib.Path("output/output_video/")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = f"{video_dir}/trajectory_video_{timestamp}.mp4"

    ani.save(video_path, writer=writer)
    plt.close()
    return video_path


def compute_metrics(game_coords, smoothed_coords, timestamp):
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

    # Create metrics dictionary
    metrics = {
        "timestamp": timestamp,
        "metrics": {
            "MAE": float(f"{mae:.2f}"),
            "RMSE": float(f"{rmse:.2f}"),
            "Cross_Correlation_X": float(f"{corr_x:.2f}"),
            "Cross_Correlation_Y": float(f"{corr_y:.2f}"),
            "Gaze_Jitter": float(f"{jitter:.2f}"),
            "Gaze_Drift": float(f"{drift:.2f}"),
        },
    }

    return metrics


def save_metrics(metrics, output_dir="output/metrics"):
    """Save metrics to a JSON file with timestamp."""
    # Create output directory if it doesn't exist
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create filename with timestamp
    filename = f"metrics_{metrics['timestamp']}.json"
    filepath = output_dir / filename

    # Save metrics to JSON file
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)

    return filepath


def generate_insight_plots(file1, file2):

    load_constants("calibration/screen_spec.json")

    timestamp = str(file1).split("_")[-1][:-4]

    game_coords, processed_coords = load_data(file1, file2)

    smoothed_coords = apply_smoothing(
        processed_coords[0], processed_coords[1], method="kalman", window_size=5
    )

    # Compute and save metrics
    metrics = compute_metrics(game_coords, smoothed_coords, timestamp)
    metrics_file = save_metrics(metrics)

    # Generate plots
    figures = plot_coordinates(game_coords, processed_coords, smoothed_coords, timestamp)
    video_path = animate_points(game_coords, smoothed_coords, timestamp)

    return figures, metrics, video_path


if __name__ == "__main__":

    # # horizontal movement data
    # game_file = "output/game_coordinates_1738772966.csv"
    # processed_file = "output/processed_coordinates_1738772966.csv"

    # vertical movement data
    game_file = "output/game_csv/game_coordinates_1738773281.csv"
    processed_file = "output/processed_csv/processed_coordinates_1738773281.csv"

    _, _ = generate_insight_plots(game_file, processed_file)
