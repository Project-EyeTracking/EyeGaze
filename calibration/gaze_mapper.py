# Check for mirroring of screen

import json
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ScreenSpecs:
    """Screen specifications for gaze mapping."""

    width_pixels: int  # Screen width in pixels
    height_pixels: int  # Screen height in pixels
    width_cm: float  # Physical screen width in centimeters
    height_cm: float  # Physical screen height in centimeters
    distance_cm: float  # Distance from eye to screen in centimeters


def read_screen_specs(file_path: str) -> ScreenSpecs:
    """Reads screen specifications from a JSON file and returns a ScreenSpecs object."""
    with open(file_path) as file:
        data = json.load(file)
        return ScreenSpecs(
            width_pixels=data["width_pixels"],
            height_pixels=data["height_pixels"],
            width_cm=data["width_cm"],
            height_cm=data["height_cm"],
            distance_cm=data["distance_cm"],
        )


class GazeMapper:
    """Maps eye gaze angles (yaw, pitch) in radians to screen coordinates.

    This class implements a geometric approach to convert gaze angles
    to screen coordinates, assuming a fixed head position.

    Attributes:
        screen (ScreenSpecs): Screen specifications
        last_gaze_point: Optional[Tuple[int, int]]: Last valid gaze point
        visualization: Optional[np.ndarray]: Current visualization image
    """

    def __init__(self, screen: ScreenSpecs):
        """Initialize the GazeMapper with screen specifications.

        Args:
            screen (ScreenSpecs): Screen specifications including dimensions and distance
        """
        self.screen = screen
        self.last_gaze_point: Optional[Tuple[int, int]] = None
        self.visualization: Optional[np.ndarray] = None

        # Calculate maximum visible angles in radians based on screen dimensions
        self.max_yaw = np.arctan2(screen.width_cm / 2, screen.distance_cm)
        self.max_pitch = np.arctan2(screen.height_cm / 2, screen.distance_cm)

        # Initialize visualization image
        self._init_visualization()

    def _init_visualization(self):
        """Initialize the white visualization image."""
        # Create white image with screen dimensions
        self.visualization = (
            np.ones((self.screen.height_pixels, self.screen.width_pixels, 3), dtype=np.uint8) * 255
        )

    def _angle_to_cm(self, yaw: float, pitch: float) -> Tuple[float, float]:
        """Convert angles (in radians) to centimeters on screen plane.

        Args:
            yaw (float): Horizontal angle in radians
            pitch (float): Vertical angle in radians

        Returns:
            Tuple[float, float]: (x, y) coordinates in centimeters
        """
        # vertical_scaling_factor = (self.screen.width_cm / self.screen.height_cm) * (self.screen.width_pixels / self.screen.height_pixels)
        # vertical_scaling_factor = self.screen.height_cm / self.screen.width_cm
        # vertical_scaling_factor = 1.2
        # x = -self.screen.distance_cm * np.tan(yaw)
        # y = -self.screen.distance_cm * np.tan(pitch) * vertical_scaling_factor
        x = -self.screen.distance_cm * np.sin(yaw) * np.cos(pitch)
        y = -self.screen.distance_cm * np.sin(pitch)
        return x, y

    def _cm_to_pixels(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        """Convert centimeter coordinates to pixel coordinates.

        Args:
            x_cm (float): X coordinate in centimeters from screen center
            y_cm (float): Y coordinate in centimeters from screen center

        Returns:
            Tuple[int, int]: (x, y) coordinates in pixels
        """
        # Convert from center-origin to top-left origin coordinate system
        x_pixel = int(
            (x_cm + self.screen.width_cm / 2) * (self.screen.width_pixels / self.screen.width_cm)
        )
        y_pixel = int(
            (y_cm + self.screen.height_cm / 2)
            * (self.screen.height_pixels / self.screen.height_cm)
        )

        # Clamp values to screen boundaries
        x_pixel = max(0, min(x_pixel, self.screen.width_pixels - 1))
        y_pixel = max(0, min(y_pixel, self.screen.height_pixels - 1))

        return x_pixel, y_pixel

    def angles_to_screen_point(self, yaw: float, pitch: float) -> Tuple[int, int]:
        """Convert gaze angles in radians to screen coordinates.

        Args:
            yaw (float): Horizontal angle in radians (positive is right)
            pitch (float): Vertical angle in radians (positive is up)

        Returns:
            Tuple[int, int]: (x, y) coordinates in pixels

        Raises:
            ValueError: If angles are outside the visible range
        """
        # Validate input angles
        if abs(yaw) > self.max_yaw or abs(pitch) > self.max_pitch:
            if self.last_gaze_point is not None:
                return self.last_gaze_point
            raise ValueError(
                f"Angles out of range. Max yaw: ±{np.degrees(self.max_yaw):.1f}°, "
                f"Max pitch: ±{np.degrees(self.max_pitch):.1f}°"
            )

        # Convert angles to screen coordinates
        x_cm, y_cm = self._angle_to_cm(yaw, pitch)
        screen_point = self._cm_to_pixels(x_cm, y_cm)

        # Update last valid gaze point
        self.last_gaze_point = screen_point

        return screen_point

    def visualize_gaze_point(
        self,
        x: int,
        y: int,
        dot_radius: int = 10,
        dot_color: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR
        thickness: int = -1,  # Filled circle
        window_name: str = "Gaze Visualization",
        show: bool = True,
    ) -> np.ndarray:
        """Create or update visualization of gaze point on screen.

        Args:
            x (int): X coordinate in pixels
            y (int): Y coordinate in pixels
            dot_radius (int, optional): Radius of gaze point. Defaults to 10.
            dot_color (Tuple[int, int, int], optional): BGR color of dot. Defaults to red.
            thickness (int, optional): Line thickness. -1 for filled. Defaults to -1.
            window_name (str, optional): Name of display window. Defaults to "Gaze Visualization".
            show (bool, optional): Whether to display the window. Defaults to True.

        Returns:
            np.ndarray: Visualization image
        """
        # Reset visualization
        self._init_visualization()

        # Draw gaze point
        cv2.circle(
            self.visualization,
            center=(x, y),
            radius=dot_radius,
            color=dot_color,
            thickness=thickness,
        )

        # Add crosshair at screen center
        center_x = self.screen.width_pixels // 2
        center_y = self.screen.height_pixels // 2
        crosshair_size = 20
        cv2.line(
            self.visualization,
            (center_x - crosshair_size, center_y),
            (center_x + crosshair_size, center_y),
            (150, 150, 150),
            1,
        )
        cv2.line(
            self.visualization,
            (center_x, center_y - crosshair_size),
            (center_x, center_y + crosshair_size),
            (150, 150, 150),
            1,
        )

        # Display window if requested
        if show:
            cv2.imshow(window_name, self.visualization)
            cv2.waitKey(1)  # Update display

        return self.visualization

    def get_visible_range(self) -> dict:
        """Get the maximum visible angles (in both radians and degrees) and screen dimensions.

        Returns:
            dict: Dictionary containing max angles and screen dimensions
        """
        return {
            "max_yaw_rad": self.max_yaw,
            "max_pitch_rad": self.max_pitch,
            "max_yaw_deg": np.degrees(self.max_yaw),
            "max_pitch_deg": np.degrees(self.max_pitch),
            "screen_width_cm": self.screen.width_cm,
            "screen_height_cm": self.screen.height_cm,
            "screen_width_px": self.screen.width_pixels,
            "screen_height_px": self.screen.height_pixels,
            "distance_cm": self.screen.distance_cm,
        }

    def cleanup(self):
        """Clean up OpenCV windows."""
        cv2.destroyAllWindows()


# Example usage
def main():
    # Example screen specifications (24-inch 1920x1080 monitor at 40cm distance)
    screen = ScreenSpecs(
        width_pixels=1470,
        height_pixels=956,
        width_cm=30.4,
        height_cm=21.49,
        distance_cm=55.0,
    )

    # Initialize mapper
    mapper = GazeMapper(screen)

    # Print visible range info
    visible_range = mapper.get_visible_range()
    print("\nVisible Range:")
    print(
        f"Max Yaw: ±{visible_range['max_yaw_rad']:.3f} rad (±{visible_range['max_yaw_deg']:.1f}°)"
    )
    print(
        f"Max Pitch: ±{visible_range['max_pitch_rad']:.3f} rad (±{visible_range['max_pitch_deg']:.1f}°)"
    )
    print(
        f"Screen dimensions: {visible_range['screen_width_cm']}x{visible_range['screen_height_cm']} cm"
    )
    print(
        f"Screen resolution: {visible_range['screen_width_px']}x{visible_range['screen_height_px']} pixels"
    )
    print(f"Distance from screen: {visible_range['distance_cm']} cm")

    try:
        # Example angles in radians (approximately 10° yaw and -5° pitch)
        # yaw = np.radians(10.0)    # ≈ 0.175 radians
        # pitch = np.radians(-5.0)  # ≈ -0.087 radians
        # yaw = -0.053
        # pitch = -0.194
        yaw = -0.089
        pitch = 0.0656

        # Convert angles to screen coordinates
        x, y = mapper.angles_to_screen_point(yaw, pitch)
        print(
            f"\nGaze angles (yaw={np.degrees(yaw):.1f}° [{yaw:.3f} rad], "
            f"pitch={np.degrees(pitch):.1f}° [{pitch:.3f} rad])"
        )
        print(f"Screen point: ({x}, {y}) pixels")

        # Visualize the gaze point
        mapper.visualize_gaze_point(x, y)
        print("\nPress any key to exit visualization...")
        cv2.waitKey(0)

    except ValueError as e:
        print(f"Error: {e}")
    finally:
        mapper.cleanup()


if __name__ == "__main__":
    main()
