import csv
import json
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import cv2
import pygame


@dataclass
class GameConfig:
    """Configuration settings for the game."""

    width: int
    height: int
    colors: Dict[str, Tuple[int, int, int]]
    speed_values: Dict[str, int]
    game_duration: int


class MovementTracker:
    """Main game class handling movement tracking and recording."""

    def __init__(self, spec_file: str):
        pygame.init()
        self.spec_file = spec_file
        self.config = self._load_config()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("Object Movement Tracker")
        self.font = pygame.font.Font(None, 36)
        self.movement_type = "Both"
        self.speed_choice = "Medium"
        self.cwd = pathlib.Path.cwd()

    def _load_config(self) -> GameConfig:
        """Load game configuration from JSON and return GameConfig object."""
        with open(self.spec_file) as file:
            data = json.load(file)
        return GameConfig(
            width=data.get("width_pixels"),
            height=data.get("height_pixels") - 100,
            colors={
                "WHITE": (255, 255, 255),
                "RED": (255, 0, 0),
                "BLACK": (0, 0, 0),
                "GRAY": (200, 200, 200),
            },
            speed_values={"Slow": 2, "Medium": 6, "Fast": 12},
            game_duration=20,
        )

    def _create_button(
        self, text: str, position: Tuple[int, int], size: Tuple[int, int], is_selected: bool
    ) -> pygame.Rect:
        """Create and draw a button."""
        button_rect = pygame.Rect(position, size)
        pygame.draw.rect(
            self.screen,
            self.config.colors["GRAY"] if is_selected else self.config.colors["WHITE"],
            button_rect,
        )
        text_surface = self.font.render(text, True, self.config.colors["BLACK"])
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)
        return button_rect

    def setup_screen(self):
        """Setup initial game screen with movement and speed selection."""
        running = True
        selected_movement = "Both"
        selected_speed = "Medium"

        button_size = (150, 50)
        center_x = self.config.width // 2
        movement_y = self.config.height // 2 - 100
        speed_y = self.config.height // 2 + 50
        start_y = self.config.height // 2 + 150

        while running:
            self.screen.fill(self.config.colors["WHITE"])

            # Draw title
            title = self.font.render(
                "Select Movement and Speed", True, self.config.colors["BLACK"]
            )
            self.screen.blit(title, (center_x - title.get_width() // 2, 50))

            # Create movement buttons
            movement_buttons = {
                "Both": self._create_button(
                    "Both", (center_x - 225, movement_y), button_size, selected_movement == "Both"
                ),
                "Horizontal": self._create_button(
                    "Horizontal",
                    (center_x - 75, movement_y),
                    button_size,
                    selected_movement == "Horizontal",
                ),
                "Vertical": self._create_button(
                    "Vertical",
                    (center_x + 75, movement_y),
                    button_size,
                    selected_movement == "Vertical",
                ),
            }

            # Create speed buttons
            speed_buttons = {
                "Slow": self._create_button(
                    "Slow", (center_x - 225, speed_y), button_size, selected_speed == "Slow"
                ),
                "Medium": self._create_button(
                    "Medium", (center_x - 75, speed_y), button_size, selected_speed == "Medium"
                ),
                "Fast": self._create_button(
                    "Fast", (center_x + 75, speed_y), button_size, selected_speed == "Fast"
                ),
            }

            # Create start button
            start_button = self._create_button(
                "Start", (center_x - button_size[0] // 2, start_y), button_size, False
            )

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos

                    # Check movement buttons
                    for movement, rect in movement_buttons.items():
                        if rect.collidepoint(pos):
                            selected_movement = movement

                    # Check speed buttons
                    for speed, rect in speed_buttons.items():
                        if rect.collidepoint(pos):
                            selected_speed = speed

                    # Check start button
                    if start_button.collidepoint(pos):
                        self.movement_type = selected_movement
                        self.speed_choice = selected_speed
                        running = False

    def setup_recording(self) -> Tuple[cv2.VideoCapture, cv2.VideoWriter, csv.writer]:
        """Setup video capture and CSV recording."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if not cap.isOpened():
            raise RuntimeError("Unable to access webcam.")

        # Set webcam resolution to 720p
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Setup video writer with MP4 format and H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = (
            self.cwd
            / "output"
            / f"GameVideo_{self.movement_type}_{self.speed_choice}_{int(time.time())}.mp4"
        )
        video_path.parent.mkdir(exist_ok=True)
        out = cv2.VideoWriter(
            str(video_path), fourcc, 30.0, (1280, 720)
        )  # 30 FPS, 720p resolution

        # Setup CSV writer
        csv_path = (
            self.cwd
            / "output"
            / f"Game_{self.movement_type}_{self.speed_choice}_{int(time.time())}.csv"
        )
        csv_path.parent.mkdir(exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["Frame", "Time", "X", "Y", "Speed_X", "Speed_Y"])

        return cap, out, writer

    def game(self):
        """Main game loop."""
        obj_pos = {"x": 400, "y": 300}
        obj_size = {"width": 20, "height": 20}
        speed = {
            "x": self.config.speed_values[self.speed_choice],
            "y": self.config.speed_values[self.speed_choice],
        }

        cap, video_writer, csv_writer = self.setup_recording()

        running = True
        clock = pygame.time.Clock()
        start_time = time.time()
        frame_count = 0

        try:
            while running:
                elapsed_time = time.time() - start_time
                if elapsed_time >= self.config.game_duration:
                    print("Game session ended.")
                    break

                self._handle_events()
                self._update_object_position(obj_pos, obj_size, speed)
                self._draw_frame(obj_pos, obj_size, elapsed_time)
                if self._record_frame(cap, video_writer):
                    frame_count += 1
                    self._log_data(csv_writer, frame_count, elapsed_time, obj_pos, speed)

                pygame.display.flip()
                clock.tick(60)

        finally:
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            pygame.quit()

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    self.movement_type = "Horizontal"
                elif event.key == pygame.K_v:
                    self.movement_type = "Vertical"
                elif event.key == pygame.K_b:
                    self.movement_type = "Both"

    def _update_object_position(
        self, obj_pos: Dict[str, int], obj_size: Dict[str, int], speed: Dict[str, int]
    ):
        """Update object position based on movement type."""
        if self.movement_type in ("Both", "Horizontal"):
            obj_pos["x"] += speed["x"]
            if obj_pos["x"] <= 0 or obj_pos["x"] >= self.config.width - obj_size["width"]:
                speed["x"] *= -1
                obj_pos["y"] += random.randint(10, 70)  # nosec

        if self.movement_type in ("Both", "Vertical"):
            obj_pos["y"] += speed["y"]
            if obj_pos["y"] <= 0 or obj_pos["y"] >= self.config.height - obj_size["height"]:
                speed["y"] *= -1
                obj_pos["x"] += random.randint(10, 70)  # nosec

    def _draw_frame(self, obj_pos: Dict[str, int], obj_size: Dict[str, int], elapsed_time: float):
        """Draw game frame."""
        self.screen.fill(self.config.colors["WHITE"])
        pygame.draw.rect(
            self.screen,
            self.config.colors["RED"],
            (obj_pos["x"], obj_pos["y"], obj_size["width"], obj_size["height"]),
        )

        time_remaining = self.config.game_duration - elapsed_time
        info_text = self.font.render(
            f"Movement: {self.movement_type} | Speed: {self.speed_choice} | Time Left: {int(time_remaining)}s",
            True,
            self.config.colors["BLACK"],
        )
        self.screen.blit(info_text, (10, 10))

    def _record_frame(self, cap: cv2.VideoCapture, video_writer: cv2.VideoWriter) -> bool:
        """Record webcam frame and return whether frame was successfully recorded."""
        ret, frame = cap.read()
        if ret:
            # Ensure frame is in 720p resolution
            frame = cv2.resize(frame, (1280, 720))
            frame = cv2.flip(frame, 1)

            video_writer.write(frame)
            # Display in a smaller window for convenience
            display_frame = cv2.resize(frame, (640, 360))
            cv2.imshow("Webcam", display_frame)
        return ret

    def _log_data(
        self,
        csv_writer: csv.writer,
        frame_number: int,
        elapsed_time: float,
        obj_pos: Dict[str, int],
        speed: Dict[str, int],
    ):
        """Log game data to CSV."""
        csv_writer.writerow(
            [
                frame_number,
                round(elapsed_time, 2),
                obj_pos["x"],
                obj_pos["y"],
                speed["x"],
                speed["y"],
            ]
        )


def main():
    game = MovementTracker(spec_file="calibration/screen_spec.json")
    game.setup_screen()
    game.game()


if __name__ == "__main__":
    main()
