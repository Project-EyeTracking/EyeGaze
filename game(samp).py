import csv
import json
import pathlib
import platform
import random
import time
import math

import cv2
import pygame

CWD = pathlib.Path.cwd()
pygame.init()

# Get screen dimensions from screen_spec.json
screen_spec_path = CWD / "calibration" / "screen_spec.json"
with open(screen_spec_path) as file:
    data = json.load(file)

# Adjust for some UI element space
WIDTH = data.get("width_pixels")
HEIGHT = data.get("height_pixels") - 100

# Calculate the 25% and 75% boundaries for both width and height
LEFT_BOUNDARY = WIDTH * 0.25
RIGHT_BOUNDARY = WIDTH * 0.75
TOP_BOUNDARY = HEIGHT * 0.25
BOTTOM_BOUNDARY = HEIGHT * 0.75

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive Icon Tracker")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)  # fallback color if icon not loaded
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Font
font = pygame.font.Font(None, 36)

# Variables to store user choices
movement_type = "Both"   # Default movement type
speed_choice = "Medium"    # Default speed
speed_values = {"Slow": 2, "Medium": 4, "Fast": 8}  # Speed mapping (pixels per frame)
game_duration = 50

# Amount to move vertically in horizontal mode (each drop is 100 pixels)
VERTICAL_STEP = 100
# Amount to move horizontally in vertical mode (each shift is 100 pixels)
HORIZONTAL_STEP = 100


def _show_wait_message():
    """Display a fun 'Please wait...' message before starting the game."""
    screen.fill(WHITE)
    wait_message_lines = [
        "Please wait...",
        "Our webcam is getting camera-ready!",
        "Watch the icon as it moves—can you track it?",
    ]
    for i, line in enumerate(wait_message_lines):
        text_surface = font.render(line, True, BLACK)
        screen.blit(
            text_surface,
            (
                WIDTH // 2 - text_surface.get_width() // 2,
                HEIGHT // 2 - text_surface.get_height() // 2 + i * 30,
            ),
        )
    pygame.display.flip()
    pygame.time.delay(3000)


def setup_screen():
    """Set up the game configuration screen."""
    global movement_type, speed_choice

    running = True
    selected_movement = "Both"  # Default selection
    selected_speed = "Medium"   # Default speed selection

    # Calculate dynamic positions
    button_width, button_height = 150, 50
    center_x = WIDTH // 2
    center_y = HEIGHT // 2

    # Arrange movement buttons in one row:
    # Positions: center_x - 225, center_x - 75, center_x + 75, center_x + 225
    movement_y_start = center_y - 100
    speed_y_start = center_y + 50
    start_button_y = center_y + 150

    while running:
        screen.fill(WHITE)
        title = font.render("Select Movement and Speed", True, BLACK)
        screen.blit(title, (center_x - title.get_width() // 2, 50))

        # Movement type buttons
        pygame.draw.rect(
            screen,
            GRAY if selected_movement == "Both" else WHITE,
            (center_x - 225, movement_y_start, button_width, button_height),
        )
        both_text = font.render("Both", True, BLACK)
        screen.blit(both_text, (center_x - 225 + 20, movement_y_start + 10))

        pygame.draw.rect(
            screen,
            GRAY if selected_movement == "Horizontal" else WHITE,
            (center_x - 75, movement_y_start, button_width, button_height),
        )
        horizontal_text = font.render("Horizontal", True, BLACK)
        screen.blit(horizontal_text, (center_x - 75 + 20, movement_y_start + 10))

        pygame.draw.rect(
            screen,
            GRAY if selected_movement == "Vertical" else WHITE,
            (center_x + 75, movement_y_start, button_width, button_height),
        )
        vertical_text = font.render("Vertical", True, BLACK)
        screen.blit(vertical_text, (center_x + 75 + 20, movement_y_start + 10))

        pygame.draw.rect(
            screen,
            GRAY if selected_movement == "Diagonal" else WHITE,
            (center_x + 225, movement_y_start, button_width, button_height),
        )
        diagonal_text = font.render("Diagonal", True, BLACK)
        screen.blit(diagonal_text, (center_x + 225 + 20, movement_y_start + 10))

        # Speed option buttons
        pygame.draw.rect(
            screen,
            GRAY if selected_speed == "Slow" else WHITE,
            (center_x - 225, speed_y_start, button_width, button_height),
        )
        slow_text = font.render("Slow", True, BLACK)
        screen.blit(slow_text, (center_x - 225 + 20, speed_y_start + 10))

        pygame.draw.rect(
            screen,
            GRAY if selected_speed == "Medium" else WHITE,
            (center_x - 75, speed_y_start, button_width, button_height),
        )
        medium_text = font.render("Medium", True, BLACK)
        screen.blit(medium_text, (center_x - 75 + 20, speed_y_start + 10))

        pygame.draw.rect(
            screen,
            GRAY if selected_speed == "Fast" else WHITE,
            (center_x + 75, speed_y_start, button_width, button_height),
        )
        fast_text = font.render("Fast", True, BLACK)
        screen.blit(fast_text, (center_x + 75 + 20, speed_y_start + 10))

        # Start button
        pygame.draw.rect(
            screen,
            GRAY,
            (center_x - button_width // 2, start_button_y, button_width, button_height),
        )
        start_text = font.render("Start", True, BLACK)
        screen.blit(start_text, (center_x - start_text.get_width() // 2, start_button_y + 10))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # Movement type selection
                if (center_x - 225 <= x <= center_x - 225 + button_width and
                    movement_y_start <= y <= movement_y_start + button_height):
                    selected_movement = "Both"
                elif (center_x - 75 <= x <= center_x - 75 + button_width and
                      movement_y_start <= y <= movement_y_start + button_height):
                    selected_movement = "Horizontal"
                elif (center_x + 75 <= x <= center_x + 75 + button_width and
                      movement_y_start <= y <= movement_y_start + button_height):
                    selected_movement = "Vertical"
                elif (center_x + 225 <= x <= center_x + 225 + button_width and
                      movement_y_start <= y <= movement_y_start + button_height):
                    selected_movement = "Diagonal"

                # Speed selection
                if (center_x - 225 <= x <= center_x - 225 + button_width and
                    speed_y_start <= y <= speed_y_start + button_height):
                    selected_speed = "Slow"
                elif (center_x - 75 <= x <= center_x - 75 + button_width and
                      speed_y_start <= y <= speed_y_start + button_height):
                    selected_speed = "Medium"
                elif (center_x + 75 <= x <= center_x + 75 + button_width and
                      speed_y_start <= y <= speed_y_start + button_height):
                    selected_speed = "Fast"

                # Start button click
                if (center_x - button_width // 2 <= x <= center_x + button_width // 2 and
                    start_button_y <= y <= start_button_y + button_height):
                    movement_type = selected_movement
                    speed_choice = selected_speed
                    running = False


def game(camera_id=0, frame_width=640, frame_height=480):
    """Main game loop with interactive icon movement and video capture."""
    global movement_type
    _show_wait_message()

    # Initialize icon dimensions
    obj_width, obj_height = 100, 100  # increased size for visibility

    # Set starting position based on movement type.
    # For Diagonal mode, we want to start at the top right corner.
    if movement_type == "Diagonal":
        # Parameterize the diagonal path using t: 0 means top right, 1 means bottom left.
        diagonal_t = 0.0
        diagonal_direction = 1  # 1 means moving from top right to bottom left; -1 means reverse.
        obj_x = (WIDTH - obj_width) * (1 - diagonal_t)  # = WIDTH - obj_width when t=0
        obj_y = (HEIGHT - obj_height) * diagonal_t         # = 0 when t=0
    else:
        # Otherwise, start at the center.
        obj_x = (WIDTH - obj_width) // 2
        obj_y = (HEIGHT - obj_height) // 2
        # For non-diagonal modes, these diagonal variables are not used.
        diagonal_t = None
        diagonal_direction = None

    speed_x = speed_values[speed_choice]
    speed_y = speed_values[speed_choice]

    # Mode-specific movement variables for other modes.
    if movement_type == "Horizontal":
        moving_right = False  # start moving left
        vertical_drop_remaining = 0
    if movement_type == "Vertical":
        moving_down = True
        horizontal_shift_remaining = 0
        shift_right = True
    # "Both" mode uses its own logic.

    # Load the interactive icon image
    try:
        icon_image = pygame.image.load("icon.png").convert_alpha()
        icon_image = pygame.transform.scale(icon_image, (obj_width, obj_height))
    except Exception as e:
        print("Error loading icon.png, falling back to red rectangle.")
        icon_image = None

    # Platform-specific camera setup
    system_platform = platform.system()
    if system_platform == "Darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif system_platform == "Linux":
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_DSHOW

    cap = cv2.VideoCapture(camera_id, backend)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    fps = 30.0
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    time_now = int(time.time())
    game_video_file_path = CWD / "output" / f"game_video_{time_now}.avi"
    game_video_file_path.parent.mkdir(exist_ok=True)
    out = cv2.VideoWriter(str(game_video_file_path), fourcc, fps, (frame_width, frame_height))

    # CSV setup
    csv_path = CWD / "output" / f"game_coordinates_{time_now}.csv"
    csv_path.parent.mkdir(exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Frame", "Time", "GameX", "GameY", "Speed_X", "Speed_Y"])

    # Main game loop
    running = True
    clock = pygame.time.Clock()
    start_time = time.time()
    frame_counter = 0

    while running:
        elapsed_time = time.time() - start_time
        if elapsed_time >= game_duration:
            print("Game session ended.")
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Allow mode changes via keyboard if desired
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    movement_type = "Horizontal"
                    moving_right = False
                    vertical_drop_remaining = 0
                elif event.key == pygame.K_v:
                    movement_type = "Vertical"
                    moving_down = True
                    horizontal_shift_remaining = 0
                    shift_right = True
                elif event.key == pygame.K_b:
                    movement_type = "Both"
                elif event.key == pygame.K_d:
                    movement_type = "Diagonal"
                    # Reset diagonal mode variables:
                    diagonal_t = 0.0
                    diagonal_direction = 1
                    obj_x = (WIDTH - obj_width) * (1 - diagonal_t)
                    obj_y = (HEIGHT - obj_height) * diagonal_t

            # Check for icon clicks to add extra interactivity
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                if obj_x <= mouse_x <= obj_x + obj_width and obj_y <= mouse_y <= obj_y + obj_height:
                    print("You clicked the icon!")  # Extend with sound or visual feedback as desired.

        screen.fill(WHITE)

        # --- Movement Logic ---
        if movement_type == "Horizontal":
            # Horizontal zigzag: move until hitting a boundary,
            # then drop vertically 100 pixels gradually and reverse horizontal direction.
            if vertical_drop_remaining > 0:
                drop_step = min(speed_y, vertical_drop_remaining)
                obj_y += drop_step
                vertical_drop_remaining -= drop_step
                if obj_y >= HEIGHT - obj_height:
                    obj_y = HEIGHT - obj_height
                    vertical_drop_remaining = 0
            else:
                if moving_right:
                    obj_x += speed_x
                    if obj_x >= WIDTH - obj_width:
                        obj_x = WIDTH - obj_width
                        vertical_drop_remaining = VERTICAL_STEP
                        moving_right = False
                else:
                    obj_x -= speed_x
                    if obj_x <= 0:
                        obj_x = 0
                        vertical_drop_remaining = VERTICAL_STEP
                        moving_right = True

        elif movement_type == "Vertical":
            # Vertical zigzag: move vertically until a boundary,
            # then slide horizontally 100 pixels gradually and reverse vertical direction.
            if horizontal_shift_remaining > 0:
                shift_step = min(speed_x, horizontal_shift_remaining)
                if shift_right:
                    obj_x += shift_step
                else:
                    obj_x -= shift_step
                horizontal_shift_remaining -= shift_step
                if obj_x >= WIDTH - obj_width:
                    obj_x = WIDTH - obj_width
                    horizontal_shift_remaining = 0
                elif obj_x <= 0:
                    obj_x = 0
                    horizontal_shift_remaining = 0
            else:
                if moving_down:
                    obj_y += speed_y
                    if obj_y >= HEIGHT - obj_height:
                        obj_y = HEIGHT - obj_height
                        horizontal_shift_remaining = HORIZONTAL_STEP
                        moving_down = False
                        shift_right = not shift_right
                else:
                    obj_y -= speed_y
                    if obj_y <= 0:
                        obj_y = 0
                        horizontal_shift_remaining = HORIZONTAL_STEP
                        moving_down = True
                        shift_right = not shift_right

        elif movement_type == "Diagonal":
            # Diagonal mode: move back and forth along the same straight line
            # from top right (t=0) to bottom left (t=1) and back.
            L = math.sqrt((WIDTH - obj_width) ** 2 + (HEIGHT - obj_height) ** 2)
            # Compute dt so that the object moves at speed (pixels per frame)
            dt = speed_values[speed_choice] / L
            # Update parameter t based on current direction.
            diagonal_t += dt * diagonal_direction
            # Clamp and reverse direction if endpoints reached.
            if diagonal_t >= 1:
                diagonal_t = 1
                diagonal_direction = -1
            elif diagonal_t <= 0:
                diagonal_t = 0
                diagonal_direction = 1
            # Interpolate position along the line from top right to bottom left.
            # Top right corresponds to t=0: (WIDTH - obj_width, 0)
            # Bottom left corresponds to t=1: (0, HEIGHT - obj_height)
            obj_x = (WIDTH - obj_width) * (1 - diagonal_t)
            obj_y = (HEIGHT - obj_height) * diagonal_t

        elif movement_type == "Both":
            # Both mode: simultaneous horizontal and vertical movement with random shifts.
            obj_x += speed_x
            if obj_x <= 0 or obj_x >= WIDTH - obj_width:
                speed_x *= -1
                new_y = obj_y + random.randint(-100, 100)  # nosec
                obj_y = max(0, min(HEIGHT - obj_height, new_y))
            obj_y += speed_y
            if obj_y <= 0 or obj_y >= HEIGHT - obj_height:
                speed_y *= -1
                new_x = obj_x + random.randint(-100, 100)  # nosec
                obj_x = max(0, min(WIDTH - obj_width, new_x))

        # Draw the interactive icon (or fallback to rectangle)
        if icon_image:
            screen.blit(icon_image, (obj_x, obj_y))
        else:
            pygame.draw.rect(screen, RED, (obj_x, obj_y, obj_width, obj_height))

        # Display movement info and time remaining
        time_remaining = game_duration - elapsed_time
        info_text = font.render(
            f"Movement: {movement_type} | Speed: {speed_choice} | Time Left: {int(time_remaining)}s",
            True,
            BLACK,
        )
        screen.blit(info_text, (10, 10))

        # Capture webcam frame and write to video
        ret, frame = cap.read()
        if ret:
            out.write(frame)

        # Log current frame data
        frame_counter += 1
        writer.writerow([frame_counter, round(elapsed_time, 2), obj_x, obj_y, speed_x, speed_y])

        pygame.display.flip()
        clock.tick(fps)

    # Cleanup resources
    csv_file.close()
    print(f"CSV file saved at: {csv_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pygame.quit()
    return csv_path, game_video_file_path


if __name__ == "__main__":
    setup_screen()
    game()
