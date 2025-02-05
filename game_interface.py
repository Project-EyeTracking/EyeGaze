import csv
import json
import pathlib
import platform
import random
import time

import cv2
import pygame

CWD = pathlib.Path.cwd()
pygame.init()

# get screen dimensions from screen_spec.json
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
pygame.display.set_caption("Object Movement Tracker")

# Colors
WHITE = (0, 0, 0)
RED = (255, 0, 0)
BLACK = (255, 255, 255)
GRAY = (200, 200, 200)

# Font
font = pygame.font.Font(None, 36)

# Variables to store user choices
movement_type = "Both"  # Default movement type
speed_choice = "Medium"  # Default speed
speed_values = {"Slow": 2, "Medium": 4, "Fast": 8}  # Speed mapping
game_duration = 40

# Amount to move vertically in horizontal mode
VERTICAL_STEP = 100
# Amount to move horizontally in vertical mode
HORIZONTAL_STEP = 100


def _show_wait_message():
    """Display a fun 'Please wait...' message before starting the game."""
    screen.fill(WHITE)

    wait_message_lines = [
        "Please wait...",
        "Our webcam needs a moment to get camera-ready!",
        "It's finding it's best angle and practicing it's selfie smile!",
    ]

    # Display each line of the message centered on the screen
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
    selected_speed = "Medium"  # Default speed selection

    # Calculate dynamic positions
    button_width, button_height = 150, 50
    center_x = WIDTH // 2
    center_y = HEIGHT // 2

    # Vertical positions for buttons
    movement_y_start = center_y - 100  # Starting y-position for movement buttons
    speed_y_start = center_y + 50  # Starting y-position for speed buttons
    start_button_y = center_y + 150  # y-position for the "Start" button

    while running:
        screen.fill(WHITE)
        title = font.render("Select Movement and Speed", True, BLACK)
        screen.blit(title, (center_x - title.get_width() // 2, 50))

        # Buttons for movement type
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

        # Buttons for speed options
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
                # Check which button is clicked
                if (
                    center_x - 225 <= x <= center_x - 225 + button_width
                    and movement_y_start <= y <= movement_y_start + button_height
                ):
                    selected_movement = "Both"
                elif (
                    center_x - 75 <= x <= center_x - 75 + button_width
                    and movement_y_start <= y <= movement_y_start + button_height
                ):
                    selected_movement = "Horizontal"
                elif (
                    center_x + 75 <= x <= center_x + 75 + button_width
                    and movement_y_start <= y <= movement_y_start + button_height
                ):
                    selected_movement = "Vertical"

                # Check which speed is selected
                if (
                    center_x - 225 <= x <= center_x - 225 + button_width
                    and speed_y_start <= y <= speed_y_start + button_height
                ):
                    selected_speed = "Slow"
                elif (
                    center_x - 75 <= x <= center_x - 75 + button_width
                    and speed_y_start <= y <= speed_y_start + button_height
                ):
                    selected_speed = "Medium"
                elif (
                    center_x + 75 <= x <= center_x + 75 + button_width
                    and speed_y_start <= y <= speed_y_start + button_height
                ):
                    selected_speed = "Fast"

                # Check if "Start" is clicked
                if (
                    center_x - button_width // 2 <= x <= center_x + button_width // 2
                    and start_button_y <= y <= start_button_y + button_height
                ):
                    movement_type = selected_movement
                    speed_choice = selected_speed
                    running = False


def game(camera_id=0, frame_width=640, frame_height=480):
    """Main game loop with object movement and video capture."""
    global movement_type
    _show_wait_message()

    # Initialize object dimensions
    obj_width, obj_height = 30, 30

    # Initialize object position at the center of the restricted area
    obj_x = LEFT_BOUNDARY
    obj_y = TOP_BOUNDARY

    speed_x = speed_values[speed_choice]
    speed_y = speed_values[speed_choice]

    # New variables for zigzag movement
    moving_right = True
    moving_down = True

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

    # Set webcam resolution
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

    # Game loop
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

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    movement_type = "Horizontal"
                    obj_y = (TOP_BOUNDARY + BOTTOM_BOUNDARY) / 2
                elif event.key == pygame.K_v:
                    movement_type = "Vertical"
                    obj_x = (LEFT_BOUNDARY + RIGHT_BOUNDARY) / 2
                elif event.key == pygame.K_b:
                    movement_type = "Both"

        screen.fill(WHITE)

        # Modified movement logic
        if movement_type == "Horizontal":
            # Horizontal zigzag movement
            if moving_right:
                obj_x += speed_x
                if obj_x >= WIDTH - obj_width:
                    moving_right = False
                    obj_y += VERTICAL_STEP
                    if obj_y >= HEIGHT - obj_height:
                        obj_y = HEIGHT - obj_height
            else:
                obj_x -= speed_x
                if obj_x <= 0:
                    moving_right = True
                    obj_y += VERTICAL_STEP
                    if obj_y >= HEIGHT - obj_height:
                        obj_y = HEIGHT - obj_height

        elif movement_type == "Vertical":
            # Vertical zigzag movement
            if moving_down:
                obj_y += speed_y
                if obj_y >= HEIGHT - obj_height:
                    moving_down = False
                    obj_x += HORIZONTAL_STEP
                    if obj_x >= WIDTH - obj_width:
                        obj_x = WIDTH - obj_width
            else:
                obj_y -= speed_y
                if obj_y <= 0:
                    moving_down = True
                    obj_x += HORIZONTAL_STEP
                    if obj_x >= WIDTH - obj_width:
                        obj_x = WIDTH - obj_width

        elif movement_type == "Both":
            # Original "Both" movement logic
            if "Both" in (movement_type, "Horizontal"):
                obj_x += speed_x
                if obj_x <= 0 or obj_x >= WIDTH - obj_width:
                    speed_x *= -1
                    new_y = obj_y + random.randint(-100, 100)  # nosec
                    obj_y = max(0, min(HEIGHT - obj_height, new_y))

            if "Both" in (movement_type, "Vertical"):
                obj_y += speed_y
                if obj_y <= 0 or obj_y >= HEIGHT - obj_height:
                    speed_y *= -1
                    new_x = obj_x + random.randint(-100, 100)  # nosec
                    obj_x = max(0, min(WIDTH - obj_width, new_x))

        # Draw object
        pygame.draw.rect(screen, RED, (obj_x, obj_y, obj_width, obj_height))

        # Display movement and time remaining
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

        # Log data into CSV
        frame_counter += 1
        writer.writerow([frame_counter, round(elapsed_time, 2), obj_x, obj_y, speed_x, speed_y])

        # Update display
        pygame.display.flip()
        clock.tick(fps)

    # Save the CSV file
    csv_file.close()
    print(f"CSV file saved at: {csv_path}")

    # Release OpenCV resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pygame.quit()

    return csv_path, game_video_file_path


if __name__ == "__main__":
    setup_screen()
    game()
