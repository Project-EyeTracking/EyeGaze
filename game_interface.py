from openpyxl import Workbook
import pygame
import cv2
import csv
import time
import json
import pathlib
import random

CWD = pathlib.Path.cwd()
pygame.init()

# Get screen dimensions from screen_spec.json dynamically
screen_spec_path = CWD / 'calibration' / 'screen_spec.json'
with open(screen_spec_path, 'r') as file:
    data = json.load(file)

WIDTH = data.get('width_pixels')
HEIGHT = data.get('height_pixels') - 100  # Adjust for some UI element space
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Object Movement Tracker")

# Colors
BLACK = (255, 255, 255)  
RED = (255, 0, 0)
WHITE = (0, 0, 0)
GRAY = (200, 200, 200)

# Font
font = pygame.font.Font(None, 36)

# Variables to store user choices
movement_type = "Both"  # Default movement type
speed_choice = "Medium"  # Default speed
speed_values = {"Slow": 2, "Medium": 6, "Fast": 12}  # Speed mapping
game_duration = 20

def _show_wait_message():
    """Display a fun 'Please wait...' message before starting the game."""
    screen.fill(WHITE)
    
    wait_message_lines = [
        "Please wait...",
        "Our webcam needs a moment to get camera-ready!",
        "It's finding its best angle and practicing its selfie smile!"
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
    pygame.time.delay(3000)  # Wait for 3 seconds

def setup_screen():
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
    speed_y_start = center_y + 50     # Starting y-position for speed buttons
    start_button_y = center_y + 150   # y-position for the "Start" button

    while running:
        screen.fill(WHITE)
        title = font.render("Select Movement and Speed", True, BLACK)
        screen.blit(title, (center_x - title.get_width() // 2, 50))

        # Buttons for movement type
        pygame.draw.rect(screen, GRAY if selected_movement == "Both" else WHITE, 
                         (center_x - 225, movement_y_start, button_width, button_height))
        both_text = font.render("Both", True, BLACK)
        screen.blit(both_text, (center_x - 225 + 20, movement_y_start + 10))

        pygame.draw.rect(screen, GRAY if selected_movement == "Horizontal" else WHITE, 
                         (center_x - 75, movement_y_start, button_width, button_height))
        horizontal_text = font.render("Horizontal", True, BLACK)
        screen.blit(horizontal_text, (center_x - 75 + 20, movement_y_start + 10))

        pygame.draw.rect(screen, GRAY if selected_movement == "Vertical" else WHITE, 
                         (center_x + 75, movement_y_start, button_width, button_height))
        vertical_text = font.render("Vertical", True, BLACK)
        screen.blit(vertical_text, (center_x + 75 + 20, movement_y_start + 10))

        # Buttons for speed options
        pygame.draw.rect(screen, GRAY if selected_speed == "Slow" else WHITE, 
                         (center_x - 225, speed_y_start, button_width, button_height))
        slow_text = font.render("Slow", True, BLACK)
        screen.blit(slow_text, (center_x - 225 + 20, speed_y_start + 10))

        pygame.draw.rect(screen, GRAY if selected_speed == "Medium" else WHITE, 
                         (center_x - 75, speed_y_start, button_width, button_height))
        medium_text = font.render("Medium", True, BLACK)
        screen.blit(medium_text, (center_x - 75 + 20, speed_y_start + 10))

        pygame.draw.rect(screen, GRAY if selected_speed == "Fast" else WHITE, 
                         (center_x + 75, speed_y_start, button_width, button_height))
        fast_text = font.render("Fast", True, BLACK)
        screen.blit(fast_text, (center_x + 75 + 20, speed_y_start + 10))

        # Start button
        pygame.draw.rect(screen, GRAY, (center_x - button_width // 2, start_button_y, button_width, button_height))
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
                if center_x - 225 <= x <= center_x - 225 + button_width and movement_y_start <= y <= movement_y_start + button_height:
                    selected_movement = "Both"
                elif center_x - 75 <= x <= center_x - 75 + button_width and movement_y_start <= y <= movement_y_start + button_height:
                    selected_movement = "Horizontal"
                elif center_x + 75 <= x <= center_x + 75 + button_width and movement_y_start <= y <= movement_y_start + button_height:
                    selected_movement = "Vertical"

                # Check which speed is selected
                if center_x - 225 <= x <= center_x - 225 + button_width and speed_y_start <= y <= speed_y_start + button_height:
                    selected_speed = "Slow"
                elif center_x - 75 <= x <= center_x - 75 + button_width and speed_y_start <= y <= speed_y_start + button_height:
                    selected_speed = "Medium"
                elif center_x + 75 <= x <= center_x + 75 + button_width and speed_y_start <= y <= speed_y_start + button_height:
                    selected_speed = "Fast"

                # Check if "Start" is clicked
                if center_x - button_width // 2 <= x <= center_x + button_width // 2 and start_button_y <= y <= start_button_y + button_height:
                    movement_type = selected_movement
                    speed_choice = selected_speed
                    running = False

def game():
    global movement_type
    #_show_wait_message()  # Show wait message before starting the game

    obj_x, obj_y = 400, 300
    obj_width, obj_height = 30, 30
    speed_x = speed_values[speed_choice]
    speed_y = speed_values[speed_choice]

    # OpenCV setup for webcam recording
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    game_video_csv_file_path = CWD / "output" / f'GameVideo_{movement_type}_{speed_choice}{int(time.time())}.avi'
    game_video_csv_file_path.parent.mkdir(exist_ok=True)
    fps = 30.0
    frame_size = (640, 480)
    out = cv2.VideoWriter(str(game_video_csv_file_path), fourcc, fps, frame_size)

    # Excel setup
    game_excel_file_path = CWD / "output" / f'Game_{movement_type}_{speed_choice}_{int(time.time())}.xlsx'
    game_excel_file_path.parent.mkdir(exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Game Data"
    sheet.append(["Frame", "Time", "GameX", "GameY", "SpeedX", "SpeedY"])  # Add headers

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

            # Allow user to change movement type during gameplay
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    movement_type = "Horizontal"
                elif event.key == pygame.K_v:
                    movement_type = "Vertical"
                elif event.key == pygame.K_b:
                    movement_type = "Both"

        # Clear screen
        screen.fill(WHITE)

        # Move object based on selected movement type
        if movement_type in ("Both", "Horizontal"):
            obj_x += speed_x
            if obj_x <= 0 or obj_x >= WIDTH - obj_width:
                speed_x *= -1
                obj_y += random.randint(5, 8)

        if movement_type in ("Both", "Vertical"):
            obj_y += speed_y
            if obj_y <= 0 or obj_y >= HEIGHT - obj_height:
                speed_y *= -1
                obj_x += random.randint(5, 8) 

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
            frame = cv2.flip(frame, 1)
            out.write(frame)
            # Display smaller frame
            display_frame = cv2.resize(frame, (640, 360))
            cv2.imshow('Webcam', display_frame)

        # Log data into Excel
        frame_counter += 1
        sheet.append([frame_counter, round(elapsed_time, 2), obj_x, obj_y, speed_x, speed_y])

        # Update display
        pygame.display.flip()
        clock.tick(30)

    # Save the Excel file
    workbook.save(str(game_excel_file_path))
    print(f"Excel file saved at: {game_excel_file_path}")

    # Release OpenCV resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pygame.quit()

setup_screen()
game()