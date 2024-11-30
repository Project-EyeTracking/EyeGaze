import cv2
import numpy as np
import os

def capturepictures():
    
     # Create a directory to save images if it doesn't already exist
    output_dir = "images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    # Counter for saved images
    image_count = 0
    max_images = 12

    print("Press 'c' to capture an image, or 'q' to quit.")

    while image_count < max_images:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Display the frame
        cv2.imshow("Webcam", frame)
        
        # Wait for user input
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'c' to capture and save the image
        if key == ord('c'):
            image_path = os.path.join(output_dir, f"image_{image_count + 1}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Captured and saved: {image_path}")
            image_count += 1
        
        # Press 'q' to quit
        elif key == ord('q'):
            print("Exiting.")
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    print(f"Captured {image_count} images. Images saved in '{output_dir}' directory.")
    return output_dir

def calibrate_camera(image_dir, pattern_size):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    obj_points = []  # 3D points in real-world space
    img_points = []  # 2D points in image plane
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

    for file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners2)
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow("Detected Corners", img)
            cv2.waitKey(1000)
        else:
            print(f"Chessboard corners not found in {img_path}")

    cv2.destroyAllWindows()

    if len(obj_points) < 1:
        print("No valid data for camera calibration. Check the input images.")
        return None, None, None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

 
    error = calculate_reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs)
    print(f"Mean Reprojection Error: {error}") 
    #mean error should be closer to 0.

    if ret:
        return camera_matrix, dist_coeffs, (camera_matrix[0, 2], camera_matrix[1, 2])
    else:
        print("Camera calibration failed.")
        return None, None, None
    
# Calculate the total reprojection error
def calculate_reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        total_error += error
    mean_error = total_error / len(obj_points)
    return mean_error



def main():

    #image_dir = capturepictures()
    image_dir = r"C:\Users\anagh\EyeGaze-1\images"  # Change to your directory
    pattern_size = (4,7)  # Inner corners in the chessboard
    camera_matrix, dist_coeffs, optical_center = calibrate_camera(image_dir, pattern_size)

    if camera_matrix is not None:
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)
        print(f"Optical Center: {optical_center}")
    
   

if __name__ == "__main__":
    main()






