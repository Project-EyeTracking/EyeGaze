from batch_face import RetinaFace, SixDRep, draw_landmarks, load_frames_rgb, Timer
import cv2
import numpy as np

def calculate_pose_change(prev_pose, current_pose):
    if prev_pose is None or current_pose is None:
        return None
    
    yaw_change = abs(current_pose['yaw'] - prev_pose['yaw'])
    pitch_change = abs(current_pose['pitch'] - prev_pose['pitch'])
    roll_change = abs(current_pose['roll'] - prev_pose['roll'])
    
    total_change = np.sqrt(yaw_change**2 + pitch_change**2 + roll_change**2)
    return total_change

if __name__ == "__main__":
    gpu_id = -1
    batch_size = 100
    threshold = 0.95

    detector = RetinaFace(gpu_id=gpu_id)
    head_pose_estimator = SixDRep(gpu_id=gpu_id)

    video_file = '../assets/video.mp4'
    frames = load_frames_rgb(video_file, cvt_color=False)
    print(f'Loaded {len(frames)} frames')
    print('image size:', frames[0].shape)

    all_faces = detector(frames, batch_size=batch_size, return_dict=True, threshold=threshold, resize=0.5)
    head_poses = head_pose_estimator(all_faces, frames, batch_size=batch_size, update_dict=True, input_face_type='dict')
    print(f"{all_faces[0]=}")
    print(f"{head_poses[0]=}")

    out_frames = []
    prev_pose = None
    for i, (faces, frame) in enumerate(zip(all_faces, frames)):
        for face in faces:
            head_pose_estimator.plot_pose_cube(frame, face['box'], **face['head_pose'])
            
            current_pose = face['head_pose']
            pose_change = calculate_pose_change(prev_pose, current_pose)
            
            if pose_change is not None:
                cv2.putText(frame, f"Pose change: {pose_change:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            prev_pose = current_pose
        
        out_frames.append(frame)

    # Display video output
    for frame in out_frames:
        cv2.imshow('Head Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
