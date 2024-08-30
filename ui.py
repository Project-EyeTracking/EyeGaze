import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from inference import load_model, process_image, select_device
from l2cs import GazeEstimator, render

st.set_page_config(page_title="Gaze Estimation", page_icon="👀")

# Add custom CSS
st.markdown(
    """
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stRadio>div {
        display: flex;
        justify-content: space-around;
    }
    .centered-title {
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


def process_webcam(cam_id, gaze_estimator, draw_head_pose=False, draw_gaze=True):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        st.error(f"Error: Could not open webcam with ID {cam_id}")
        return

    stframe = st.empty()
    stop_button = st.button("Stop Processing")

    while True and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = gaze_estimator.predict(frame)

        if results.pitch is not None and len(results.pitch) > 0:
            # Visualize output
            if draw_gaze:
                frame = render(frame, results, draw_landmarks=True, draw_bboxes=True)

            if draw_head_pose:
                for bbox, head_orientation in zip(results.bboxes, results.head_orientations):
                    gaze_estimator.head_pose_estimator.plot_pose_cube(
                        frame, bbox, **head_orientation
                    )
            stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()


def process_video(video_path, gaze_estimator, draw_head_pose=False, draw_gaze=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
        return

    stframe = st.empty()
    stop_button = st.button("Stop Processing")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture frame")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = gaze_estimator.predict(frame)

        if results.pitch is not None and len(results.pitch) > 0:
            # Visualize output
            if draw_gaze:
                frame = render(frame, results, draw_landmarks=False, draw_bboxes=True)

            if draw_head_pose:
                for bbox, head_orientation in zip(results.bboxes, results.head_orientations):
                    gaze_estimator.head_pose_estimator.plot_pose_cube(
                        frame, bbox, **head_orientation
                    )

            stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()


def main():

    st.markdown("<h1 class='centered-title'>Gaze Estimation 👀</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # Load model
    device = select_device()
    model_path = "models/L2CSNet_gaze360.pkl"  # Update this path if necessary
    model = load_model(model_path, device)
    gaze_estimator = GazeEstimator(
        model=model,
        device=device,
        include_detector=True,
        confidence_threshold=0.9,
        include_head_pose=False,
    )

    # Input selection
    st.subheader("Select Input Type")
    input_type = st.radio(
        label="Input Type", options=["Image", "Video", "Webcam"], label_visibility="collapsed"
    )
    st.markdown("---")

    if input_type == "Image":
        st.subheader("Image Upload")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Estimate Gaze"):
                with st.spinner("Estimating gaze..."):
                    frame, results = process_image(
                        image,
                        gaze_estimator,
                        draw_head_pose=gaze_estimator.include_head_pose,
                        draw_gaze=True,
                    )
                    st.image(frame, caption="Gaze Estimation Result", use_column_width=True)
                    for i, (pitch, yaw) in enumerate(zip(results.pitch, results.yaw)):
                        st.write(
                            f"Face {i+1} - Predicted gaze angles (pitch, yaw): {pitch:.2f}, {yaw:.2f}"
                        )

    elif input_type == "Video":
        st.subheader("Video Upload")
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            st.video(uploaded_video)
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    # Save the uploaded video to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                        tmp_file.write(uploaded_video.read())
                        process_video(
                            tmp_file.name,
                            gaze_estimator,
                            draw_head_pose=gaze_estimator.include_head_pose,
                            draw_gaze=True,
                        )

    elif input_type == "Webcam":
        st.subheader("Webcam Input")
        if st.button("Start Webcam"):
            with st.spinner("Starting webcam..."):
                process_webcam(
                    0,
                    gaze_estimator,
                    draw_head_pose=gaze_estimator.include_head_pose,
                    draw_gaze=True,
                )


if __name__ == "__main__":
    main()
