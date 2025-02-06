import os
import pathlib
import time

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from calibration import ArucoDetector, GazeMapper, read_screen_specs
from gen_insights import generate_insight_plots
from inference import load_model, select_device
from l2cs import GazeEstimator
from processing import process_video

st.set_page_config(page_title="Gaze Estimation", page_icon="👀")

# Add custom CSS
st.markdown(
    """
<style>

    .centered-title {
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_model():
    CWD = pathlib.Path.cwd()
    model_path = CWD / "models" / "L2CSNet_gaze360.pkl"
    screen_spec_path = CWD / "calibration" / "screen_spec.json"

    device = select_device()
    model = load_model(model_path, device)

    gaze_estimator = GazeEstimator(
        model=model,
        device=device,
        include_detector=True,
        confidence_threshold=0.9,
        include_head_pose=False,
    )

    screen = read_screen_specs(screen_spec_path)
    ar_uco_detector = ArucoDetector(
        marker_real_width=4.0, calibration_file="calibration/calibration_results.json"
    )
    mapper = GazeMapper(screen)

    return gaze_estimator, mapper, ar_uco_detector


def save_uploaded_file(uploaded_file):
    CWD = pathlib.Path.cwd()
    # Create temp directory if it doesn't exist
    temp_dir = CWD / "temp"
    temp_dir.mkdir(exist_ok=True)

    save_path = temp_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def main():
    st.markdown(
        "<h1 class='centered-title'>Gaze Estimation Analysis 👀</h1>", unsafe_allow_html=True
    )
    st.markdown("---")

    # Initialize session state for model
    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = False

    # Create two tabs for different functionalities
    video_tab, analysis_tab = st.tabs(["Video Processing", "CSV Analysis"])

    # Video Processing Tab
    with video_tab:
        st.subheader("Upload and Process Video")
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi"], key="video")

        if uploaded_video:
            st.video(uploaded_video)

            # Model initialization moved inside video processing
            if not st.session_state.model_initialized:
                st.warning("Model needs to be initialized before processing video.")
                if st.button("Initialize Model"):
                    with st.spinner("Initializing model..."):
                        (
                            st.session_state.gaze_estimator,
                            st.session_state.mapper,
                            st.session_state.ar_detector,
                        ) = initialize_model()
                        st.session_state.model_initialized = True
                    st.success("Model initialized successfully!")

            if st.session_state.model_initialized and st.button("Process Video"):
                with st.spinner("Processing video..."):
                    # Save uploaded video
                    video_path = save_uploaded_file(uploaded_video)

                    # Generate output path
                    CWD = pathlib.Path.cwd()
                    csv_file_path = (
                        CWD
                        / "output"
                        / "processed_csv"
                        / f"processed_coordinates_{str(video_path).split('_')[-1][:-4]}.csv"
                    )

                    try:
                        # Process video
                        process_video(
                            video_path,
                            st.session_state.gaze_estimator,
                            draw_head_pose=False,
                            draw_gaze=True,
                            output_mode=None,
                            mapper=st.session_state.mapper,
                            ar_detector=st.session_state.ar_detector,
                            marker_id=42,
                            csv_file_path=csv_file_path,
                        )
                        st.success("Video processing completed!")

                        # Clean up temp file
                        if os.path.exists(video_path):
                            os.remove(video_path)

                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                        if os.path.exists(video_path):
                            os.remove(video_path)

    # CSV Analysis Tab
    with analysis_tab:
        st.subheader("Select CSV Files for Analysis")
        col1, col2 = st.columns(2)

        with col1:
            game_file = st.file_uploader("Upload game coordinates CSV", type=["csv"], key="game")

        with col2:
            processed_file = st.file_uploader(
                "Upload processed coordinates CSV", type=["csv"], key="processed"
            )

        if game_file and processed_file:
            if st.button("Generate Insights"):
                with st.spinner("Generating insights..."):
                    try:
                        # Save uploaded CSVs
                        game_path = save_uploaded_file(game_file)
                        processed_path = save_uploaded_file(processed_file)

                        # Generate and display plots
                        figures, metrics = generate_insight_plots(
                            str(game_path), str(processed_path)
                        )

                        # Display metrics
                        st.markdown("---")
                        st.subheader("Analysis Metrics")
                        metrics_df = pd.DataFrame([metrics["metrics"]]).T
                        metrics_df.columns = ["Value"]
                        st.dataframe(metrics_df)
                        # Show metrics file location
                        st.info("Metrics saved to: output/metrics")

                        # Display plots
                        st.markdown("---")
                        st.subheader("Analysis Plots")
                        st.pyplot(figures)
                        plt.close(figures)
                        st.info("Metrics saved to: output/plot")

                        # Clean up temp files
                        for path in [game_path, processed_path]:
                            if os.path.exists(path):
                                os.remove(path)

                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
                        # Clean up temp files in case of error
                        for path in [game_path, processed_path]:
                            if os.path.exists(path):
                                os.remove(path)


if __name__ == "__main__":
    main()
