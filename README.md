# EyeGaze

### Usage

```python
# To run inference on a image
python inference.py --image_path assets/input_image.png
```

```python
# To run inference on a video file
python inference.py --video_path assets/video.mp4
```

```python
# To run inference on webcam with cam id 0
python inference.py --cam 0
```

```python
# To launch the web app, run the following command
streamlit run ui.py
```

Download the pre-trained gaze estimation models from [here](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) and Store it to *models/*.

### Results

Gaze estimation results on a sample image.
![alt text](assets/output_image_gaze.png)

Head pose estimation results on a sample image.
![alt text](assets/output_image_head_orientation.png)
