# 🎾 Tennis Match Detection and Tracking 🏃‍♂️

This Python project utilizes **YOLO** object detection models and **MediaPipe Pose** to detect and track objects in a tennis match, specifically detecting a **tennis ball**, a **tennis court**, and tracking the **player's movement**. The system calculates the ball's and the player's speed in real-time and overlays the information on the video.

## 🚀 Requirements

To run this project, you need the following Python libraries:

- `opencv-python` 📷
- `numpy` ➗
- `ultralytics` (for YOLO model) 🤖
- `mediapipe` 🖐️
- `sort` (for object tracking) 🧑‍💻

You can install them using `pip`:

```bash
pip install opencv-python numpy ultralytics mediapipe sort
