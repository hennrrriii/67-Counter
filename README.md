A lightweight live streaming server for Raspberry Pi Camera Module 3 with real-time gesture detection and an interactive browser-based game.

The application streams MJPEG video over HTTP, detects 6-7 movements, and provides a 20-second challenge interface with live overlay tracking.




<img src="https://github.com/user-attachments/assets/a45211cc-4fb2-440f-869a-596956021dfc" width="600">




🧠 How It Works

- Video is captured using Picamera2
- Frames are encoded as MJPEG
- Skin detection is performed in HSV color space
- PCA-based contour analysis determines arm axis
- Alternating arm raises trigger a counter
- Gesture state is exposed via /gesture endpoint
- Frontend polls the API and renders overlays in real time

📦 Requirements

- Raspberry Pi (tested with Camera Module 3)
- Python 3.9+
- Raspberry Pi OS (Bookworm recommended)

Python Dependencies
- pip install numpy opencv-python

Picamera2 must be installed via apt:
- sudo apt install python3-picamera2


CURRENT RECORD: 45 in 20 Seconds
