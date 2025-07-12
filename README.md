# virtual_mouse_controller
This project is a Virtual Mouse Controller built using Computer Vision and OpenCV.

🔍 Project Overview
This Virtual Mouse Controller is a computer vision-based project that allows users to control their mouse cursor using only hand gestures, captured in real-time through a webcam. It offers a touchless, intuitive way to interact with your computer — ideal for accessibility solutions, smart environments, or gesture-based UI applications.

->  Key Features:
🖐️ Real-time Hand Tracking using MediaPipe.
🖱️ Cursor Movement based on the position of the index finger.
👆 Left Click using index and middle finger tap gesture.
✊ Drag and Drop using hand open/close gestures.
🔄 High Responsiveness and smooth tracking with low latency.

Tech Stack
Language: Python

Libraries:

OpenCV: for webcam access and image processing

MediaPipe: for detecting hand landmarks

PyAutoGUI: to simulate mouse movement and actions

--> Installation & Usage
pip install opencv-python mediapipe pyautogui
python virtual_mouse.py



