👞📸 Real-Time Attendance and Footwear Safety Monitoring System
A smart, real-time system designed to automate student attendance marking and PPE compliance monitoring (footwear) using face recognition and YOLOv8-based object detection from a live webcam feed.

🔍 Overview
This project combines computer vision and safety compliance for academic or industrial labs by detecting:

Student presence via face recognition (LBPH algorithm).

Footwear presence using a custom-trained YOLOv8 model.

The system automatically:

Marks attendance in a CSV file.

Detects missing safety shoes.

Saves annotated images of non-compliant individuals.

Tracks alerts and attendance with cooldown logic to prevent duplicate entries.

🧠 Tech Stack
Language: Python

Computer Vision: OpenCV, Haar Cascades

Object Detection: Ultralytics YOLOv8 (pre-trained for persons, custom-trained for shoes)

Data Storage: CSV, directory-based image storage

Hardware: Webcam (Live Feed)

📦 Features
✅ Real-time face detection & recognition for attendance
✅ YOLOv8 person + custom shoe detection
✅ Foot region extraction for targeted footwear monitoring
✅ CSV-based student roster updates (timestamped)
✅ Annotated non-compliance image logging
✅ Cooldown logic to prevent spammy re-entries
✅ Modular structure with training support for face data
