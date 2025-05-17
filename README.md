Crowd Detection Project
This project uses YOLOv11 to detect people in a video feed and identify crowds (groups of 3 or more people standing close to each other for a specified number of frames). It draws bounding boxes around detected groups in the video feed and logs crowd events to a CSV file.
Features

Detects people using YOLOv11 (nano version).
Identifies groups of 3+ people based on proximity (distance threshold).
Tracks groups across frames to detect persistent crowds.
Draws bounding boxes around detected groups in the video feed.
Logs crowd events (frame number and person count) to a CSV file.

Requirements

Python 3.7+
Libraries listed in requirements.txt

Installation

Clone the repository:git clone https://github.com/your-username/resolute_ai_task.git
cd resolute_ai_task


Install dependencies:pip install -r requirements.txt


Download the YOLOv11 weights (yolov11n.pt) if not already present. The Ultralytics library will attempt to download it automatically on first run.

Usage

Place your input video file in the project directory (e.g., input_video.mp4).
Update the VIDEO_PATH variable in crowd_det.py to point to your video file:VIDEO_PATH = "input_video.mp4"


Run the script:python crowd_det.py


The script will:
Display the video feed with bounding boxes around groups of 3+ people.
Log detected crowds to crowd_detection_results.csv.
Press q to quit the video feed.



Parameters

DISTANCE_THRESHOLD: Pixel distance threshold for considering people "close" (default: 200).
CROWD_SIZE: Minimum number of people to form a crowd (default: 3).
FRAME_THRESHOLD: Number of consecutive frames a group must persist to be logged as a crowd (default: 5).

Output

A CSV file (crowd_detection_results.csv) with columns: Frame Number, Person Count in Crowd.
Video feed with green bounding boxes around detected groups.

Notes

Adjust DISTANCE_THRESHOLD based on your video resolution and the typical distance between people.
The video file (input_video.mp4) is not included in the repository due to size constraints. Provide your own video file for testing.

License
This project is licensed under the MIT License.
