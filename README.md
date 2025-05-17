Crowd Detection Project with YOLOv11
 
A Python-based project that uses YOLOv11 to detect people in a video feed, identify crowds (groups of 3 or more people standing close together), and visualize them with bounding boxes. The project logs crowd events to a CSV file for further analysis.

🌟 Features

Person Detection: Utilizes YOLOv11 (nano version) for fast and accurate person detection.
Crowd Identification: Detects groups of 3+ people based on proximity using a customizable distance threshold.
Persistence Tracking: Tracks groups across frames to identify persistent crowds (default: 5 consecutive frames).
Visualization: Draws green bounding boxes around detected groups in the video feed with labels indicating group size.
Logging: Saves crowd events (frame number and person count) to a CSV file.

📋 Prerequisites

Python: 3.7 or higher
Git: To clone the repository
A video file (.mp4) for input
An internet connection (to download YOLOv11 weights on first run)

🛠️ Installation

Clone the Repository:
git clone https://github.com/your-username/resolute_ai_task.git
cd resolute_ai_task


Install Dependencies:
pip install -r requirements.txt


YOLOv11 Weights:

The script uses yolov11n.pt (nano version of YOLOv11). The Ultralytics library will automatically download it on the first run if not present locally.
Alternatively, you can manually download it from the Ultralytics GitHub releases and place it in the project directory.



🚀 Usage

Prepare Your Video:

Place your input video file (e.g., input_video.mp4) in the project directory.
Update the VIDEO_PATH variable in crowd_det.py to point to your video:VIDEO_PATH = "input_video.mp4"




Run the Script:
python crowd_det.py


What to Expect:

A video window will open, showing the live feed with green bounding boxes around detected groups of 3+ people.
The console will display debugging information, such as the minimum distance between people and the number of groups detected per frame.
If a crowd persists for the specified number of frames (default: 5), it will be logged to crowd_detection_results.csv.
Press q to exit the video feed.



⚙️ Configuration
You can tweak the following parameters in crowd_det.py to suit your needs:



Parameter
Description
Default Value



DISTANCE_THRESHOLD
Pixel distance threshold for "close" people
200


CROWD_SIZE
Minimum number of people to form a crowd
3


FRAME_THRESHOLD
Consecutive frames a group must persist to be logged
5


Example:

If people in your video are farther apart, increase DISTANCE_THRESHOLD to 300 or 400.
To detect crowds faster, reduce FRAME_THRESHOLD to 3.

📊 Output

Video Feed: Displays the input video with green bounding boxes around detected groups, labeled with the number of people in each group.
CSV File: crowd_detection_results.csv contains:
Frame Number: The frame where a crowd was detected.
Person Count in Crowd: Number of people in the crowd.



Sample CSV Output:
Frame Number,Person Count in Crowd
150,4
220,3

📸 Screenshots
(Add a screenshot or GIF of the video feed with bounding boxes here. You can record a short clip, convert it to a GIF using tools like ezgif.com, and upload it to the repository.)
🔧 Troubleshooting

No Groups Detected:
Check the console output for Minimum distance between people. If it’s consistently larger than DISTANCE_THRESHOLD, increase the threshold.
Ensure your video has groups of 3+ people standing close together.


YOLO Weights Not Found:
Ensure you’re connected to the internet so Ultralytics can download yolov11n.pt.
Or manually download the weights and place them in the project directory.


Performance Issues:
YOLOv11 nano is lightweight, but if the script runs slowly, consider using a smaller video resolution or a more powerful GPU.



📝 Notes

The input video (input_video.mp4) is not included in this repository due to GitHub’s file size limits. Provide your own video file for testing.
The script assumes a top-down or angled view of people for accurate distance calculations.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
🙌 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request with improvements or bug fixes.

Built with ❤️ by [AYAAN SHAHEER] on May 17, 2025
