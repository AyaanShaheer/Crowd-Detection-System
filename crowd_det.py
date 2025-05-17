import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from scipy.spatial import distance
import time

# Load YOLOv11 model
model = YOLO("C:/Desktop/resolute_ai_task/yolo11n.pt")

# Define parameters
VIDEO_PATH = "C:/Desktop/resolute_ai_task/dataset_video.mp4"
OUTPUT_CSV = "crowd_detection_results.csv"
DISTANCE_THRESHOLD = 200
CROWD_SIZE = 3
FRAME_THRESHOLD = 5

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

# Prepare CSV logging
csv_data = []
frame_number = 0
group_tracker = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Perform YOLOv11 detection
    results = model(frame)
    detections = results[0].boxes

    # Store person bounding boxes and centers
    person_boxes = []
    person_centers = []
    for box in detections:
        if int(box.cls) == 0:  # Class 0 is "person"
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            person_centers.append((center_x, center_y))
            person_boxes.append((x1, y1, x2, y2))

    # Find groups of people based on proximity
    if len(person_centers) >= CROWD_SIZE:
        distances = distance.cdist(person_centers, person_centers, metric="euclidean")
        
        # Debugging: Print minimum distance between any two people
        if len(person_centers) > 1:
            np.fill_diagonal(distances, np.inf)
            min_distance = np.min(distances)
            print(f"Frame {frame_number}: Minimum distance between people: {min_distance:.2f} pixels")

        groups = []
        used = set()

        for i in range(len(person_centers)):
            if i in used:
                continue
            group = [i]
            used.add(i)
            for j in range(i + 1, len(person_centers)):
                if j in used:
                    continue
                if distances[i][j] < DISTANCE_THRESHOLD:
                    group.append(j)
                    used.add(j)
            if len(group) >= CROWD_SIZE:
                groups.append(group)

        print(f"Frame {frame_number}: Detected {len(groups)} groups of 3+ people")

        # Draw bounding boxes around detected groups
        for group in groups:
            # Get the bounding boxes of all people in the group
            group_boxes = [person_boxes[i] for i in group]
            # Find the min and max coordinates to enclose the group
            x1 = min(box[0] for box in group_boxes)
            y1 = min(box[1] for box in group_boxes)
            x2 = max(box[2] for box in group_boxes)
            y2 = max(box[3] for box in group_boxes)
            
            # Add some padding to the bounding box
            padding = 20
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
            
            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Add label with the number of people in the group
            label = f"Group: {len(group)} people"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Track groups across frames
        current_groups = set(tuple(sorted(group)) for group in groups)
        new_tracker = {}

        for group in current_groups:
            group_id = hash(group)
            if group_id in group_tracker:
                frame_count, person_count = group_tracker[group_id]
                new_tracker[group_id] = (frame_count + 1, len(group))
            else:
                new_tracker[group_id] = (1, len(group))

        for group_id, (frame_count, person_count) in new_tracker.items():
            if frame_count >= FRAME_THRESHOLD:
                csv_data.append({"Frame Number": frame_number, "Person Count in Crowd": person_count})
                print(f"Crowd detected at frame {frame_number}: {person_count} people")

        group_tracker = new_tracker

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save results to CSV
if csv_data:
    df = pd.DataFrame(csv_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")
else:
    print("No crowds detected.")

cap.release()
cv2.destroyAllWindows()