import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from sort import Sort

# Load YOLO models
model1 = YOLO('C:/Users/Najeeb ULLAH/Desktop/Projects/runs/detect/train19/weights/best.pt').to('cuda')
model2 = YOLO('C:/Users/Najeeb ULLAH/Desktop/Projects/runs/detect/train11/weights/best.pt').to('cuda')

# Initialize tracker and other variables
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
ball_positions = []

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, 
                    enable_segmentation=False, smooth_segmentation=True, 
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open video file
cap = cv2.VideoCapture('Tennis-Match3.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Desired resolution
desired_width = 1280
desired_height = 720

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter('output_resized.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (desired_width, desired_height))

# Additional variables
previous_ball_position = None
previous_person_position = None

# Loop through video frames
while True:
    success, img = cap.read()
    if not success:
        print("Video ended or error in reading frame.")
        break

    # Resize frame
    img = cv2.resize(img, (desired_width, desired_height))

    # Convert frame to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultsmp = pose.process(img_rgb)

    # Run YOLO model2 for court detection
    results2 = model2(img)
    court_box = None
    for r in results2:
        for box in r.boxes:
            xc1, yc1, xc2, yc2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            if model2.names[cls] == 'court':
                court_box = (xc1, yc1, xc2, yc2)
                cv2.rectangle(img, (xc1, yc1), (xc2, yc2), (255, 0, 0), 2)
                label = f'{model2.names[cls]}: {conf:.2f}'
                cv2.putText(img, label, (xc1, yc1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Run YOLO model1 for object detection (e.g., tennis ball)
    results1 = model1(img)
    current_ball_position = None
    for r in results1:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{model1.names[cls]}: {conf:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if model1.names[cls] == 'tennis ball' and conf > 0.4:
                # Track ball position
                x, y = (x1 + x2) // 2, (y1 + y2) // 2
                current_ball_position = (x, y)
                cv2.circle(img, (x, y), 3, (255, 0, 0), 5)

    # Calculate ball speed
    if previous_ball_position and current_ball_position:
        ball_speed = np.linalg.norm(np.array(current_ball_position) - np.array(previous_ball_position)) * fps
        cv2.putText(img, f"Ball Speed: {ball_speed:.2f} px/s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    previous_ball_position = current_ball_position

    # Get person position from MediaPipe Pose
    current_person_position = None
    if resultsmp.pose_landmarks:
        landmarks = resultsmp.pose_landmarks.landmark
        # Use the center of the hips as the person's position
        x = int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2 * img.shape[1])
        y = int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2 * img.shape[0])
        current_person_position = (x, y)

    # Calculate person speed
    if previous_person_position and current_person_position:
        person_speed = np.linalg.norm(np.array(current_person_position) - np.array(previous_person_position)) * fps
        cv2.putText(img, f"Person Speed: {person_speed:.2f} px/s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    previous_person_position = current_person_position

    # Display frame
    cv2.imshow("Tennis Detection", img)
    out.write(img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
