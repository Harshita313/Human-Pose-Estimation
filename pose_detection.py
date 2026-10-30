
import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Input video
video_path = "dance.mp4"

# Open the input video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open dance.mp4")
    print("Make sure dance.mp4 is in the project folder.")
    exit()

# Get input video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

# Create video writers
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

original_writer = cv2.VideoWriter(
    "original_with_pose.mp4",
    fourcc,
    fps,
    (width, height)
)

skeleton_writer = cv2.VideoWriter(
    "skeleton_only.mp4",
    fourcc,
    fps,
    (width, height)
)

# Check output writers
if not original_writer.isOpened() or not skeleton_writer.isOpened():
    print("Error: Could not create output video files.")
    cap.release()
    original_writer.release()
    skeleton_writer.release()
    exit()

print("Processing video... Please wait!")

# Initialize pose detection
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():

        # Read a frame
        success, frame = cap.read()

        if not success:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect pose
        results = pose.process(rgb_frame)

        # Create black background
        skeleton_frame = np.zeros_like(frame)

        # Draw skeleton if pose is detected
        if results.pose_landmarks:

            # Draw skeleton on original video
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Draw skeleton on black background
            mp_drawing.draw_landmarks(
                skeleton_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # Save processed frames
        original_writer.write(frame)
        skeleton_writer.write(skeleton_frame)

        # Resize videos for display
        frame_small = cv2.resize(
            frame,
            (480, 360)
        )

        skeleton_small = cv2.resize(
            skeleton_frame,
            (480, 360)
        )

        # Display both videos
        cv2.imshow(
            "Original Dance with Pose Detection",
            frame_small
        )

        cv2.imshow(
            "Skeleton Only",
            skeleton_small
        )

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.")
            break

# Release everything
cap.release()
original_writer.release()
skeleton_writer.release()
cv2.destroyAllWindows()

print("\nProcessing finished!")
print("Saved: original_with_pose.mp4")
print("Saved: skeleton_only.mp4")
print("Location:", os.getcwd())

