import cv2
import torch
from ultralytics import YOLO
import os

# Load your YOLOv11 model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path_to_your_yolov11_weights.pt')  # Update with yolov11 path
model = YOLO('Yolo_Model/Yolov11_v3_21000.pt')

# Set input and output video paths
source = 'UAV_IR/20190925_111757_1_9/visible'
output_video_path = source + "_out.mp4"
source = source + ".mp4"

# Open the video file
cap = cv2.VideoCapture(source)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files

# Initialize video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

for results in model(source, stream=True):
    # Extract the current frame
    ret, frame = cap.read()
    if not ret:
        break  # End of video
    
    # Iterate over detections in the current frame
    for box in results.boxes:
        # Extract bounding box coordinates, confidence, and class
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()  # Bounding box
        confidence = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class index
        
        # Draw the bounding box on the frame
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        label = f"{results.names[class_id]}: {confidence:.2f}"
        cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to: {output_video_path}")
