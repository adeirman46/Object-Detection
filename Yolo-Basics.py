from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')

# Inference
results = model('Images\Toyota_Many.jpg', show=True)
cv2.waitKey(0)
