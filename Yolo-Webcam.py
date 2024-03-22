import cv2
from ultralytics import YOLO
import supervision as sv
import torch

video = cv2.VideoCapture('Videos\cars.mp4')
# load the model
model = YOLO("yolov8m.pt")
bbox_annotator = sv.BoxAnnotator()

# check torch using GPU
print(torch.cuda.is_available())


while True:
    ret, frame = video.read()
    # resize frame
    frame = cv2.resize(frame, (360, 240))
    results = model(frame)[0]
    detection = sv.Detections.from_ultralytics(results)
    detection = detection[detection.confidence > 0.5]
    labels = [
        results.names[class_id]
        for class_id in detection.class_id
    ]
    # add confidence value to labels
    labels = [
        f"{label} {conf:.2f}"
        for label, conf in zip(labels, detection.confidence)
    ]
    frame = bbox_annotator.annotate(frame, detection, labels)
    cv2.imshow("Image", frame)
    cv2.waitKey(1)