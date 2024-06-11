import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

class NumberPlateDetector:
    def __init__(self, coco_model_path="yolov8n.pt", license_plate_model_path="license_plate_detector.pt"):
        self.coco_model = YOLO(coco_model_path)
        self.license_plate_model = YOLO(license_plate_model_path)
        self.reader = easyocr.Reader(['en'])
        self.frame_skip = 10  # Process every 10th frame
        self.detected_texts = []

        # Ensure you have the correct class names for your model
        self.coco_classes = self.coco_model.names

    def _detect_vehicle(self, img):
        results = self.coco_model(img)
        boxes = []
        for result in results:
            for i in range(len(result.boxes)):
                cls = int(result.boxes[i].cls)
                if self.coco_classes[cls] in ["car", "motorbike", "bus", "truck"]:
                    boxes.append(result.boxes[i].xyxy[0].tolist())
        return boxes

    def _detect_plate(self, img):
        results = self.license_plate_model(img)
        plates = []
        for result in results:
            for i in range(len(result.boxes)):
                plates.append(result.boxes[i].xyxy[0].tolist())
        return plates

    def _apply_easyocr(self, img):
        if img is None:
            raise ValueError("No plate image detected.")
        result = self.reader.readtext(img)
        if result:
            text = result[0][-2]
            print(f"Detected text: {text}")
            return text
        else:
            print("No text detected.")
            return None

    def detect_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return []

        frame_count = 0
        detected_texts = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_skip == 0:
                vehicle_boxes = self._detect_vehicle(frame)

                for (x1, y1, x2, y2) in vehicle_boxes:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    plate_boxes = self._detect_plate(frame)
                    for (px1, py1, px2, py2) in plate_boxes:
                        plate_img = frame[int(py1):int(py2), int(px1):int(px2)]
                        cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)
                        plate_text = self._apply_easyocr(plate_img)
                        if plate_text:
                            detected_texts.append(plate_text)
                            cv2.putText(frame, plate_text, (int(px1), int(py1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            frame_count += 1

        cap.release()
        return detected_texts
