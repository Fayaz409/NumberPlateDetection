import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import logging
from django.http import StreamingHttpResponse
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

class NumberPlateDetector:
    def __init__(self, coco_model_path="yolov8n.pt", license_plate_model_path="license_plate_detector.pt"):
        self.coco_model = YOLO(coco_model_path)
        self.license_plate_model = YOLO(license_plate_model_path)
        self.reader_easyocr = easyocr.Reader(['en'])
        self.detected_texts = []
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
                box = result.boxes[i].xyxy[0].tolist()
                plates.append({
                    "box": box
                })
        return plates

    def _apply_brightness_contrast(self, input_img, brightness=0, contrast=0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def _preprocess_image(self, img):
        # Apply detail enhancement
        detail = cv2.detailEnhance(img, sigma_s=20, sigma_r=0.15)

        # Apply brightness and contrast adjustments
        magic_image_c1 = self._apply_brightness_contrast(detail, 120, 0)
        magic_image_c2 = self._apply_brightness_contrast(detail, 0, 40)
        magic_image_c3 = self._apply_brightness_contrast(detail, 50, 40)

        # Return the most appropriate preprocessed image
        return magic_image_c1

    def _apply_easyocr(self, img):
        preprocessed_img = self._preprocess_image(img)
        result = self.reader_easyocr.readtext(preprocessed_img)
        if result:
            return result[0][-2], preprocessed_img
        else:
            return None, preprocessed_img

    def _apply_tesseract(self, img):
        preprocessed_img = self._preprocess_image(img)
        text = pytesseract.image_to_string(preprocessed_img, config='--psm 8')
        return text.strip(), preprocessed_img

    def _compare_ocr(self, img):
        text_easyocr, preprocessed_easyocr = self._apply_easyocr(img)
        text_tesseract, preprocessed_tesseract = self._apply_tesseract(img)
        return text_easyocr, preprocessed_easyocr, text_tesseract, preprocessed_tesseract

    def gen_frames(self, Vehicle):
        cap = cv2.VideoCapture(0)  # Use 0 for webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            vehicle_boxes = self._detect_vehicle(frame)
            for (x1, y1, x2, y2) in vehicle_boxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            plate_boxes = self._detect_plate(frame)
            for plate in plate_boxes:
                (px1, py1, px2, py2) = plate['box']
                plate_img = frame[int(py1):int(py2), int(px1):int(px2)]
                text_easyocr, preprocessed_easyocr, text_tesseract, preprocessed_tesseract = self._compare_ocr(plate_img)
                plate_text = f"EasyOCR: {text_easyocr}, Tesseract: {text_tesseract}"
                cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)
                if plate_text:
                    cv2.putText(frame, plate_text, (int(px1), int(py1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Display the preprocessed image alongside the main frame
                    preprocessed_resized = cv2.resize(preprocessed_easyocr, (int(px2 - px1), int((py2 - py1) / 2)))
                    x_offset = int(px1)
                    y_offset = int(py2) + 10
                    if y_offset + preprocessed_resized.shape[0] <= frame.shape[0]:
                        frame[y_offset:y_offset + preprocessed_resized.shape[0], x_offset:x_offset + preprocessed_resized.shape[1]] = preprocessed_resized

            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        cap.release()
        cv2.destroyAllWindows()

    def process_video_stream(self, video_path, Vehicle):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            vehicle_boxes = self._detect_vehicle(frame)
            for (x1, y1, x2, y2) in vehicle_boxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            plate_boxes = self._detect_plate(frame)
            for plate in plate_boxes:
                (px1, py1, px2, py2) = plate['box']
                plate_img = frame[int(py1):int(py2), int(px1):int(px2)]
                text_easyocr, preprocessed_easyocr, text_tesseract, preprocessed_tesseract = self._compare_ocr(plate_img)
                plate_text = f"EasyOCR: {text_easyocr}, Tesseract: {text_tesseract}"
                cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)
                if plate_text:
                    cv2.putText(frame, plate_text, (int(px1), int(py1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    print(plate_text)

                    # Display the preprocessed image alongside the main frame
                    preprocessed_resized = cv2.resize(preprocessed_easyocr, (int(px2 - px1), int((py2 - py1) / 2)))
                    x_offset = int(px1)
                    y_offset = int(py2) + 10
                    if y_offset + preprocessed_resized.shape[0] <= frame.shape[0]:
                        frame[y_offset:y_offset + preprocessed_resized.shape[0], x_offset:x_offset + preprocessed_resized.shape[1]] = preprocessed_resized

            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        cap.release()
        cv2.destroyAllWindows()
