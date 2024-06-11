import logging
from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import os
from .models import Vehicle  # Import the Vehicle model
from .utils.number_plate import NumberPlateDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the NumberPlateDetector with your configuration files
coco_model_path = "yolov8n.pt" # Path the model can be accessed from OpenCVanpr\anprproject\baseanpr\utils\license_plate_detector.pt in this project
license_plate_model_path = "license_plate_detector.pt" # Path model can be accessed from OpenCVanpr\anprproject\baseanpr\utils\haarcascades\yolov8n.pt in this project

detector = NumberPlateDetector(coco_model_path=coco_model_path, license_plate_model_path=license_plate_model_path)

def gen_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    frame_count = 0  # Counter to skip frames
    vehicle_boxes = []
    plate_boxes = []
    plate_texts = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform vehicle and number plate detection every n frames
        if frame_count % detector.frame_skip == 0:
            vehicle_boxes = detector._detect_vehicle(frame)
            plate_boxes = []
            plate_texts = []
            for (px1, py1, px2, py2) in detector._detect_plate(frame):
                plate_img = frame[int(py1):int(py2), int(px1):int(px2)]
                plate_boxes.append((px1, py1, px2, py2))
                plate_text = detector._apply_easyocr(plate_img)
                plate_texts.append(plate_text)

        # Draw bounding boxes on every frame
        for (x1, y1, x2, y2) in vehicle_boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        for (px1, py1, px2, py2), plate_text in zip(plate_boxes, plate_texts):
            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)
            if plate_text:
                # Check if the detected plate text exists in the database
                plate_exists = Vehicle.objects.filter(plate=plate_text).exists()
                if plate_exists:
                    logger.info(f"Number plate {plate_text} exists in the database.")
                    cv2.putText(frame, f"{plate_text} (Exists)", (int(px1), int(py1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    logger.info(f"Number plate {plate_text} does not exist in the database.")
                    cv2.putText(frame, f"{plate_text} (Not Exists)", (int(px1), int(py1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        frame_count += 1
        
        # Encode frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        
        # Yield frame as multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()
    cv2.destroyAllWindows()

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def upload_video(request):
    if request.method == 'POST':
        video_file = request.FILES['file']
        video_path = f'temp_{video_file.name}'
        with open(video_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)
        
        def process_video_stream(video_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("Error opening video file.")
                return
            
            frame_count = 0
            vehicle_boxes = []
            plate_boxes = []
            plate_texts = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % detector.frame_skip == 0:
                    vehicle_boxes = detector._detect_vehicle(frame)
                    plate_boxes = []
                    plate_texts = []
                    for (px1, py1, px2, py2) in detector._detect_plate(frame):
                        plate_img = frame[int(py1):int(py2), int(px1):int(px2)]
                        plate_boxes.append((px1, py1, px2, py2))
                        plate_text = detector._apply_easyocr(plate_img)
                        plate_texts.append(plate_text)

                # Draw bounding boxes on every frame
                for (x1, y1, x2, y2) in vehicle_boxes:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                for (px1, py1, px2, py2), plate_text in zip(plate_boxes, plate_texts):
                    cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (255, 0, 0), 2)
                    if plate_text:
                        # Check if the detected plate text exists in the database
                        plate_exists = Vehicle.objects.filter(plate=plate_text).exists()
                        if plate_exists:
                            logger.info(f"Number plate {plate_text} exists in the database.")
                            cv2.putText(frame, f"{plate_text} (Exists)", (int(px1), int(py1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            logger.info(f"Number plate {plate_text} does not exist in the database.")
                            cv2.putText(frame, f"{plate_text} (Not Exists)", (int(px1), int(py1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                frame_count += 1

                # Encode frame as JPEG
                _, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
            cap.release()
            os.remove(video_path)

        return StreamingHttpResponse(process_video_stream(video_path), content_type='multipart/x-mixed-replace; boundary=frame')

    return render(request, 'baseanpr/upload.html')

def process_page(request):
    return render(request, 'baseanpr/process.html')

def index(request):
    return render(request, 'index.html')
