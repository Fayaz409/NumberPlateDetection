from django.shortcuts import render
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import logging
import os
from .models import Vehicle  # Import the Vehicle model
from .utils.number_plate2 import NumberPlateDetector  # Adjusted import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the NumberPlateDetector with your configuration files
coco_model_path = r"C:\Users\ftuni\Desktop\OpenCVanpr\anprproject\baseanpr\utils\haarcascades\yolov8n.pt" 
license_plate_model_path = r"C:\Users\ftuni\Desktop\OpenCVanpr\anprproject\baseanpr\utils\haarcascades\license_plate_detector.pt"

detector = NumberPlateDetector(coco_model_path=coco_model_path, license_plate_model_path=license_plate_model_path)

def gen_frames():
    return detector.gen_frames(Vehicle)

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
        
        def process_video_stream():
            return detector.process_video_stream(video_path, Vehicle)

        return StreamingHttpResponse(process_video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

    return render(request, 'baseanpr/upload.html')

def process_page(request):
    return render(request, 'baseanpr/process.html')

def index(request):
    return render(request, 'index.html')
