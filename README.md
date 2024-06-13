# ANPR Project

This project is an Automatic Number Plate Recognition (ANPR) system built using Django, OpenCV, EasyOCR, and Ultralytics YOLO models. The system is designed to detect and recognize number plates from video streams.

## Installation

### Prerequisites

Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Setup

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd anprproject
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows:

      ```bash
      venv\Scripts\activate
      ```

    - On macOS and Linux:

      ```bash
      source venv/bin/activate
      ```

4. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    Make sure your `requirements.txt` includes the following dependencies:
    - OpenCV
    - EasyOCR
    - Ultralytics YOLO
    - Django

    If you don't have a `requirements.txt`, you can install them manually:

    ```bash
    pip install opencv-python easyocr ultralytics django
    ```

## Running the Server

1. **Navigate to the project directory:**

    ```bash
    cd anprproject
    ```

2. **Run the Django development server:**

    ```bash
    python manage.py runserver
    ```

3. **Access the application:**

    Open your web browser and go to `http://127.0.0.1:8000`.

## Important Code Files

### number_plate.py

This file contains the `NumberPlateDetector` class responsible for detecting vehicles and number plates from video frames.

- **Initialization**: Loads YOLO models for COCO and license plate detection, and initializes EasyOCR.
- **Vehicle Detection**: `_detect_vehicle` method uses the COCO model to detect vehicles.
- **Plate Detection**: `_detect_plate` method uses the license plate model to detect plates.
- **OCR Application**: `_apply_easyocr` method uses EasyOCR to extract text from detected plates.
- **Video Processing**: `detect_from_video` method processes video frames, detects vehicles and plates, applies OCR, and collects detected texts.

### views.py

This file contains the Django views for handling video streams and file uploads.

- **gen_frames**: Captures video from the webcam, processes frames using `NumberPlateDetector`, and streams the processed frames.
- **video_feed**: Provides the video stream to the client.
- **upload_video**: Handles video file uploads, processes the uploaded video, and streams the processed frames.
- **process_page**: Renders the processing page.
- **index**: Renders the home page.

### urls.py

This file defines the URL patterns for the application.

- **upload_video/**: Endpoint for video file uploads.
- **process/**: Endpoint for video streaming using webcame.
- **/**: Home page.

## Model Paths

The YOLO models are required for vehicle and license plate detection. Ensure the models are correctly placed and accessible within the project.

- **COCO Model Path**: 

  ```python
  coco_model_path = "OpenCVanpr/anprproject/baseanpr/utils/yolov8n.pt"
  ```

- **License Plate Model Path**:

  ```python
  license_plate_model_path = "OpenCVanpr/anprproject/baseanpr/utils/license_plate_detector.pt"
  ```

Ensure these paths are correct and the models are present at these locations in your project directory.

## How to Use

1. **Start the server** as described in the "Running the Server" section.
2. **Access the home page** at `http://127.0.0.1:8000`.
3. **Upload a video** through `http://127.0.0.1:8000/upload_video` page.
4. **View the video feed** `http://127.0.0.1:8000/process` to see real-time number plate detection and recognition through webcame.
5. **Check the database** for recognized plates and their existence status.

This project aims to provide a comprehensive ANPR solution by integrating powerful models and libraries within a Django framework, making it easy to deploy and scale.










