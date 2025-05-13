from ultralytics import YOLO
import cv2
import math
import numpy as np
import requests
from urllib.parse import urljoin
import time


def main():
    # ESP32-CAM IP address - update this with your ESP32's IP address
    # The IP address will be printed in the Arduino Serial Monitor when the ESP32 connects to WiFi
    esp32_ip = "192.168.1.22"  # Using your ESP32-CAM's IP address from the error message

    # URL for the video stream
    stream_url = f"http://{esp32_ip}:81/stream"

    # Load YOLO model
    model = YOLO("yolo-Weights/yolov8n.pt")

    # Object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    # Create a window
    cv2.namedWindow('ESP32-CAM YOLO Detection', cv2.WINDOW_NORMAL)

    # Connect to ESP32-CAM stream using OpenCV VideoCapture
    print(f"Connecting to ESP32-CAM stream at {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Failed to open ESP32-CAM stream. Trying alternative method...")
        # If the above method fails, we'll use a different approach with requests
        use_requests = True
    else:
        use_requests = False

    # If using requests method
    if use_requests:
        try:
            # Start a session for better performance
            session = requests.Session()
            # Send a request to the stream URL
            response = session.get(stream_url, stream=True, timeout=10)

            if response.status_code != 200:
                print(f"Failed to connect to ESP32-CAM stream. Status code: {response.status_code}")
                return

            # Variables for parsing MJPEG stream
            bytes_data = bytes()

            # Process the multipart stream
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    bytes_data += chunk
                    # Look for JPEG frame markers
                    a = bytes_data.find(b'\xff\xd8')  # JPEG start
                    b = bytes_data.find(b'\xff\xd9')  # JPEG end

                    if a != -1 and b != -1:
                        # Extract the JPEG frame
                        jpg = bytes_data[a:b + 2]
                        bytes_data = bytes_data[b + 2:]

                        # Decode the image
                        img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                        if img is None:
                            continue

                        # Run YOLO detection
                        process_frame(img, model, classNames)

                        # Display the frame
                        cv2.imshow('ESP32-CAM YOLO Detection', img)

                        # Break the loop if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        except Exception as e:
            print(f"Error accessing ESP32-CAM stream: {e}")
    else:
        # Using OpenCV VideoCapture method
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to receive frame from ESP32-CAM")
                # Wait a bit before retrying
                time.sleep(1)
                continue

            # Run YOLO detection
            process_frame(img, model, classNames)

            # Display the frame
            cv2.imshow('ESP32-CAM YOLO Detection', img)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    if not use_requests:
        cap.release()
    cv2.destroyAllWindows()


def process_frame(img, model, classNames):
    """Process a single frame with YOLO detection"""
    # Run model prediction
    results = model(img, stream=True)

    # Process detection results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Calculate confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # Get class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # Add text with class name
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, f"{classNames[cls]} {confidence:.2f}", org, font, fontScale, color, thickness)


if __name__ == "__main__":
    main()