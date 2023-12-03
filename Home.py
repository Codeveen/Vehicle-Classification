import cv2
from flask import Flask, render_template, request, jsonify
import os

import cvzone
from ultralytics import YOLO
import cv2 as cv
import math
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/post', methods=['POST'])
def post():
    if 'imageUpload' not in request.files:
        return "No file part"

    file = request.files['imageUpload']

    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to a folder (e.g., 'static')
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Load the YOLO model
    model = YOLO('C://Users//prave//OneDrive//Documents//GitHub//Vehicle-Classification//Vehicles-OpenImages.v1-416x416.yolov8//runs//detect//train4//weights//best.pt')

    # Define the class names
    classNames = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

    # Read the image
    img = cv.imread(file_path)

    # Perform object detection
    results = model(img, stream=True)

    data = {'Ambulance': 0, 'Bus': 0, 'Car': 0, 'Motorcycle': 0, 'Truck': 0}

    # Process the detected objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            #
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            if classNames[cls] in data:
                data[classNames[cls]] += 1

    display_folder = 'static'
    display_path = os.path.join(display_folder, file.filename)
    cv2.imwrite(display_path, img)
    #
    # cv.imshow("Image", img)
    # cv.waitKey(0)
    return jsonify({'success': True, 'image_path': display_path, 'Ambulance':data['Ambulance'], 'Bus': data['Bus'], 'Car': data['Car'], 'Motorcycle': data['Motorcycle'], 'Truck':data['Truck']})

if __name__ == "__main__":
    app.run(debug=True)

