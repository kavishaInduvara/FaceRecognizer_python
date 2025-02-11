import cv2
import numpy as np
from PIL import Image  # pip install pillow
import os

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    # Convert the image to grayscale for face detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)

    for (x, y, w, h) in features:
        # Draw a rectangle around detected faces
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Predict the ID and confidence using the trained model
        id, pred = clf.predict(gray_img[y:y + h, x:x + w])
        confidence = int(100 * (1 - pred / 300))

        # Display the name based on the ID and confidence
        if confidence > 70:
            if id == 1:
                cv2.putText(img, "Kavisha", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif id == 2:
                cv2.putText(img, "Manish", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    return img


# Load the Haar Cascade face detector
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Check if the Haar Cascade file exists
if faceCascade.empty():
    raise FileNotFoundError("Error: 'haarcascade_frontalface_default.xml' not found. Place it in the script directory.")

# Load the trained classifier
clf = cv2.face.LBPHFaceRecognizer_create()

try:
    clf.read("classifier.xml")
except Exception as e:
    raise FileNotFoundError("Error: 'classifier.xml' not found. Train the classifier first.") from e

# Start the video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera

if not video_capture.isOpened():
    raise Exception("Error: Unable to access the camera.")

print("Press 'Enter' to exit the video feed.")

while True:
    ret, img = video_capture.read()

    if not ret:
        print("Error: Unable to read from the camera.")
        break

    img = draw_boundary(img, faceCascade, 1.3, 6, (255, 255, 255), "Face", clf)
    cv2.imshow("Face Detection", img)

    # Exit the loop when 'Enter' is pressed
    if cv2.waitKey(1) == 13:  # 13 is the ASCII code for 'Enter'
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
