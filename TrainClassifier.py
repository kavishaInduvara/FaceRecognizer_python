import os
import cv2
from PIL import Image  # pip install pillow
import numpy as np     # pip install numpy

def train_classifier(data_dir):
    # Check if the dataset directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' does not exist.")
        return

    # Get paths of all .jpg files in the dataset directory
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")]
    if not path:
        print(f"Error: No .jpg files found in the dataset directory '{data_dir}'.")
        return

    faces = []
    ids = []

    for image in path:
        try:
            # Open and convert the image to grayscale
            img = Image.open(image).convert('L')  # 'L' means grayscale
            imageNp = np.array(img, 'uint8')     # Convert image to NumPy array

            # Extract ID from the filename (e.g., user.1.1.jpg -> ID = 1)
            id = int(os.path.split(image)[1].split(".")[1])

            # Append the image and ID to the lists
            faces.append(imageNp)
            ids.append(id)
        except Exception as e:
            print(f"Error processing file {image}: {e}")
            continue

    ids = np.array(ids)  # Convert IDs list to NumPy array

    # Train the Local Binary Patterns Histograms (LBPH) face recognizer
    print("Training the classifier...")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)

    # Save the trained classifier to a file
    classifier_path = "classifier.xml"
    clf.write(classifier_path)
    print(f"Training completed. Classifier saved as '{classifier_path}'.")

# Run the function with the dataset directory
train_classifier("data")
