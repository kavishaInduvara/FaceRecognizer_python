import cv2
import numpy
import os
import tkinter as tk
from tkinter import messagebox
 
window = tk.Tk()
window.title("Face Recognition system")
 
l1 = tk.Label(window, text="Name", font=("Algerian",20))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)
 
l2 = tk.Label(window, text="Age", font=("Algerian",20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)
 
l3 = tk.Label(window, text="Address", font=("Algerian",20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)
 
def detect_face():
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
                    cv2.putText(img, "Amma", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
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
    

b1 = tk.Button(window, text="Detect Face", font=("Algerian",20),bg="orange",fg="red", command=detect_face)
b1.grid(column=0, row=4)

def train_classifier(data_dir):
    # Check if the dataset directory exists
    data_dir="D:/FaceRecognizer/VSCode/Python/data"
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
    messagebox.showinfo('Result','Training Dataset is Completed!')
 
b2 = tk.Button(window, text="Train Classifier", font=("Algerian",20), bg="green", fg="orange", command=train_classifier)
b2.grid(column=1, row=4)

def generate_dataset():
    if(t1.get()=="" or t2.get()=="" or t3.get()==""):
        messagebox.showinfo('Result','Please Fill all Details!')
    else:
        # Load the pre-trained Haar Cascade for face detection
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Function to crop faces from the frame
        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            # Check if no faces are detected
            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]
                return cropped_face

        # Open the default camera (0 for default)
        cap = cv2.VideoCapture(0)

        # Check if the camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open the camera.")
            return

        # User ID and image ID initialization
        id = 1
        img_id = 0

        # Ensure the 'data' directory exists
        if not os.path.exists("data"):
            os.makedirs("data")

        # Capture frames continuously
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Process the frame to detect and crop faces
            cropped_face = face_cropped(frame)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Save the cropped face image
                file_name_path = f"data/user.{id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)

                # Display the current image ID on the face
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                # Show the cropped face in a window
                cv2.imshow("Cropped Face", face)

            # Exit the loop if 'Enter' is pressed or 200 images are captured
            if cv2.waitKey(1) == 13 or img_id == 200:  # 13 is the Enter key
                break

        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result','Generate Dataset Completed!')
# Run the function
 
b3 = tk.Button(window, text="Generate dataset", font=("Algerian",20), bg="pink", fg="black", command=generate_dataset)
b3.grid(column=2, row=4)
 
window.geometry("800x200")
window.mainloop()
 
