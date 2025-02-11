import cv2
import os

def generate_dataset():
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
    print("Collecting samples is completed.")

# Run the function
generate_dataset()
