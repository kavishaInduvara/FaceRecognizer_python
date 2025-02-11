# Face Recognizer Using Python, OpenCV, and Haarcascade

## Overview
This project is a face recognition system developed using Python, OpenCV, and Haarcascade. The system can:
- Enroll faces by capturing 200 images per person.
- Train a face recognition model on the captured dataset.
- Recognize faces in real time.

## Features
- **Face Enrollment:** Capture 200 images per person for dataset creation.
- **Model Training:** Train a face recognition model using the enrolled faces.
- **Face Recognition:** Detect and recognize enrolled faces in real-time.

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have OpenCV and NumPy installed:
   ```bash
   pip install opencv-python numpy
   ```

## Usage

### Step 1: Face Enrollment
Run the following script to enroll a new person:
```bash
python enroll_faces.py
```
Follow the prompts to enter the person's name and capture 200 images for training.

### Step 2: Train the Face Recognition Model
Once face images are collected, train the model by running:
```bash
python train_model.py
```
This will generate a face recognition model using the enrolled face images.

### Step 3: Face Recognition
Start the face recognition process by running:
```bash
python recognize_faces.py
```
The system will use the trained model to recognize faces in real time.

## File Structure
```
project-root/
|-- enroll_faces.py       # Script to capture and store face images
|-- train_model.py        # Script to train the face recognition model
|-- recognize_faces.py    # Script to detect and recognize faces
|-- haarcascade_frontalface_default.xml # Haarcascade file for face detection
|-- requirements.txt      # Dependencies
|-- dataset/              # Folder for storing captured face images
|-- model/                # Folder for storing trained model
```

## Requirements
- Python 3.6+
- OpenCV
- NumPy

## Notes
- The Haarcascade XML file is used for face detection.
- Ensure the camera is properly connected and accessible.
- The system captures 200 images per person to ensure robust face recognition.

## License
This project is licensed under the MIT License.

## Contributions
Feel free to contribute by submitting issues or pull requests.

## Author
[Kavisha Induvara](https://github.com/YourGitHubUsername)

## Acknowledgements
- OpenCV for computer vision functionalities.
- The Haarcascade model for face detection.

