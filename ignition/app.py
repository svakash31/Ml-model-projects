import cv2
import numpy as np
import face_recognition
import os
import pickle

# Paths for storing known faces
KNOWN_FACES_DIR = "known_faces"
ALERT_LOG = "alert_log.txt"

# Load or create known face encodings
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Load known face encodings
known_face_encodings = []
known_face_names = []

if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

def send_alert():
    """Logs an alert when an unknown face is detected."""
    with open(ALERT_LOG, "a") as f:
        f.write("ALERT: Unknown face detected - Ignition Blocked!\n")
    print("❌ ALERT: Unknown face detected! Ignition blocked.")

def save_new_face(face_encoding, frame, name):
    """Saves a new face encoding and image to the database."""
    face_filename = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    cv2.imwrite(face_filename, frame)
    
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    
    # Save updated encodings
    with open("known_faces.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print(f"✅ New face {name} added to the system!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    import cv2
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime

# Paths for storing known faces
KNOWN_FACES_DIR = "known_faces"
KNOWN_FACES_FILE = "known_faces.pkl"
ALERT_LOG = "alert_log.txt"

# Create directory if not exists
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Load known faces
if os.path.exists(KNOWN_FACES_FILE) and os.path.getsize(KNOWN_FACES_FILE) > 0:
    with open(KNOWN_FACES_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings, known_face_names = [], []

# Initialize webcam
cap = cv2.VideoCapture(0)

def send_alert():
    """Logs an alert when an unknown face is detected."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ALERT_LOG, "a") as f:
        f.write(f"[{timestamp}] ALERT: Unknown face detected - Ignition Blocked!\n")
    print(f"❌ [{timestamp}] ALERT: Unknown face detected! Ignition blocked.")

def save_new_face(face_encoding, frame, face_location, name):
    """Saves a new face encoding and cropped image."""
    top, right, bottom, left = face_location
    face_image = frame[top:bottom, left:right]  # Crop the face
    face_filename = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    cv2.imwrite(face_filename, face_image)

    # Update face data
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

    # Save updated encodings
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)

    print(f"✅ New face {name} added to the system!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            ignition_status = "✅ Ignition Allowed"
            color = (0, 255, 0)  # Green
        else:
            ignition_status = "❌ Ignition Blocked"
            color = (0, 0, 255)  # Red
            send_alert()  # Alert for unrecognized face
            
            # Save new face
            new_name = f"user_{len(known_face_names) + 1}"
            save_new_face(face_encoding, frame, face_location, new_name)

        # Draw bounding box and label
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name} - {ignition_status}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show frame
    cv2.imshow("Face Recognition Ignition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
            ignition_status = "✅ Ignition Allowed"
            color = (0, 255, 0)  # Green for recognized face
        else:
            ignition_status = "❌ Ignition Blocked"
            color = (0, 0, 255)  # Red for unrecognized face
            send_alert()  # Alert for unrecognized face
            
            # Save new face
            new_name = f"user_{len(known_face_names) + 1}"
            save_new_face(face_encoding, frame, new_name)

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"{name} - {ignition_status}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show frame
    cv2.imshow("Face Recognition Ignition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
