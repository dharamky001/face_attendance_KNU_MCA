from database import get_connection
import os
import cv2
import pickle
import numpy as np
from datetime import datetime
import face_recognition
import tkinter as tk
from tkinter import messagebox
import shutil


# -----------------------------
# Blur Detection
# -----------------------------
def is_blurry(image, threshold=30):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return lap_var < threshold


# -----------------------------
# Check Duplicate Face
# -----------------------------
def check_face_already_registered(new_image_path):

    encoding_path = "encodings/encodings.pkl"

    if not os.path.exists(encoding_path):
        return False, None

    with open(encoding_path, "rb") as f:
        data = pickle.load(f)

    image = cv2.imread(new_image_path)

    if image is None:
        return False, None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)

    if not boxes:
        return False, None

    new_encoding = face_recognition.face_encodings(rgb, boxes)[0]

    distances = face_recognition.face_distance(data["encodings"], new_encoding)

    if len(distances) == 0:
        return False, None

    min_distance = min(distances)

    if min_distance < 0.5:
        matched_name = data["names"][np.argmin(distances)]
        return True, matched_name

    return False, None


# -----------------------------
# Capture Images
# -----------------------------
def capture_images(name):

    path = os.path.join("dataset", name)
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    count = 0
    attempts = 0

    while count < 50 and attempts < 100:

        ret, frame = cap.read()
        if not ret:
            break

        if not is_blurry(frame):
            img_path = os.path.join(path, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            message = f"Image {count+1}/50 saved"
            count += 1
        else:
            message = "Blurry image skipped"

        cv2.putText(frame, message,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)
        cv2.putText(frame, "Look straight. Hold still.", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        cv2.imshow("Face Registration", frame)

        attempts += 1

        # Increase waitKey time to slow down capture (e.g., 1000 ms = 1 second)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count < 50:
        raise ValueError(f"Only {count}/50 images captured. Try again.")


# -----------------------------
# Encode Faces
# -----------------------------
def encode_faces(new_user=None, new_image_path=None):

    dataset_path = "dataset"
    encoding_path = "encodings/encodings.pkl"

    known_encodings = []
    known_names = []
    messages = []

    if not os.path.exists(dataset_path):
        return ["Dataset folder not found"]

    for person_name in os.listdir(dataset_path):

        person_folder = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_folder):
            continue

        for image_file in os.listdir(person_folder):

            # only allow valid image files
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(person_folder, image_file)

            try:
                print("Encoding:", image_path)

                image = cv2.imread(image_path)

                # Skip corrupted image
                if image is None:
                    print("Skipping corrupted:", image_path)
                    continue

                # Ensure 3 channel image
                if len(image.shape) != 3 or image.shape[2] != 3:
                    print("Invalid format:", image_path)
                    continue

                # Ensure uint8 format
                if image.dtype != "uint8":
                    image = image.astype("uint8")

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                boxes = face_recognition.face_locations(rgb)

                if len(boxes) == 0:
                    print("No face found:", image_path)
                    continue

                encodings = face_recognition.face_encodings(rgb, boxes)

                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(person_name)

            except Exception as e:

                print("Error processing:", image_path, e)
                continue

    os.makedirs("encodings", exist_ok=True)

    with open(encoding_path, "wb") as f:
        pickle.dump(
            {
                "encodings": known_encodings,
                "names": known_names
            },
            f
        )

    messages.append(f"Encoded {len(known_encodings)} faces from {len(set(known_names))} users")

    return messages


# -----------------------------
# Mark Attendance
# -----------------------------
from database import get_connection  # make sure this is imported

def mark_attendance():
    encoding_path = "encodings/encodings.pkl"
    if not os.path.exists(encoding_path):
        print("Encodings file not found")
        return []

    # load encodings
    with open(encoding_path, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Camera not opened")
        return []

    recognized_names = []
    conn = get_connection()
    cursor = conn.cursor()

    print("Camera started... Press ESC to stop")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = frame.astype("uint8")
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            if name != "Unknown" and name not in recognized_names:
                recognized_names.append(name)
                print("Attendance marked for:", name)

                # Save attendance in DB
                now = datetime.now()
                today_str = now.strftime("%Y-%m-%d")

                # split name into actual name and roll
                parts = name.split("_")
                student_name = parts[0]
                student_roll = parts[1] if len(parts) > 1 else "N/A"

                # Check if attendance already exists
                cursor.execute(
                    "SELECT * FROM attendance WHERE name=%s AND roll=%s AND date=%s",
                    (student_name, student_roll, today_str)
                )
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO attendance (name, roll, date, time) VALUES (%s, %s, %s, %s)",
                        (student_name, student_roll, today_str, now.strftime("%H:%M:%S"))
                    )
                    conn.commit()

            # draw rectangle around face
            top, right, bottom, left = face_location
            top *= 4; right *= 4; bottom *= 4; left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Attendance Camera", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    cursor.close()
    conn.close()

    return recognized_names

# -----------------------------
# Delete User
# -----------------------------
def delete_user_full(user_name, roll):

    unique_name = f"{user_name}_{roll}"

    dataset_folder = os.path.join("dataset", unique_name)

    encoding_path = "encodings/encodings.pkl"

    if os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)

    if os.path.exists(encoding_path):

        with open(encoding_path, "rb") as f:
            data = pickle.load(f)

        encodings = data["encodings"]
        names = data["names"]

        filtered_encodings = []
        filtered_names = []

        for enc, name in zip(encodings, names):

            if name != unique_name:
                filtered_encodings.append(enc)
                filtered_names.append(name)

        with open(encoding_path, "wb") as f:

            pickle.dump({
                "encodings": filtered_encodings,
                "names": filtered_names
            }, f)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM students WHERE name=%s AND roll=%s",
        (user_name, roll)
    )

    cursor.execute(
        "DELETE FROM attendance WHERE name=%s AND roll=%s",
        (user_name, roll)
    )

    conn.commit()

    cursor.close()
    conn.close()