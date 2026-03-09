from flask import Flask, render_template, request
from datetime import datetime
import cv2
import os

# Import utility functions
from utils import capture_images, encode_faces, mark_attendance, check_face_already_registered

# Import database connection
from database import get_connection

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['POST'])
def register():

    name = request.form['name'].strip()
    roll = request.form['roll'].strip()
    email = request.form['email'].strip()

    unique_name = f"{name}_{roll}"
    temp_image_path = "temp_check.jpg"

    try:

        # Capture one image to check duplicate face
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        if not ret:
            return render_template('index.html', message="❌ Failed to open camera.")

        cv2.imwrite(temp_image_path, frame)

        cap.release()
        cv2.destroyAllWindows()

        # Check if face already registered
        exists, matched_name = check_face_already_registered(temp_image_path)

        os.remove(temp_image_path)

        if exists:
            return render_template('index.html', message=f"⚠️ Face already registered as {matched_name}")

        # Capture dataset images
        capture_images(unique_name)

        # Encode faces
        messages = encode_faces(new_user=unique_name, new_image_path=f"dataset/{unique_name}/0.jpg")

        for msg in messages:
            if "already registered" in msg:
                return render_template('index.html', message=msg)

        # ==============================
        # SAVE STUDENT INTO MYSQL
        # ==============================

        conn = get_connection()
        cursor = conn.cursor()

        now = datetime.now()

        query = """
        INSERT INTO students (name,email,roll,date,time)
        VALUES (%s,%s,%s,%s,%s)
        """

        cursor.execute(query, (
            name,
            email,
            roll,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S")
        ))

        conn.commit()

        cursor.close()
        conn.close()

        return render_template('index.html', message=f"✅ Successfully registered {name}")

    except Exception as e:
        return render_template('index.html', message=f"❌ Error: {str(e)}")


@app.route('/mark', methods=['POST'])
def mark():

    try:

        recognized = mark_attendance()

        if recognized:
            return render_template('index.html',
                                   message=f"✅ Attendance marked for: {', '.join(recognized)}")
        else:
            return render_template('index.html',
                                   message="⚠️ No known face recognized.")

    except Exception as e:
        return render_template('index.html', message=f"❌ Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)