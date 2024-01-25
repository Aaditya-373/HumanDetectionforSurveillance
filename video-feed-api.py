from flask import Flask, render_template, Response
import cv2 as cv
import numpy as np

app = Flask(__name__, template_folder='templates')
video_feed = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(
    './cascades/haarcascade_frontalface_default.xml')
body_cascade = cv.CascadeClassifier('./cascades/haarcascade_fullbody.xml')
eye_cascade = cv.CascadeClassifier('./cascades/haarcascade_eye.xml')


def generate_frames():
    while True:
        success, frame = video_feed.read()
        if not success:
            break

        grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            grayscale_frame, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
        bodies = body_cascade.detectMultiScale(
            grayscale_frame, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
        eyes = eye_cascade.detectMultiScale(
            grayscale_frame, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv.putText(frame, 'Face', (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        for x, y, w, h in eyes:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, 'Eyes', (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')


@app.route('/live-feed')
def live_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
