import cv2 as cv
import numpy as np

video_feed = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(
    './cascades/haarcascade_frontalface_default.xml')
body_cascade = cv.CascadeClassifier('./cascades/haarcascade_fullbody.xml')
eye_cascade = cv.CascadeClassifier('./cascades/haarcascade_eye.xml')

while True:
    isTrue, frame = video_feed.read()

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

    cv.imshow('Live surveillance', frame)

    if cv.waitKey(60) & 0xFF == ord('q'):
        break

video_feed.release()
cv.destroyAllWindows()
