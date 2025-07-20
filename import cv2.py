import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
pygame.init()
alarm_sound = pygame.mixer.Sound("alarm.mp3") 
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
video_capture = cv2.VideoCapture(0)
eye_aspect_ratio_threshold = 0.3
frames_closed_threshold = 48
frames_closed = 0
drowsiness_detected = False
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        shape = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        left_eye_ear = eye_aspect_ratio(left_eye)
        right_eye_ear = eye_aspect_ratio(right_eye)
        ear = (left_eye_ear + right_eye_ear) / 2.0
        if ear < eye_aspect_ratio_threshold:
            frames_closed += 1
            if frames_closed >= frames_closed_threshold:
                if not drowsiness_detected:
                    print("Drowsiness detected!")
                    drowsiness_detected = True
                    alarm_sound.play() 
        else:
            frames_closed = 0
            drowsiness_detected = False
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()