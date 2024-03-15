import json

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from train_data import train_data,trained_models

# data training
trained_models = train_data()

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, []

    detected_faces = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        detected_faces.append((roi, (x, y, w, h)))

    return img, detected_faces

def search(trained_models, face):
    confidence = 0
    user = 'unknown'
    for user_id, model in trained_models.items():
        result = model.predict(face)
        # print('label' + label + 'confidence ' + result)
        curr_confidence = float(100 * (1 - (result[1]) / 300))
        print(user_id, curr_confidence)
        if curr_confidence > confidence:
            confidence = curr_confidence
            user = user_id
    return confidence, user


img = cv2.imread('Test/test01.jpg')
img, detected_faces = face_detector(img)
print(len(trained_models))
for roi, (x,y,w,h) in detected_faces:
    # print(x, y + h)
    try:
        face_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        confidence, user = search(trained_models,face_gray)
        print(confidence)
        if confidence > 82:
            cv2.putText(img, user, (x, y+h), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        else:
            cv2.putText(img, user, (x, y + h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    except:
        cv2.putText(img, "Face not found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        pass


img = cv2.resize(img,(500,500))
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()