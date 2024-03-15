import cv2

face_classifier = cv2.CascadeClassifier('D:/Image Processing/haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)


    if faces is None:
        return False
    else:
        return True


# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture frame-by-frame
while True:
    ret, frame = cap.read()
    if face_extractor(frame) == True:
        break
    elif cv2.waitKey(0) == 13:
        break

# Save the captured frame as an image file
count = 31
filename = 'Test/test' + str(count) + '.jpg'
cv2.imshow('Captured Image', frame)
cv2.waitKey(0)
cv2.imwrite(filename, frame)

# Release the webcam
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()
