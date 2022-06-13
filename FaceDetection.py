import cv2
import numpy as np
from StackImages import  stackImages

face_cascase = cv2.CascadeClassifier('Resources/Haarcascades/haarcascade_frontalface_alt2.xml')

capture = cv2.VideoCapture(0)
while True:
    success, img = capture.read()
    img = cv2.resize(img, (640, 480))
    imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(img, 250, 300)
    imgDialation = cv2.dilate(imgCanny, np.ones((5, 5), np.uint8), iterations=1)
    imgEroded = cv2.erode(imgDialation, np.ones((5, 5), np.uint8), iterations=1)
    faces = face_cascase.detectMultiScale(imgGray, 1.1, 1)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    imgStack = stackImages(0.6, ([img, imgGray, imgBlur], [imgCanny, imgDialation, imgEroded]))
    cv2.imshow("Face Detection", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
print(face_cascase.__doc__)