import cv2
import numpy as np

cat_cascase = cv2.CascadeClassifier('Resources/haarcascades/haarcascade_frontalcatface_extended.xml')
img = cv2.imread("Resources/Photos/catmeme.jpg")
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cats = cat_cascase.detectMultiScale(imgGray, 1.1, 5)
for (x, y, w, h) in cats:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("Cat Detection", img)
cv2.waitKey(0)