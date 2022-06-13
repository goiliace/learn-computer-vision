import cv2
import numpy as np

# def resize(img, scale,inter=cv2.INTER_AREA):
#     (h,w) = img.shape[:2]
#     return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=inter)

car_cascade = cv2.CascadeClassifier('Resources/Haarcascades/cars.xml')

# img = cv2.imread("Resources/Photos/car.png",1)
# img = resize(img, 0.5)
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cars = car_cascade.detectMultiScale(imgGray, 1.1, 1)

capture = cv2.VideoCapture("Resources/Videos/Data 03_Highway Cars.mp4")
while True:
    success, img = capture.read()
    img = cv2.resize(img, (640, 480))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(imgGray, 1.1, 1)
    for (x,y,w,h) in cars:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.imshow("Car Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)