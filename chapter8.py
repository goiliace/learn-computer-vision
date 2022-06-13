import numpy as np
import cv2
from StackImages import stackImages

img = cv2.imread('Resources/Photos/shapes.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgCan = cv2.Canny(img, 100, 200)
imgBlack = np.zeros_like(img)
imgContour = img.copy()
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        cv2.drawContours(imgContour, contour, -1, (255, 0, 0), 3)
        if area > 500:
            cv2.drawContours(imgContour, contour, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(contour, True)
            print(peri)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            objContour = len(approx)
            if objContour == 3:
                objectType = "Triangle"
            elif objContour == 4:
                aspRatio = w/float(h)
                print(aspRatio)
                if aspRatio > 0.8 and aspRatio < 1.2:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objContour > 4:
                objectType = "Circle"
            else:
                objectType = "None"
            cv2.putText(imgContour, objectType, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
    return imgContour

imgContour =  getContours(imgCan)
imgStack = stackImages(0.7, ([img, imgGray, imgBlur], [imgCan, imgContour, imgBlack]))
cv2.imshow("img",imgStack)
cv2.waitKey(0)