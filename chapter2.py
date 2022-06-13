import cv2
import numpy as np
import StackImages
import matplotlib.pyplot as plt
img = cv2.imread("Resources/Photos/cats.jpg")
kernel = np.ones((5, 5), np.uint8)

#BASIC FUNCTIONS

imgGray = cv2.cvtColor(img, 6)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgMedian = cv2.medianBlur(imgGray, 7)
imgCanny = cv2.Canny(img, 250, 300)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

imgStack = np.hstack([imgGray,imgBlur,imgMedian])
cv2.imshow("Face Detection", imgStack)
print(cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)



# plt.imshow(imgRGB)
# plt.title("Original Image")
# plt.show()

# (b,g,r) = cv2.split(img)
# img = cv2.merge((r,g,b))
# plt.imshow(img)
# plt.title("Original Image")
# plt.show()