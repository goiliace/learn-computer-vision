import cv2
from StackImages import stackImages

img = cv2.imread('Resources/Photos/lena.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgStack = stackImages(0.5, ([img, imgGray, img], [img, img, img]))

# imgHor = np.hstack((img,img))
# imgVer = np.vstack((img,img))
#
# cv2.imshow("Horizontal",imgHor)
# cv2.imshow("Vertical",imgVer)
cv2.imshow("ImageStack", imgStack)

cv2.waitKey(0)