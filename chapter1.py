import cv2
## READ IMAGE
# LOAD AN IMAGE USING 'IMREAD'
img = cv2.imread("Resources/Photos/cat.jpg")
# DISPLAY
cv2.imshow("Cats", img)
cv2.waitKey(0)

# READ VIDEO
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("Resources/Videos/dog.mp4") # 0 is the webcam
while True:
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
