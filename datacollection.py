
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import time 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        # Check if imgCrop is valid and non-empty
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                hcal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (hcal, imgSize))
            else:
                k = imgSize / w
                wcal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, wcal))

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Calculate the center position to paste imgResize into imgWhite
            y1, y2 = (imgSize - imgResize.shape[0]) // 2, (imgSize - imgResize.shape[0]) // 2 + imgResize.shape[0]

            # Calculate the gap width on both sides
            wGap = (imgSize - imgResize.shape[1]) // 2

            # Copy imgResize into imgWhite with a gap in between
            imgWhite[y1:y2, wGap:wGap + imgResize.shape[1]] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("image", img)

    # Check for 'S' key press to save the image
    key = cv2.waitKey(1)
    if key == ord("S") or key == ord("s"):
        filename = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(filename, imgWhite)
        counter += 1
        print(f"Image saved as '{filename}'")

    # Check for key press to exit the loop
    if key == 27:  # Press 'Esc' key to exit
        break 

cap.release()
cv2.destroyAllWindows()
