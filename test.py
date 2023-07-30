
 



import cv2
import numpy as np
import math
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the hand detector
detector = HandDetector(maxHands=1)

# File paths for the model and labels
model_file = "keras_model.h5"
labels_file = "labels.txt"

# Get the absolute paths for the model and labels
model_path = os.path.abspath(model_file)
labels_path = os.path.abspath(labels_file)

# Create your custom classifier
customClassifier = Classifier(model_path, labels_path)

# Offset and image size for cropping and resizing the hand region
offset = 20
imgSize = 300
labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()

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
                hcal = int(k * h)
                imgResize = cv2.resize(imgCrop, (int(w * k), imgSize), interpolation=cv2.INTER_AREA)
            else:
                k = imgSize / w
                wcal = int(k * w)
                imgResize = cv2.resize(imgCrop, (imgSize, int(h * k)), interpolation=cv2.INTER_AREA)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Calculate the center position to paste imgResize into imgWhite
            y1, y2 = (imgSize - imgResize.shape[0]) // 2, (imgSize - imgResize.shape[0]) // 2 + imgResize.shape[0]
            y1, y2 = int(y1), int(y2)

            # Calculate the gap width on both sides
            wGap = (imgSize - imgResize.shape[1]) // 2
            wGap = int(wGap)

            # Copy imgResize into imgWhite with a gap in between
            imgWhite[y1:y2, wGap:wGap + imgResize.shape[1]] = imgResize

            # Get the letter prediction from your custom classifier
            prediction, index = customClassifier.getPrediction(imgWhite)
            print(prediction, index)

            # Draw the recognized letter on the output image
            cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255, 2))

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("image", imgOutput)

    # Check for key press to exit the loop
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()
