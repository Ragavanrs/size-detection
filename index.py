import cv2
import numpy as np

# Load the captured image
image = cv2.imread('captured_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use HoughCircles to detect circles in the image
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=30, minRadius=5, maxRadius=50)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Draw the detected circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)

# Show the output image
cv2.imshow('Image with Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
