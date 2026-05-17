# debug_masks.py — muestra las máscaras de color puras
import cv2
import numpy as np

cap = cv2.VideoCapture("/home/victor/videos/videobueno.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
ret, frame = cap.read()
cap.release()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
hsv[:,:,2] = clahe.apply(hsv[:,:,2])

amarillo = cv2.inRange(hsv, np.array([15, 30,  80]),  np.array([40, 255, 255]))
blanco = cv2.inRange(hsv, np.array([0,  0,  220]), np.array([180, 30,  255]))

cv2.imshow("Original",  frame)
cv2.imshow("Amarillo",  amarillo)
cv2.imshow("Blanco",    blanco)
cv2.waitKey(0)
cv2.destroyAllWindows()