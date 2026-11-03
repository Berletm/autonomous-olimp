import cv2
import os
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
import numpy as np

def nothing(*arg):
    pass


imfile = "obs.png"

img = cv2.imread(imfile)


cv2.namedWindow("img")

cv2.createTrackbar('Hl', 'img', 0, 255, nothing)
cv2.createTrackbar('Sl', 'img', 0, 255, nothing)
cv2.createTrackbar('Vl', 'img', 0, 255, nothing)
cv2.createTrackbar('Hh', 'img', 255, 255, nothing)
cv2.createTrackbar('Sh', 'img', 255, 255, nothing)
cv2.createTrackbar('Vh', 'img', 255, 255, nothing)


while True:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    h1 = cv2.getTrackbarPos('Hl', 'img')
    s1 = cv2.getTrackbarPos('Sl', 'img')
    v1 = cv2.getTrackbarPos('Vl', 'img')

    h2 = cv2.getTrackbarPos('Hh', 'img')
    s2 = cv2.getTrackbarPos('Sh', 'img')
    v2 = cv2.getTrackbarPos('Vh', 'img')

    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    thresh = cv2.inRange(hsv, h_min, h_max)

    kernel = np.ones((15, 15), np.uint8)
        
    mask = cv2.bitwise_and(img, img, mask=thresh)
    
    cv2.imshow('img', mask)
    
 
    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()
