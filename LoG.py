import cv2
import numpy as np
#import matplotlib.pyplot as plt

img = cv2.imread(r'images\bfly_c.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img1 = cv2.GaussianBlur(img,(7,7),0)
laplacian1 = cv2.Laplacian(img,cv2.CV_64F,ksize=3,scale=1)

img2 = cv2.GaussianBlur(img,(5,5),0)
laplacian2 = cv2.Laplacian(img2,cv2.CV_64F,ksize=3,scale=1)

img_diff = img1-img2
img_diff = cv2.convertScaleAbs(img_diff)

cv2.imshow('img',img_diff)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()