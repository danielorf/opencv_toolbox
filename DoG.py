import cv2
import numpy as np

img = cv2.imread(r'images\bfly_c.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img1 = cv2.GaussianBlur(img,(15,15),2)
img2 = cv2.GaussianBlur(img,(7,7),0)

img_diff = img1-img2
img_diff = cv2.convertScaleAbs(img_diff)

cv2.imshow('img',img_diff)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()