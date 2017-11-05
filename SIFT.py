import cv2
import numpy as np

img = cv2.imread(r'images\home.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
#kp = sift.detect(gray,None)
kp, des = sift.detectAndCompute(gray,None)

print(len(kp))
print(type(kp))
print(type(kp[0]))
print(kp[0].class_id)
print()
print(len(des))
print(type(des))
print(type(des[0]))
print(des[0])

img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('img',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()