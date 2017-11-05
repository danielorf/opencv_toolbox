import cv2
import numpy as np

img1 = cv2.imread(r'images\star.jpg',0)
img2 = cv2.imread(r'images\matchshapes.jpg',0)

ret, thresh = cv2.threshold(img1, 127, 255, 0)
ret, thresh2 = cv2.threshold(img2, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, 2, 1)
cnt1 = contours[0]
im2, contours, hierarchy = cv2.findContours(thresh2, 2, 1)

cnt2 = contours[8]

ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
print(ret)

print(hierarchy)


img3 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img3,cnt2,-1,(255,100,0), 2)

cv2.imshow('img',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()