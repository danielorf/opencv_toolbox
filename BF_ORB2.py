import numpy as np
import cv2

img1 = cv2.imread(r'images\mccaw.png',0)
#img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY,1)
img2 = cv2.imread(r'images\mccaws_90.jpg',0)
#img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY,1)

orb = cv2.ORB_create()

kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# print (type(matches[0][1].distance))
# print (matches[0][1].trainIdx)

good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = img1.copy()
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img3,flags=2)



cv2.imshow('img3',img3)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()