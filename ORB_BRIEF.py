import numpy as np
import cv2
# from matplotlib import pyplot as plt

img = cv2.imread(r'images\simple.jpg',0)

orb = cv2.ORB_create()

kp = orb.detect(img,None)

kp,des = orb.compute(img,kp)

img2 = cv2.drawKeypoints(img,kp,None,color=(255,0,0),flags=0)

cv2.imshow('img2',img2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()