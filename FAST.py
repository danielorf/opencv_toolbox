import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread(r'images\simple.jpg',0)

fast = cv2.FastFeatureDetector_create()

kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img,kp,None,color=(255,0,0))

print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

cv2.imshow('img',img2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()



fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,255))

cv2.imshow('img',img3)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()