import numpy as np
import cv2


img = cv2.imread(r'images\simple.jpg',0)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY,1)

star = cv2.xfeatures2d.StarDetector_create()

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

kp = star.detect(img,None)

kp, des = brief.compute(img, kp)

print(brief.descriptorSize())
print(des)