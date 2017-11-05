import cv2
import matplotlib.pyplot as plt

img = cv2.imread('bfly_m.png',0)

surf = cv2.xfeatures2d.SURF_create(400)

a, b = surf.detectAndCompute(img,None)

print(len(a))

print(surf.getHessianThreshold())

surf.setHessianThreshold(60000)

a, b = surf.detectAndCompute(img,None)

print(len(a))

img2 = cv2.drawKeypoints(img,a,None,(255,0,0),4)

plt.imshow(img2),plt.show()