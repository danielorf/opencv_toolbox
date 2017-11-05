import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r'images\sudoku.jpg',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)#,ksize=7)
sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=5)

# cv2.imshow('img',sobel)
plt.imshow(sobel,cmap = 'gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()