import cv2
import numpy as np

img = cv2.imread(r'images\sudoku.jpg',0)

img = cv2.medianBlur(img,5)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

thresh2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,5)
thresh3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

blur = cv2.GaussianBlur(img,(5,5),0)
ret3,thresh_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('img',thresh_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()