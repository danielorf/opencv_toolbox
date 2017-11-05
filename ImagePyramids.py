import cv2
import numpy as np
import urllib.request

# img = cv2.imread(r'images\sudoku.jpg',0)
#
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(img))))))

img = cv2.imread(r'images\mccaws.jpg')

# req = urllib.request.urlopen('https:')
# arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
# img = cv2.imdecode(arr,-1)
#
# img = cv2.pyrDown(cv2.pyrDown(img))
#
img = cv2.pyrMeanShiftFiltering(img,55,21)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
